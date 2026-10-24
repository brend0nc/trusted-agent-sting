"""Conservative request bounds from a saved public OpenRouter endpoint catalogue.

Rates in manifests are USD per token. OpenRouter max_price uses USD per million.
The bound covers a full endpoint context, output, cache prices and request fees.
It assumes the provider respects its published context and billing contract.
It is not an invoice guarantee, and does not cover other clients or credit fees.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone, timedelta
from decimal import Decimal, InvalidOperation
import json
from pathlib import Path
from urllib.request import urlopen

from delegated_influence import costs, models, task
from delegated_influence.journal import RunPaused


PRICE_KEYS = ('prompt', 'completion', 'input_cache_read', 'input_cache_write', 'request')


class MissingUsageEvidence(RunPaused):
    """Pause for missing provider evidence without declaring a contract violation."""


def amount(value, label):
    try:
        if isinstance(value, bool):
            raise ValueError()
        number = Decimal(str(value))
        if not number.is_finite() or number < 0:
            raise ValueError()
        return number
    except (InvalidOperation, ValueError):
        raise ValueError(f'{label} must be a finite nonnegative amount') from None


def positive_int(value, label):
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f'{label} must be a positive integer')
    return value


def price_ceiling(pricing):
    """Cover every published long-context price tier; reject unknown charge types."""
    for field in ('prompt', 'completion'):
        if field not in pricing:
            raise ValueError(f'missing {field} price')
    rows = [pricing, *pricing.get('overrides', [])]
    result = {}
    for row in rows:
        for field, value in row.items():
            if field in (*PRICE_KEYS, 'internal_reasoning'):
                result[field] = max(result.get(field, Decimal(0)), amount(value, field))
            elif field not in {'overrides', 'min_prompt_tokens', 'max_prompt_tokens', 'discount',
                                'web_search', 'image', 'audio', 'audio_output', 'image_output',
                                'input_audio_cache', 'input_cache_write_1h'}:
                if amount(value, field):
                    raise ValueError(f'unsupported price field: {field}')
    return {key: str(result.get(key, Decimal(0))) for key in
            (*PRICE_KEYS, *(('internal_reasoning',) if 'internal_reasoning' in result else ()))}


class RequestBound:
    def __init__(self, manifest, model, model_config=None):
        if manifest.get('schema_version') != 1:
            raise ValueError('unknown spending manifest schema')
        verified = datetime.fromisoformat(manifest['verified_at'].replace('Z', '+00:00'))
        age = datetime.now(timezone.utc) - verified
        if not timedelta(0) <= age <= timedelta(days=1):
            raise ValueError('spending manifest must be fresh (at most 24 hours old)')
        row = manifest.get('models', {}).get(model)
        if not row:
            raise ValueError(f'missing spending route for {model}')
        self._configure(row, task.sampling_instrument(model, model_config))

    @classmethod
    def from_recorded(cls, row, instrument):
        """Validate recovery against saved controls, without authorizing a new request.

        Recovery must not consult a changed model registry or require a fresh
        catalogue. Normal construction still checks both before new dispatch.
        """
        if not isinstance(row, dict) or not isinstance(instrument, dict):
            raise ValueError('recovery needs saved route and sampling evidence')
        if not {'model_spec', 'max_tokens', 'reasoning_effort', 'temperature'}.issubset(instrument):
            raise ValueError('recovery has incomplete saved sampling evidence')
        bound = cls.__new__(cls)
        bound._configure(row, instrument)
        return bound

    def _configure(self, row, instrument):
        self.instrument = instrument
        if row['model'] != self.instrument['model_spec']:
            raise ValueError('spending route model differs from sampling evidence')
        self.row = row
        self.context = positive_int(row['context_length'], 'context_length')
        self.output = positive_int(self.instrument['max_tokens'], 'max_tokens')
        declared = positive_int(row['max_completion_tokens'], 'max_completion_tokens')
        if self.output > declared:
            raise ValueError('output limit exceeds provider support')
        # Declared output maxima are not enforced everywhere: deepseek-r1 at novita returned 19,460
        # reasoning tokens against 16,000, qwen3-235b-thinking at alibaba 81,920 against 8,000
        # (2026-09-10). The context window is the one ceiling no reply can exceed, so token checks
        # and the dollar reservation use it (or the declared maximum, if a route declares it larger).
        self.max_completion = max(self.context, declared)
        if not row.get('provider_tag') or not row.get('provider_name') or not row.get('evidence'):
            raise ValueError('spending route needs provider identity and evidence')
        rates = {key: amount(row['price_ceiling'][key], key) for key in PRICE_KEYS
                 if key in row['price_ceiling']}
        for key in PRICE_KEYS:
            if key not in rates:
                raise ValueError(f'missing {key} ceiling')
        # Cache prices are full per-token prices. Normalized input buckets are
        # disjoint, and check_usage verifies raw prompt totals before release.
        # Reserve for the context window, not the requested max_tokens: a bill above the
        # reservation stops the whole ledger permanently.
        self.maximum_usd = (self.context * max(rates[k] for k in
                            ('prompt', 'input_cache_read', 'input_cache_write'))
                            + self.max_completion * (rates['completion'] + amount(
                                row['price_ceiling'].get('internal_reasoning', 0), 'internal_reasoning'))
                            + rates['request'])
        self.rates = rates
        # A direct route reaches one provider by construction, so there is nothing to pin and no
        # OpenRouter price filter to send. Everything else keeps its pinned route and ceiling.
        self.direct = bool(row.get('direct'))
        if self.direct:
            self.routing = None
            if (self.instrument.get('provider_routing') or {}).get('only'):
                raise ValueError('a direct route cannot also pin an OpenRouter provider')
        else:
            routing_rates = {key: amount(value, key) for key, value in
                             row.get('routing_price_ceiling', row['price_ceiling']).items()}
            self.routing = {'only': [row['provider_tag']], 'allow_fallbacks': False,
                            'require_parameters': True, 'max_price': {
                                'prompt': float(routing_rates['prompt'] * 1_000_000),
                                'completion': float(routing_rates['completion'] * 1_000_000),
                                'request': float(routing_rates['request'])}}
            original = self.instrument.get('provider_routing') or {}
            if original.get('only') and original['only'] not in ([row['provider_tag']], [row['provider_name']]):
                raise ValueError('spending provider differs from experiment provider')
        self.effort = self.instrument['reasoning_effort']
        supported = row.get('supported_parameters', [])
        if not {'max_tokens', 'max_completion_tokens'}.intersection(supported):
            raise ValueError('provider must support an output limit')
        if self.effort is not None and not {'reasoning', 'reasoning_effort'}.intersection(supported):
            raise ValueError('provider does not support reasoning configuration')
        reasoning = row.get('reasoning') or {}
        efforts = reasoning.get('supported_efforts')
        effort = 'xhigh' if self.effort == 'max' else self.effort
        if effort is not None and efforts is not None and effort not in efforts:
            raise ValueError(f'provider does not advertise reasoning effort {effort}')
        if self.effort == 'none' and (reasoning.get('mandatory') or not efforts or 'none' not in efforts):
            raise ValueError('reasoning none requires explicit advertised support')

    def service_model_name(self):
        """The model id the provider itself is sent: OpenRouter keeps the slug, a direct route drops
        its provider prefix (openai/gpt-5.4 -> gpt-5.4, anthropic/claude-opus-4-8 -> …-4-8)."""
        spec = self.row['model']
        return spec.partition('/')[2] if self.direct else spec.removeprefix('openrouter/')

    def _check_openrouter_request(self, body):
        allowed = {'model','provider','messages','max_tokens','max_completion_tokens','seed',
                   'reasoning','temperature','tools','tool_choice','n','stream','stream_options','usage'}
        if set(body) - allowed:
            raise RunPaused(f'unbounded request fields: {sorted(set(body) - allowed)}')
        if body.get('model') != self.service_model_name():
            raise RunPaused('request model differs from reserved route')
        if body.get('provider') != self.routing:
            raise RunPaused('request provider controls differ from reservation')
        limits = [body[key] for key in ('max_tokens', 'max_completion_tokens') if key in body]
        if not limits or any(isinstance(limit, bool) or limit != self.output for limit in limits):
            raise RunPaused('request output limit differs from reservation')
        if body.get('n', 1) != 1 or any(body.get(k) for k in ('plugins', 'tools', 'models', 'transforms')):
            raise RunPaused('unbounded request feature')
        if 'temperature' in body and self.instrument['temperature'] is None:
            raise RunPaused('request unexpectedly sets temperature')
        expected = 'xhigh' if self.effort == 'max' else self.effort
        if (body.get('reasoning') or {}).get('effort') != expected:
            raise RunPaused('request reasoning effort differs from instrument')

    # Fields each direct API may carry. Anything else is unbounded until it is priced and checked.
    DIRECT_FIELDS = {
        'openai': {'model','messages','max_completion_tokens','max_tokens','reasoning_effort',
                   'temperature','n','stream','stream_options','seed'},
        'anthropic': {'model','messages','system','max_tokens','stream','tools','tool_choice',
                      'cache_control','thinking','output_config','stop_sequences'},
    }

    def _check_direct_request(self, body):
        provider = self.row['model'].partition('/')[0]
        allowed = self.DIRECT_FIELDS.get(provider)
        if allowed is None:
            raise RunPaused(f'no direct request bounds for {provider}')
        if set(body) - allowed:
            raise RunPaused(f'unbounded request fields: {sorted(set(body) - allowed)}')
        if body.get('model') != self.service_model_name():
            raise RunPaused('request model differs from reserved route')
        limits = [body[key] for key in ('max_tokens', 'max_completion_tokens') if key in body]
        if not limits or any(isinstance(limit, bool) or limit != self.output for limit in limits):
            raise RunPaused('request output limit differs from reservation')
        # Anthropic's SDK streams whenever the output budget is large enough to risk its
        # long-request limit; the reply and its usage still arrive in one HTTP exchange.
        streamed_ok = provider == 'anthropic'
        if (body.get('n', 1) != 1 or (body.get('stream') and not streamed_ok)
                or body.get('tools') or body.get('tool_choice')):
            raise RunPaused('unbounded request feature')
        expected_temperature = self.instrument['temperature']
        if body.get('temperature') != expected_temperature and not (
                expected_temperature is None and 'temperature' not in body):
            raise RunPaused('request temperature differs from instrument')
        expected = 'xhigh' if self.effort == 'max' else self.effort
        # OpenAI names it reasoning_effort on chat completions; Anthropic carries a thinking block.
        if provider == 'openai':
            if body.get('reasoning_effort') != expected:
                raise RunPaused('request reasoning effort differs from instrument')
        # Anthropic states the effort in output_config and opens a thinking block alongside it.
        elif (body.get('output_config') or {}).get('effort') != expected:
            raise RunPaused('request reasoning effort differs from instrument')

    def check_request(self, raw):
        """Called synchronously before the HTTP client sends the exact body."""
        body = json.loads(raw)
        if self.direct:
            self._check_direct_request(body)
        else:
            self._check_openrouter_request(body)
        for message in body.get('messages', []):
            content = message.get('content')
            if isinstance(content, list) and any(block.get('type') != 'text' for block in content):
                raise RunPaused('spending bounds cover text messages only')
            for block in [message, *(content if isinstance(content, list) else [])]:
                cache = block.get('cache_control') or {}
                if cache and (cache.get('type') != 'ephemeral' or cache.get('ttl', '5m') != '5m'):
                    raise RunPaused('spending bounds cover default five-minute cache writes only')

    def check_usage(self, usage):
        if usage.get('served_provider') in (None, ''):
            raise MissingUsageEvidence('provider identity is missing from saved usage evidence')
        if usage.get('served_provider') != self.row['provider_name']:
            raise RunPaused('served provider differs from reserved provider')
        # Providers overshoot max_tokens and even their declared maxima; the dollar bound is
        # still enforced at settlement, so tokens are checked against the context window.
        for key, ceiling in (('completion_tokens', self.max_completion), ('prompt_tokens', self.context)):
            value = usage.get(key)
            if value is None:
                raise MissingUsageEvidence(f'provider {key} is missing from saved usage evidence')
            if isinstance(value, bool) or not isinstance(value, int) or value < 0 or value > ceiling:
                raise RunPaused(f'provider {key} outside declared limits')
        for key in ('cache_read_tokens', 'cache_write_tokens', 'reasoning_tokens'):
            value = usage.get(key, 0)
            if value is None:
                raise MissingUsageEvidence(f'provider {key} is missing from saved usage evidence')
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise RunPaused(f'provider {key} is invalid')
        prompt = sum(usage.get(k, 0) or 0 for k in
                     ('prompt_tokens', 'cache_read_tokens', 'cache_write_tokens'))
        if prompt > self.context:
            raise RunPaused('provider total input exceeds reserved context')
        raw = usage.get('provider_usage') or {}
        for row in (raw, raw.get('prompt_tokens_details') or {}, raw.get('completion_tokens_details') or {}):
            for key, value in row.items():
                if 'tokens' in key and not isinstance(value, dict) and value is not None:
                    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                        raise RunPaused(f'raw provider {key} is invalid')
        if raw.get('prompt_tokens') is not None and raw['prompt_tokens'] != prompt:
            raise RunPaused('raw provider prompt total differs from disjoint input buckets')
        if raw.get('completion_tokens') is not None and raw['completion_tokens'] != usage['completion_tokens']:
            raise RunPaused('raw provider completion tokens differ from total completion')
        # DeepSeek R1 via OpenRouter reports reasoning tokens slightly above completion tokens.
        # Keep the raw values; bound reasoning by the reserved output like completion.
        if (usage.get('reasoning_tokens') or 0) > self.max_completion:
            raise RunPaused('provider reasoning_tokens outside declared limits')
        if self.effort == 'none' and (usage.get('reasoning_tokens') or 0) > 0:
            raise RunPaused('provider returned reasoning tokens for effort none')


def direct_price_ceiling(model, rates):
    """Per-token ceiling for a directly served model, from the run config's own price table.

    OpenAI and Anthropic bill tokens, not dollars, so their price table replaces OpenRouter's
    invoice as the basis for the reservation and the settled cost. A missing cache rate falls back
    to the prompt rate, which can only over-reserve.
    """
    if not rates:
        raise ValueError(f'direct route needs cost.model_prices for {model}')
    def per_token(field, default=None):
        value = rates.get(field, default)
        if value is None:
            raise ValueError(f'missing {field} for {model}')
        return Decimal(str(value)) / Decimal(1_000_000)
    prompt = per_token('input_usd_per_mtok')
    return {'prompt': str(prompt), 'completion': str(per_token('output_usd_per_mtok')),
            'input_cache_read': str(per_token('cache_read_usd_per_mtok', rates['input_usd_per_mtok'])),
            'input_cache_write': str(per_token('cache_write_usd_per_mtok', rates['input_usd_per_mtok'])),
            'request': '0'}


def compile_manifest(tasks, fetch=None, cost=None):
    """Read public metadata only. Do not silently substitute unavailable models."""
    if fetch is None:
        def fetch(url):
            with urlopen(url, timeout=30) as response:
                return json.load(response)
    catalog = {row['id']: row for row in fetch('https://openrouter.ai/api/v1/models')['data']}
    price_table = costs.prices(cost or {}) or {}
    instruments = {}
    for _, task_value in tasks:
        for sample in task_value.dataset:
            for model in set(sample.metadata['seat_map'].values()):
                config = (sample.metadata.get('model_configs') or {}).get(model)
                instruments.setdefault(model, []).append(task.sampling_instrument(model, config))
    result = {'schema_version': 1, 'verified_at': datetime.now(timezone.utc).isoformat(), 'models': {}}
    for model, settings in instruments.items():
        spec = settings[0]['model_spec']
        if models.direct_spec(model) == spec:
            # Served by its own provider: public catalogue for the limits, our table for the price.
            slug = models.openrouter_spec(model).removeprefix('openrouter/')
            if slug not in catalog:
                raise ValueError(f'model is absent from current catalogue: {slug}')
            entry = catalog[slug]
            params = set(entry.get('supported_parameters', []))
            if not {'max_tokens', 'max_completion_tokens'}.intersection(params):
                raise ValueError(f'direct route declares no output limit: {model}')
            declared = entry.get('top_provider', {}).get('max_completion_tokens') or entry['context_length']
            if declared < max(s['max_tokens'] for s in settings):
                raise ValueError(f'direct route output limit is below the requested max_tokens: {model}')
            result['models'][model] = {
                'model': spec, 'provider_tag': spec.partition('/')[0],
                'provider_name': spec.partition('/')[0], 'context_length': entry['context_length'],
                'max_completion_tokens': declared,
                'price_ceiling': direct_price_ceiling(model, price_table.get(model)),
                'supported_parameters': ['max_completion_tokens', 'max_tokens',
                                         'reasoning', 'reasoning_effort'],
                'reasoning': entry.get('reasoning') or {},
                'evidence': 'cost.model_prices in the run config; limits from the public catalogue',
                'catalog_model': entry, 'direct': True,
            }
            continue
        if not spec.startswith('openrouter/'):
            raise ValueError(f'live catalogue only supports OpenRouter: {model}')
        slug = spec.removeprefix('openrouter/')
        if slug not in catalog:
            raise ValueError(f'model is absent from current catalogue: {slug}')
        url = f'https://openrouter.ai/api/v1/models/{slug}/endpoints'
        response = fetch(url)
        choices = []
        for endpoint in response['data']['endpoints']:
            if endpoint.get('status', 0) != 0:
                continue
            if any((s.get('provider_routing') or {}).get('only') and
                   endpoint['tag'] not in s['provider_routing']['only'] and
                   endpoint['provider_name'] not in s['provider_routing']['only'] for s in settings):
                continue
            params = set(endpoint.get('supported_parameters', []))
            if not {'max_tokens','max_completion_tokens'}.intersection(params):
                continue
            if any(s['reasoning_effort'] is not None for s in settings) and not {'reasoning','reasoning_effort'}.intersection(params):
                continue
            if any(s['temperature'] is not None for s in settings) and 'temperature' not in params:
                continue
            if (endpoint.get('max_completion_tokens') or 0) < max(s['max_tokens'] for s in settings):
                continue
            try:
                ceiling = price_ceiling(endpoint['pricing'])
            except ValueError:
                continue
            # Prefer the normal route over flex/fast variants. Price breaks ties.
            choices.append((('/' in endpoint['tag']), amount(ceiling['prompt'], 'prompt'), endpoint, ceiling))
        if not choices:
            raise ValueError(f'no eligible bounded provider for {model}')
        _, _, endpoint, ceiling = min(choices, key=lambda choice: choice[:2])
        # OpenRouter base slugs include their /variant routes. Cover every
        # published variant, even one the price filter might currently reject.
        routing_ceiling = ceiling
        matched = [entry for entry in response['data']['endpoints']
                   if entry['tag'] == endpoint['tag'] or
                   ('/' not in endpoint['tag'] and entry['tag'].startswith(endpoint['tag'] + '/')
                    and entry['tag'].split('/')[-1] not in {'fast', 'flex'})]
        ceilings = [price_ceiling(entry['pricing']) for entry in matched]
        keys = set().union(*(row.keys() for row in ceilings))
        ceiling = {key: str(max(amount(row.get(key, 0), key) for row in ceilings)) for key in sorted(keys)}
        result['models'][model] = {
            'model': spec, 'provider_tag': endpoint['tag'], 'provider_name': endpoint['provider_name'],
            'context_length': max(entry['context_length'] for entry in matched),
            'max_completion_tokens': endpoint['max_completion_tokens'], 'price_ceiling': ceiling,
            'routing_price_ceiling': routing_ceiling,
            'supported_parameters': endpoint['supported_parameters'],
            'reasoning': catalog[slug].get('reasoning') or {}, 'evidence': url,
            'catalog_model': catalog[slug], 'endpoint': endpoint, 'matching_endpoints': matched,
        }
    return result


def main():
    from delegated_influence import experiment
    parser = argparse.ArgumentParser(description='Save current public endpoint limits. Sends no model requests.')
    parser.add_argument('config')
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    cfg = experiment.load_config(args.config)
    manifest = compile_manifest(experiment.build_tasks(cfg, config_name=Path(args.config).stem),
                                cost=cfg.get('cost'))
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(manifest, indent=2) + '\n')
    print(f'Saved {len(manifest["models"])} bounded routes to {output}. No model requests sent.')


if __name__ == '__main__':
    main()

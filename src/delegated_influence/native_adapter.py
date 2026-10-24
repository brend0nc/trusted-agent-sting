"""Direct OpenRouter transport, without Inspect scheduling, retries, or cost estimates.

The installed provider API owns request mapping, including prompt cache markers.
`total_cost` comes only from OpenRouter's raw response `usage.cost`. Missing billing
stays unknown. Token accounting uses disjoint input/cache buckets; output includes
reasoning once. No local price catalog participates in this adapter.

`provider_request_body` is the exact UTF-8 HTTP request body after provider mapping.
No HTTP headers or credentials are stored. A provider-library request snapshot is
marked separately when no HTTP request exists, such as offline mock responses.
Callers can set `generate.on_request(body)` to synchronously save the wire body
before dispatch. A callback failure aborts dispatch. Callers must await
`generate.aclose()` when they finish using a generator.
"""
from __future__ import annotations

from contextvars import ContextVar
import math
import time

from inspect_ai.model import get_model

from delegated_influence import models, task


SUPPORTED_PROVIDERS = ("openrouter/", "openai/", "anthropic/", "mockllm/")
# Providers we call directly. They bill tokens, not dollars, so their cost comes from the
# reserved route's own price table instead of the response body.
DIRECT_PROVIDERS = ("openai/", "anthropic/")
CLIENT_TIMEOUT_S = task.CALL_TIMEOUT_S + 60  # HTTP client limit; the call cap (task.CALL_TIMEOUT_S) must fire first



def _number(value):
    return (isinstance(value, (int, float)) and not isinstance(value, bool)
            and math.isfinite(value) and value >= 0)


def _count(value, default=0):
    return value if isinstance(value, int) and not isinstance(value, bool) and value >= 0 else default


def _priced_cost(result, rates):
    """Dollar cost of a direct provider's call from the reserved route's per-token ceiling.

    Reasoning tokens are already inside completion_tokens on both direct APIs, so they are never
    added again. Any missing token count makes the cost unknown rather than wrong.
    """
    buckets = (("prompt_tokens", "prompt"), ("completion_tokens", "completion"),
               ("cache_read_tokens", "input_cache_read"), ("cache_write_tokens", "input_cache_write"))
    total = rates.get("request", 0)
    for field, rate in buckets:
        tokens = result.get(field)
        if not isinstance(tokens, int) or isinstance(tokens, bool) or tokens < 0:
            return None
        total += tokens * rates[rate]
    return float(total)


def _usage(output, call, spec, elapsed, request_bodies, rates=None):
    raw = call.response if call is not None and isinstance(call.response, dict) else {}
    choices = raw.get("choices")
    raw_choice = choices[0] if isinstance(choices, list) and choices and isinstance(choices[0], dict) else {}
    raw_usage = raw.get("usage")
    raw_usage = raw_usage if isinstance(raw_usage, dict) else {}
    prompt_details = raw_usage.get("prompt_tokens_details") or {}
    completion_details = raw_usage.get("completion_tokens_details") or {}
    prompt_details = prompt_details if isinstance(prompt_details, dict) else {}
    completion_details = completion_details if isinstance(completion_details, dict) else {}
    usage = output.usage
    base_usage_present = usage is not None or any(
        raw_usage.get(key) is not None for key in ("prompt_tokens", "completion_tokens"))
    cache_default = 0 if base_usage_present else None
    cache_read = _count(raw_usage.get("cache_read_input_tokens"),
                       _count(prompt_details.get("cached_tokens"),
                              _count(usage.input_tokens_cache_read) if usage else cache_default))
    cache_write = _count(raw_usage.get("cache_creation_input_tokens"),
                        _count(prompt_details.get("cache_write_tokens"),
                               _count(usage.input_tokens_cache_write) if usage else cache_default))
    prompt_total = raw_usage.get("prompt_tokens")
    # Inspect's provider adapter already subtracts cache tokens in ModelUsage.
    # Prefer raw totals; never subtract caches again from its normalized fallback.
    prompt = (max(0, prompt_total - cache_read - cache_write)
              if isinstance(prompt_total, int) and not isinstance(prompt_total, bool)
              else (_count(usage.input_tokens, None) if usage else None))
    cost = raw_usage.get("cost") if spec.startswith("openrouter/") else None
    reported_cost = cost if _number(cost) else None
    cost_source = "openrouter_response_usage_cost" if reported_cost is not None else "unknown"
    result = {
        "prompt_tokens": prompt,
        "completion_tokens": _count(raw_usage.get("completion_tokens"),
                                     _count(usage.output_tokens, None) if usage else None),
        "reasoning_tokens": _count(completion_details.get("reasoning_tokens"),
                                    _count(usage.reasoning_tokens) if usage else cache_default),
        "cache_read_tokens": cache_read,
        "cache_write_tokens": cache_write,
        "total_cost": reported_cost,
        "cost_source": cost_source,
        "latency_seconds": elapsed,
        "served_model": raw.get("model") or output.model,
        # A direct route reaches exactly one host, and it names itself nowhere in the body, so the
        # provider the request was addressed to IS the served provider.
        "served_provider": (spec.partition("/")[0] if spec.startswith(DIRECT_PROVIDERS)
                            else raw.get("provider")),
        "provider_request_id": raw.get("id"),
        "stop_reason": output.choices[0].stop_reason if output.choices else None,
        "finish_reason": raw_choice.get("finish_reason"),
        "native_finish_reason": raw_choice.get("native_finish_reason"),
        "provider_usage": raw_usage,
        "provider_request_source": "http_request_body" if request_bodies else "provider_library_snapshot",
        "provider_request_body": request_bodies[-1] if request_bodies else None,
        "http_requests": len(request_bodies),
    }
    if reported_cost is None and rates is not None and spec.startswith(DIRECT_PROVIDERS):
        priced = _priced_cost(result, rates)
        if priced is not None:
            result["total_cost"] = priced
            result["cost_source"] = "run_config_price_table"
    if not request_bodies and call is not None:
        result["provider_request_snapshot"] = call.request
    if output.error:
        result["error"] = output.error
    return result


# Anthropic's provider asks its free count_tokens endpoint how long a prompt is before generating.
# It bills nothing and generates nothing, so it must not count against the one-request-per-
# reservation rule, which exists to catch a second BILLED generation (2026-09-11).
METERING_PATHS = ('/count_tokens',)


def is_metering_request(request):
    return str(getattr(request, 'url', '')).endswith(METERING_PATHS)


def build_generator(value, seed, seat=None, model_config=None, request_bound=None):
    """Build an async generate(messages) -> (completion, usage) transport.

    OpenRouter, the OpenAI and Anthropic APIs, and offline mockllm are supported. Each owns one
    client; neither the provider SDK nor Inspect may retry its calls. The native
    journal owns failures and recovery. `seat` stays outside the model's prompt.
    """
    spec = task.resolve_model(value)[0]
    if not isinstance(spec, str) or not spec.startswith(SUPPORTED_PROVIDERS):
        raise ValueError("native transport supports OpenRouter, OpenAI, Anthropic and offline mockllm only")
    config = task.build_generate_config(value, seed, model_config)
    config.max_retries = 0
    if request_bound is not None:
        # A direct route has no OpenRouter provider block to send; everything else pins its route.
        if request_bound.routing is not None:
            config.extra_body = {**(config.extra_body or {}), 'provider': request_bound.routing}
        # Game randomness is always seeded locally. Do not send an unsupported
        # sampling parameter that require_parameters would reject upstream.
        if 'seed' not in request_bound.row.get('supported_parameters', []):
            config.seed = None
    # max_retries as a model argument configures AsyncOpenAI, independently of
    # GenerateConfig.max_retries, which only controls Inspect's outer loop.
    # The OpenAI SDK's own read timeout (600 s) fired before the 900 s call cap: fable-5.1 and
    # llama-4-scout replies between 600 and 900 s parked as billing-unknown ReadTimeouts (2026-09-10).
    # Give the HTTP client a longer limit than the call cap, so only the call cap ends a slow call.
    # Give the HTTP client a longer limit than the call cap, so only the call cap ends a slow call.
    # The OpenAI-compatible providers name it client_timeout; the Anthropic SDK takes timeout.
    if spec.startswith(('openrouter/', 'openai/')):
        timeout = {'client_timeout': float(CLIENT_TIMEOUT_S)}
    elif spec.startswith('anthropic/'):
        timeout = {'timeout': float(CLIENT_TIMEOUT_S)}
    else:
        timeout = {}
    # OpenAI's Responses API bills a one-off capability probe with no output limit before the first
    # real call (2026-09-11). One reservation must buy exactly one bounded request, so use chat
    # completions, which sends a single capped request per call.
    if spec.startswith('openai/'):
        timeout['responses_api'] = False
    model = get_model(spec, config=config, max_retries=0, memoize=False, **timeout)
    if request_bound is not None and request_bound.routing is not None:
        # The OpenAI-compatible SDK renames this field for GPT models, but some
        # OpenRouter routes advertise only max_tokens under require_parameters.
        original_params = model.api.completion_params
        supported = request_bound.row.get('supported_parameters', [])

        def completion_params(config, tools):
            params = original_params(config, tools)
            if ('max_completion_tokens' in params and 'max_completion_tokens' not in supported
                    and 'max_tokens' in supported):
                params['max_tokens'] = params.pop('max_completion_tokens')
            return params

        model.api.completion_params = completion_params
    client = getattr(model.api, "client", None)
    request_capture = ContextVar("native_request_capture", default=None)
    response_capture = ContextVar("native_response_capture", default=None)
    if spec.startswith(("openrouter/", *DIRECT_PROVIDERS)):
        if client is None or client.max_retries != 0:
            raise ValueError("native provider client must disable SDK retries")

        async def capture_request(request):
            bodies = request_capture.get()
            if bodies is not None and not is_metering_request(request):
                body = (await request.aread()).decode("utf-8")
                if bodies:
                    from delegated_influence.journal import RunPaused
                    raise RunPaused('native transport attempted more than one HTTP request for one reservation')
                if request_bound is not None:
                    request_bound.check_request(body)
                bodies.append(body)
                if generate.on_request is not None:
                    generate.on_request(body)

        client._client.event_hooks.setdefault("request", []).append(capture_request)

        async def capture_response(response):
            evidence = response_capture.get()
            if evidence is not None:
                evidence.update(status_code=response.status_code,
                                retry_after=response.headers.get('Retry-After'), body_complete=False)
                # Capture before Inspect turns a 200 error body into an exception
                # that drops the HTTP status, generation ID, and any usage bill.
                await response.aread()
                try:
                    evidence['body'] = response.json()
                except ValueError:
                    evidence['body'] = response.text
                evidence['body_complete'] = True

        client._client.event_hooks.setdefault("response", []).append(capture_response)

    async def generate(messages):
        request_bodies = []
        capture_token = request_capture.set(request_bodies)
        response_evidence = {}
        response_token = response_capture.set(response_evidence)
        started = time.monotonic()
        try:
            result = await model.api.generate(
                input=task.to_chat_messages(messages), tools=[], tool_choice="none", config=config,
            )
            output, call = result if isinstance(result, tuple) else (result, None)
            if isinstance(output, Exception):
                raise output
            usage = _usage(output, call, spec, time.monotonic() - started, request_bodies,
                           getattr(request_bound, 'rates', None))
            return (None if output.error else output.completion), usage
        except BaseException as exc:
            exc.native_response = response_evidence
            raise
        finally:
            request_capture.reset(capture_token)
            response_capture.reset(response_token)

    async def aclose():
        if client is not None:
            await client.close()

    generate.on_request = None
    generate.aclose = aclose
    return generate

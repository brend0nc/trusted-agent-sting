"""Explicit price estimates. Provider-reported billing always stays separate.

Inspect's input_tokens excludes cache-read and cache-write tokens. Output tokens
already include reasoning tokens. Count each category once.
"""
from __future__ import annotations

import math

# Original full-ledger smoke estimate: about 491k tokens over 250 turns.
# One source for both planning and live token-price estimates.
DEFAULT_COST = {'avg_tokens_per_turn': 2000, 'usd_per_mtok': 2.0}

RATE_FIELDS = {'input_usd_per_mtok', 'output_usd_per_mtok',
               'cache_read_usd_per_mtok', 'cache_write_usd_per_mtok'}
EXTRA_COST_FIELDS = {'model_prices', 'avg_input_tokens_per_call', 'avg_output_tokens_per_call'}


def number(value, label, *, positive=False):
    if (not isinstance(value, (int, float)) or isinstance(value, bool)
            or not math.isfinite(value) or value < 0 or (positive and value == 0)):
        raise ValueError(f'{label} must be a finite {"positive" if positive else "nonnegative"} number')
    return value


def prices(cost):
    table = cost.get('model_prices')
    if table is None:
        return None
    if not isinstance(table, dict) or not table:
        raise ValueError('cost.model_prices must be a nonempty mapping')
    for model, rates in table.items():
        if not isinstance(rates, dict) or set(rates) - RATE_FIELDS:
            raise ValueError(f'invalid price fields for {model}')
        for required in ('input_usd_per_mtok','output_usd_per_mtok'):
            if required not in rates:
                raise ValueError(f'missing {required} price for {model}')
        for key, value in rates.items():
            number(value, f'cost.model_prices.{model}.{key}')
    return table


def usage_estimate(cost, model, usage):
    """Return an estimate, or None if the chosen price table does not cover it."""
    table = prices(cost)
    if any(usage.get(key) is None for key in ('prompt_tokens','completion_tokens')):
        return None
    for key in ('prompt_tokens','completion_tokens','cache_read_tokens','cache_write_tokens'):
        if key in usage and usage[key] is not None:
            number(usage[key], f'usage.{key}')
    if table is None:
        rate = number(cost.get('usd_per_mtok', DEFAULT_COST['usd_per_mtok']), 'cost.usd_per_mtok', positive=True)
        total = sum(usage.get(key, 0) or 0 for key in (
            'prompt_tokens','completion_tokens','cache_read_tokens','cache_write_tokens'))
        return total * rate / 1e6
    rates = table.get(model)
    if rates is None:
        return None
    total = (usage.get('prompt_tokens', 0) or 0) * rates['input_usd_per_mtok']
    total += (usage.get('completion_tokens', 0) or 0) * rates['output_usd_per_mtok']
    for usage_key, rate_key in (('cache_read_tokens','cache_read_usd_per_mtok'),
                               ('cache_write_tokens','cache_write_usd_per_mtok')):
        tokens = usage.get(usage_key, 0) or 0
        if tokens and rate_key not in rates:
            return None
        total += tokens * rates.get(rate_key, 0)
    return total / 1e6


def planning_call_cost(cost, seat_map):
    table = prices(cost)
    if table is None:
        tokens = number(cost.get('avg_tokens_per_turn', DEFAULT_COST['avg_tokens_per_turn']),
                        'cost.avg_tokens_per_turn', positive=True)
        rate = number(cost.get('usd_per_mtok', DEFAULT_COST['usd_per_mtok']),
                      'cost.usd_per_mtok', positive=True)
        return tokens * rate / 1e6
    for key in ('avg_input_tokens_per_call','avg_output_tokens_per_call'):
        if key not in cost:
            raise ValueError(f'per-model pricing requires cost.{key}')
        number(cost[key], f'cost.{key}', positive=True)
    missing = set(seat_map.values()) - set(table)
    if missing:
        raise ValueError(f'missing model prices: {sorted(missing)}')
    return sum((cost['avg_input_tokens_per_call'] * table[model]['input_usd_per_mtok']
                + cost['avg_output_tokens_per_call'] * table[model]['output_usd_per_mtok']) / 1e6
               for model in seat_map.values()) / len(seat_map)

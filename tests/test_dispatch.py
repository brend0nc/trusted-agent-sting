import json
from datetime import datetime, timezone

import pytest

from delegated_influence import dispatch
from delegated_influence.journal import RunPaused


def manifest():
    return {'schema_version': 1, 'verified_at': datetime.now(timezone.utc).isoformat(),
            'models': {'gpt-5.5': {'model': 'openrouter/openai/gpt-5.5',
                'provider_tag': 'openai', 'provider_name': 'OpenAI',
                'context_length': 1000, 'max_completion_tokens': 100,
                'price_ceiling': {'prompt': '0.00001', 'completion': '0.00003',
                                  'input_cache_read': '0.000001', 'input_cache_write': '0', 'request': '0.01'},
                'supported_parameters': ['max_tokens','seed','reasoning'],
                'reasoning': {'supported_efforts': ['none','medium']}, 'evidence': 'https://openrouter.ai/api/v1/models/openai/gpt-5.5/endpoints'}}}


def test_bound_covers_full_context_output_cache_and_request_fee():
    rule = dispatch.RequestBound(manifest(), 'gpt-5.5', {'max_tokens': 100, 'reasoning_effort': 'none'})
    assert float(rule.maximum_usd) == pytest.approx(.05)  # 1000 ctx * (1e-5 prompt + 3e-5 completion) + .01 request
    assert rule.routing['max_price'] == {'prompt': 10.0, 'completion': 30.0, 'request': .01}
    assert rule.routing['only'] == ['openai']
    assert rule.routing['allow_fallbacks'] is False


def test_request_guard_rejects_instrument_drift_before_http():
    rule = dispatch.RequestBound(manifest(), 'gpt-5.5', {'max_tokens': 100, 'reasoning_effort': 'none'})
    body = {'model':'openai/gpt-5.5','provider':rule.routing,'max_tokens':100,
            'seed':7,'reasoning':{'effort':'none'},'messages':[{'role':'user','content':'hello'}]}
    rule.check_request(json.dumps(body))
    for patch in ({'max_tokens':101}, {'provider':{}}, {'plugins':[{'id':'web'}]},
                  {'model':'another/model'}, {'tools':[{'type':'function'}]}, {'n':2},
                  {'max_completion_tokens':1000000}, {'web_search_options':{}}, {'modalities':['text','image']}):
        with pytest.raises(RunPaused):
            rule.check_request(json.dumps({**body, **patch}))


def test_manifest_must_be_fresh_and_complete():
    value=manifest(); value['verified_at']='2020-01-01T00:00:00+00:00'
    with pytest.raises(ValueError, match='fresh'):
        dispatch.RequestBound(value, 'gpt-5.5', {'max_tokens':100})
    value=manifest(); del value['models']['gpt-5.5']['price_ceiling']['completion']
    with pytest.raises(ValueError, match='completion'):
        dispatch.RequestBound(value, 'gpt-5.5', {'max_tokens':100})


def test_price_tiers_use_maximum_not_catalogue_base_rate():
    endpoint={'pricing': {'prompt':'0.000005','completion':'0.00003',
                         'input_cache_read':'0.0000005', 'overrides':[
                             {'min_prompt_tokens':272000,'prompt':'0.00001','completion':'0.000045'}]}}
    ceiling=dispatch.price_ceiling(endpoint['pricing'])
    assert ceiling['prompt']=='0.00001'
    assert ceiling['completion']=='0.000045'
    with pytest.raises(ValueError, match='unsupported'):
        dispatch.price_ceiling({'prompt':'0','completion':'0','mystery_fee':'.1'})
    assert dispatch.price_ceiling({'prompt':'0','completion':'0','input_audio_cache':'.1'})['prompt'] == '0'


def test_usage_above_declared_limits_stops_even_below_dollar_bound():
    rule=dispatch.RequestBound(manifest(), 'gpt-5.5', {'max_tokens':100})
    with pytest.raises(RunPaused, match='completion'):
        rule.check_usage({'served_provider':'OpenAI','prompt_tokens':1,'completion_tokens':1001})  # beyond the context window
    with pytest.raises(RunPaused, match='provider'):
        rule.check_usage({'served_provider':'Azure','prompt_tokens':1,'completion_tokens':1})


def test_dollar_bound_reserves_the_context_window_not_the_requested_output():
    data = manifest(); data['models']['gpt-5.5']['max_completion_tokens'] = 500
    rule = dispatch.RequestBound(data, 'gpt-5.5', {'max_tokens': 100, 'reasoning_effort': 'none'})
    # 1000 ctx * 1e-5 + 1000 ctx * 3e-5 + 0.01 request; neither max_tokens nor the declared maximum caps the bill.
    assert float(rule.maximum_usd) == pytest.approx(.05)


def test_token_overshoot_within_context_window_is_kept_and_bounded():
    rule = dispatch.RequestBound(manifest(), 'gpt-5.5', {'max_tokens': 50})
    # Providers overshoot max_tokens and their declared maxima (qwen3-235b-thinking, deepseek-r1);
    # DeepSeek R1 also reports reasoning above completion. Only the context window is a hard ceiling.
    rule.check_usage({'served_provider': 'OpenAI', 'prompt_tokens': 1, 'completion_tokens': 800, 'reasoning_tokens': 900})
    with pytest.raises(RunPaused, match='completion_tokens'):
        rule.check_usage({'served_provider': 'OpenAI', 'prompt_tokens': 1, 'completion_tokens': 1001})
    with pytest.raises(RunPaused, match='reasoning_tokens'):
        rule.check_usage({'served_provider': 'OpenAI', 'prompt_tokens': 1, 'completion_tokens': 80, 'reasoning_tokens': 1001})


def test_published_internal_reasoning_charge_has_an_output_bound():
    data=manifest()
    pricing=dispatch.price_ceiling({'prompt':'.00001','completion':'.00003','internal_reasoning':'.00003'})
    data['models']['gpt-5.5']['price_ceiling']=pricing
    rule=dispatch.RequestBound(data, 'gpt-5.5', {'max_tokens':100})
    assert float(rule.maximum_usd) == pytest.approx(.07)  # 1000 ctx * (1e-5 + 3e-5 completion + 3e-5 internal reasoning)
    # Reserve a possible separate reasoning charge without raising the route's completion price filter.
    assert rule.routing['max_price']['completion'] == 30

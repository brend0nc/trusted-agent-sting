"""Direct OpenAI/Anthropic routes: the spending bound, the request guard, and the priced cost.

These models bill tokens, not dollars, so the run config's price table replaces the provider
invoice. Every check here protects that substitution: the reservation, the single bounded request,
and the settled amount.
"""
import pytest

from delegated_influence import dispatch, models, native_adapter, task
from delegated_influence.journal import RunPaused

TABLE = {
    'gpt-5.4-nano': {'input_usd_per_mtok': 0.2, 'output_usd_per_mtok': 1.25,
                     'cache_read_usd_per_mtok': 0.02, 'cache_write_usd_per_mtok': 0},
    'sonnet-5': {'input_usd_per_mtok': 2.0, 'output_usd_per_mtok': 10.0,
                 'cache_read_usd_per_mtok': 0.2, 'cache_write_usd_per_mtok': 2.5},
}


def bound(key, **overrides):
    spec = models.direct_spec(key)
    row = {'model': spec, 'provider_tag': spec.split('/')[0], 'provider_name': spec.split('/')[0],
           'context_length': 200_000, 'max_completion_tokens': 64_000,
           'price_ceiling': dispatch.direct_price_ceiling(key, TABLE[key]),
           'supported_parameters': ['max_tokens', 'max_completion_tokens'],
           'evidence': 'test', 'direct': True, **overrides}
    task.set_direct_models([key])
    return dispatch.RequestBound.from_recorded(row, task.sampling_instrument(key))


def teardown_function():
    task.set_direct_models([])


def test_direct_spec_covers_openai_and_anthropic_only():
    assert models.direct_spec('gpt-5.4-nano') == 'openai/gpt-5.4-nano'
    assert models.direct_spec('opus-4.8') == 'anthropic/claude-opus-4-8'
    assert models.direct_spec('gemini-pro') is None


def test_price_ceiling_is_per_token_and_covers_every_bucket():
    ceiling = dispatch.direct_price_ceiling('sonnet-5', TABLE['sonnet-5'])
    assert ceiling == {'prompt': '0.000002', 'completion': '0.00001', 'input_cache_read': '2E-7',
                       'input_cache_write': '0.0000025', 'request': '0'}
    with pytest.raises(ValueError, match='cost.model_prices'):
        dispatch.direct_price_ceiling('sonnet-5', None)


def test_direct_route_reserves_the_context_window_and_pins_nothing():
    reserved = bound('sonnet-5')
    assert reserved.direct is True
    assert reserved.routing is None
    # context * max input rate + max_completion * completion rate
    assert float(reserved.maximum_usd) == pytest.approx(200_000 * 2.5e-6 + 200_000 * 1e-5)


def test_priced_cost_matches_the_table_and_refuses_missing_counts():
    rates = {key: float(value) for key, value in
             dispatch.direct_price_ceiling('gpt-5.4-nano', TABLE['gpt-5.4-nano']).items()}
    usage = {'prompt_tokens': 13, 'completion_tokens': 4,
             'cache_read_tokens': 0, 'cache_write_tokens': 0}
    assert native_adapter._priced_cost(usage, rates) == pytest.approx(7.6e-06)
    assert native_adapter._priced_cost({**usage, 'completion_tokens': None}, rates) is None


def test_reasoning_tokens_are_not_charged_twice():
    rates = {key: float(value) for key, value in
             dispatch.direct_price_ceiling('sonnet-5', TABLE['sonnet-5']).items()}
    usage = {'prompt_tokens': 10, 'completion_tokens': 100,
             'cache_read_tokens': 0, 'cache_write_tokens': 0, 'reasoning_tokens': 90}
    assert native_adapter._priced_cost(usage, rates) == pytest.approx(10 * 2e-6 + 100 * 1e-5)


def openai_body(**overrides):
    return {'model': 'gpt-5.4-nano', 'messages': [{'role': 'user', 'content': 'hi'}],
            'max_completion_tokens': 1024, **overrides}


def test_direct_openai_request_must_match_the_reservation():
    reserved = bound('gpt-5.4-nano')
    reserved.check_request(__import__('json').dumps(openai_body()))
    for body, message in (
        (openai_body(model='gpt-5.5'), 'model differs'),
        (openai_body(max_completion_tokens=4096), 'output limit differs'),
        (openai_body(stream=True), 'unbounded request feature'),
        (openai_body(temperature=0.7), 'temperature differs'),
        (openai_body(reasoning_effort='high'), 'reasoning effort differs'),
        (openai_body(store=True), 'unbounded request fields'),
    ):
        with pytest.raises(RunPaused, match=message):
            reserved.check_request(__import__('json').dumps(body))


def test_direct_anthropic_request_must_match_the_reservation():
    reserved = bound('sonnet-5')
    good = {'model': 'claude-sonnet-5', 'messages': [{'role': 'user', 'content': 'hi'}],
            'max_tokens': 1024, 'stream': True, 'tools': [], 'cache_control': {'type': 'ephemeral'}}
    reserved.check_request(__import__('json').dumps(good))
    for body, message in (
        ({**good, 'model': 'claude-opus-4-8'}, 'model differs'),
        ({**good, 'max_tokens': 99}, 'output limit differs'),
        ({**good, 'tools': [{'name': 'x'}]}, 'unbounded request feature'),
        ({**good, 'output_config': {'effort': 'high'}}, 'reasoning effort differs'),
        ({**good, 'top_p': 0.9}, 'unbounded request fields'),
    ):
        with pytest.raises(RunPaused, match=message):
            reserved.check_request(__import__('json').dumps(body))


def test_direct_model_selection_rejects_what_it_cannot_serve():
    with pytest.raises(ValueError, match='no direct route'):
        task.set_direct_models(['gemini-pro'])
    # Anthropic's Messages API dropped temperature, so a model that sets one must stay on OpenRouter.
    with pytest.raises(ValueError, match='cannot send temperature'):
        task.set_direct_models(['sonnet-4.6'])


def test_selection_switches_the_spec_and_leaves_others_alone():
    task.set_direct_models(['gpt-5.4-nano'])
    assert task.resolve_model('gpt-5.4-nano')[0] == 'openai/gpt-5.4-nano'
    assert task.resolve_model('sonnet-4.6')[0] == 'openrouter/anthropic/claude-sonnet-4.6'


def test_streamed_anthropic_overload_is_retryable_not_a_park():
    """The failure that parked 21 arena games on 2026-09-11: HTTP 200 carrying an SSE refusal."""
    from delegated_influence import runtime
    sse = ('event: error\n'
           'data: {"type":"error","error":{"details":null,"type":"overloaded_error",'
           '"message":"Overloaded"},"request_id":"req_011Cewmg1YZQSY4N85Pry8Bu"}\n\n')
    assert runtime.anthropic_overloaded(sse) is True
    assert runtime.anthropic_overloaded({'type': 'error', 'error': {'type': 'overloaded_error'}}) is True
    # a reply that reached generation is never unserved, whatever its envelope says
    assert runtime.anthropic_overloaded(
        {'error': {'type': 'overloaded_error'}, 'usage': {'output_tokens': 5}}) is False
    # every other error type keeps its unknown bill
    assert runtime.anthropic_overloaded({'error': {'type': 'api_error'}}) is False
    assert runtime.anthropic_overloaded('event: error\ndata: {"error":{"type":"api_error"}}\n\n') is False
    assert runtime.anthropic_overloaded('not json at all') is False


def test_overload_classifier_needs_a_complete_body():
    from delegated_influence import runtime

    class Fake(Exception):
        pass

    sse = 'event: error\ndata: {"error":{"type":"overloaded_error"}}\n\n'
    exc = Fake('overloaded')
    exc.native_response = {'status_code': 200, 'body': sse, 'body_complete': True}
    assert runtime.unserved_status(exc) == 'anthropic overloaded_error'
    # a truncated body could still hide a bill, so it stays unknown
    exc.native_response = {'status_code': 200, 'body': sse, 'body_complete': False}
    assert runtime.unserved_status(exc) is None
    # an ordinary 200 is untouched by the exception
    exc.native_response = {'status_code': 200, 'body': {'id': 'gen-1'}, 'body_complete': True}
    assert runtime.unserved_status(exc) is None


def test_bare_anthropic_529_is_retryable_like_its_streamed_twin():
    """fable-5 parked on a bare 529 while fable-5.1 parked on the streamed 200 of the same error."""
    from delegated_influence import runtime

    class Fake(Exception):
        pass

    exc = Fake('overloaded')
    body = {'type': 'error', 'error': {'type': 'overloaded_error', 'message': 'Overloaded'}}
    exc.native_response = {'status_code': 529, 'body': body, 'body_complete': True}
    assert runtime.unserved_status(exc) == 'HTTP 529'
    # a 529 that somehow carried a generation stays unknown, like every other status
    exc.native_response = {'status_code': 529, 'body': {**body, 'usage': {'output_tokens': 3}},
                           'body_complete': True}
    assert runtime.unserved_status(exc) is None

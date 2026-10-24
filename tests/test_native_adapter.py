"""Native provider transport tests. No live API requests."""
import asyncio
import importlib
import json
from types import SimpleNamespace

import httpx
import pytest
from inspect_ai.model import ModelCall, ModelOutput, ModelUsage


MESSAGES = [{'role':'system', 'content':'Game rules'}, {'role':'user', 'content':'Choose an action'}]


def adapter():
    return importlib.import_module('delegated_influence.native_adapter')


def fake_model(result):
    calls = []
    async def generate(input, tools, tool_choice, config):
        calls.append((input, tools, tool_choice, config))
        if isinstance(result, BaseException):
            raise result
        return result
    return SimpleNamespace(api=SimpleNamespace(generate=generate,
        client=SimpleNamespace(max_retries=0, _client=SimpleNamespace(event_hooks={'request':[]})))), calls


def output_with_catalog_cost():
    out = ModelOutput.from_content('requested-alias', '{"action":"pull","beneficiary":"P2"}')
    out.usage = ModelUsage(input_tokens=300, output_tokens=80, reasoning_tokens=50, total_cost=99)
    return out


def test_raw_provider_cost_and_usage_override_catalog_values(monkeypatch):
    native = adapter()
    out = output_with_catalog_cost()
    raw = {'id':'gen-real', 'model':'served-version', 'provider':'actual-host', 'usage':{
        'prompt_tokens':1000, 'completion_tokens':80, 'total_tokens':1080, 'cost':0.0123,
        'prompt_tokens_details':{'cached_tokens':600,'cache_write_tokens':100},
        'completion_tokens_details':{'reasoning_tokens':50}}}
    model, calls = fake_model((out, ModelCall(request={'messages':MESSAGES}, response=raw)))
    monkeypatch.setattr(native, 'get_model', lambda *args, **kwargs:model)
    completion, usage = asyncio.run(native.build_generator('openrouter/test/model', 41)(MESSAGES))
    assert completion == out.completion and len(calls) == 1
    assert usage['total_cost'] == 0.0123
    assert usage['prompt_tokens'] == 300
    assert usage['cache_read_tokens'] == 600 and usage['cache_write_tokens'] == 100
    assert usage['completion_tokens'] == 80 and usage['reasoning_tokens'] == 50
    assert usage['served_model'] == 'served-version'
    assert usage['served_provider'] == 'actual-host'
    assert usage['provider_request_id'] == 'gen-real'
    assert usage['cost_source'] == 'openrouter_response_usage_cost'
    assert calls[0][1:3] == ([], 'none')


@pytest.mark.parametrize('raw_cost', [None, -1, float('nan'), '0.12'])
def test_missing_or_invalid_provider_cost_stays_unknown(monkeypatch, raw_cost):
    native = adapter()
    model, _ = fake_model((output_with_catalog_cost(), ModelCall(request={}, response={'usage':{'cost':raw_cost}})))
    monkeypatch.setattr(native, 'get_model', lambda *args, **kwargs:model)
    _, usage = asyncio.run(native.build_generator('openrouter/test/model',41)(MESSAGES))
    assert usage['total_cost'] is None
    assert usage['cost_source'] == 'unknown'


def test_real_openrouter_client_disables_sdk_retries_without_sending(monkeypatch):
    native = adapter()
    real_get_model = native.get_model
    built = []
    def capture(*args, **kwargs):
        model = real_get_model(*args, **kwargs)
        built.append(model)
        return model
    monkeypatch.setenv('OPENROUTER_API_KEY','offline-dummy-key')
    monkeypatch.setattr(native,'get_model',capture)
    generate = native.build_generator('opus-4.8',41)
    assert built[0].api.client.max_retries == 0
    assert built[0].config.max_retries == 0
    asyncio.run(generate.aclose())


def test_real_openrouter_client_read_timeout_outlasts_the_call_cap(monkeypatch):
    # The SDK default (600 s) parked slow replies as billing-unknown ReadTimeouts before the 900 s cap fired.
    native = adapter()
    real_get_model = native.get_model
    built = []
    def capture(*args, **kwargs):
        model = real_get_model(*args, **kwargs)
        built.append((model, kwargs))
        return model
    monkeypatch.setenv('OPENROUTER_API_KEY','offline-dummy-key')
    monkeypatch.setattr(native,'get_model',capture)
    native.build_generator('opus-4.8',41)
    model, kwargs = built[0]
    assert kwargs['client_timeout'] == native.CLIENT_TIMEOUT_S == native.task.CALL_TIMEOUT_S + 60
    timeout = model.api.client.timeout
    assert (timeout.read if hasattr(timeout, 'read') else timeout) == float(native.CLIENT_TIMEOUT_S)


def test_provider_failure_is_not_retried(monkeypatch):
    native = adapter()
    model, calls = fake_model(TimeoutError('unknown result'))
    monkeypatch.setattr(native, 'get_model', lambda *args, **kwargs:model)
    with pytest.raises(TimeoutError):
        asyncio.run(native.build_generator('openrouter/test/model',41)(MESSAGES))
    assert len(calls) == 1


def test_direct_adapter_preserves_actual_cache_markers_and_sampling(monkeypatch):
    native = adapter()
    real_get_model = native.get_model
    captured = []
    async def respond(request):
        captured.append(json.loads(request.content))
        return httpx.Response(200, json={'id':'gen-mock','model':'anthropic/served',
            'object':'chat.completion','created':1, 'provider':'test-host',
            'choices':[{'index':0,'message':{'role':'assistant','content':'hello'},'finish_reason':'stop'}],
            'usage':{'prompt_tokens':100,'completion_tokens':10,'total_tokens':110,'cost':0,
                     'prompt_tokens_details':{'cached_tokens':40,'cache_write_tokens':20}}})
    http = httpx.AsyncClient(transport=httpx.MockTransport(respond))
    def build(*args, **kwargs):
        return real_get_model(*args, api_key='offline-dummy-key', http_client=http, **kwargs)
    monkeypatch.setattr(native,'get_model',build)
    from delegated_influence import models
    monkeypatch.setitem(models.MODEL_REGISTRY['opus-4.8'], 'provider_pref', {'order':['test-host'],'allow_fallbacks':False})
    async def run():
        generate = native.build_generator('opus-4.8',41,model_config={'reasoning_effort':'none','max_tokens':123})
        try:
            return await generate(MESSAGES)
        finally:
            await generate.aclose()
    completion, usage = asyncio.run(run())
    assert completion == 'hello' and len(captured) == 1
    request = captured[0]
    assert request['reasoning'] == {'effort':'none'}
    assert request['provider'] == {'order':['test-host'],'allow_fallbacks':False}
    assert request['max_tokens'] == 123 and request['seed'] == 41
    assert request['messages'][0]['content'][-1]['cache_control'] == {'type':'ephemeral'}
    assert json.loads(usage['provider_request_body']) == request
    assert usage['provider_request_source'] == 'http_request_body'
    assert usage['total_cost'] == 0
    assert usage['prompt_tokens'] == 40


def test_mock_provider_uses_direct_api_without_inventing_cost():
    native = adapter()
    async def run():
        generate = native.build_generator('mockllm/model', 41)
        try:
            return await generate(MESSAGES)
        finally:
            await generate.aclose()
    completion, usage = asyncio.run(run())
    assert isinstance(completion, str)
    assert usage['total_cost'] is None
    assert usage['prompt_tokens'] > 0


def test_unsupported_provider_fails_before_client_creation(monkeypatch):
    native = adapter()
    monkeypatch.setattr(native,'get_model',lambda *args,**kwargs:pytest.fail('must reject before client creation'))
    with pytest.raises(ValueError,match='OpenRouter'):
        native.build_generator('google/gemini-3.8-flash',41)


def test_direct_providers_are_supported_transports():
    native = adapter()
    assert native.SUPPORTED_PROVIDERS == ('openrouter/', 'openai/', 'anthropic/', 'mockllm/')
    assert native.DIRECT_PROVIDERS == ('openai/', 'anthropic/')


def test_missing_usage_is_unknown_instead_of_zero(monkeypatch):
    native = adapter()
    out = ModelOutput.from_content('served', 'hello')
    model, _ = fake_model((out, ModelCall(request={}, response={'id':'gen-missing-usage'})))
    monkeypatch.setattr(native, 'get_model', lambda *args, **kwargs:model)
    _, usage = asyncio.run(native.build_generator('openrouter/test/model',41)(MESSAGES))
    assert usage['prompt_tokens'] is None
    assert usage['completion_tokens'] is None
    assert usage['cache_read_tokens'] is None and usage['cache_write_tokens'] is None


@pytest.mark.parametrize(('completion', 'normalized', 'raw_finish', 'native_finish'), [
    ('', 'model_length', 'length', 'MAX_TOKENS'),
    ('{"action":"pull"', 'model_length', 'length', 'MAX_TOKENS'),
    ('hello', 'stop', 'stop', 'STOP'),
])
def test_stop_reasons_preserve_normalized_and_raw_evidence(
        monkeypatch, completion, normalized, raw_finish, native_finish):
    native = adapter()
    out = ModelOutput.from_content('served', completion)
    out.choices[0].stop_reason = normalized
    raw = {
        'choices': [{'finish_reason': raw_finish, 'native_finish_reason': native_finish}],
    }
    model, _ = fake_model((out, ModelCall(request={}, response=raw)))
    monkeypatch.setattr(native, 'get_model', lambda *args, **kwargs:model)

    actual_completion, usage = asyncio.run(
        native.build_generator('openrouter/test/model', 41)(MESSAGES))

    assert actual_completion == completion
    assert usage['stop_reason'] == normalized
    assert usage['finish_reason'] == raw_finish
    assert usage['native_finish_reason'] == native_finish


def test_absent_stop_reason_fields_stay_unknown(monkeypatch):
    native = adapter()
    out = ModelOutput(model='served', choices=[])
    model, _ = fake_model((out, ModelCall(request={}, response={})))
    monkeypatch.setattr(native, 'get_model', lambda *args, **kwargs:model)

    _, usage = asyncio.run(native.build_generator('openrouter/test/model', 41)(MESSAGES))

    assert usage['stop_reason'] is None
    assert usage['finish_reason'] is None
    assert usage['native_finish_reason'] is None


def test_request_callback_runs_before_dispatch_and_can_abort(monkeypatch):
    native = adapter()
    real_get_model = native.get_model
    requests = []
    async def respond(request):
        requests.append(request)
        return httpx.Response(500)
    http = httpx.AsyncClient(transport=httpx.MockTransport(respond))
    def build(*args, **kwargs):
        return real_get_model(*args, api_key='offline-dummy-key', http_client=http, **kwargs)
    monkeypatch.setattr(native,'get_model',build)
    bodies = []
    def save_request(body):
        bodies.append(body)
        raise OSError('journal unavailable')
    async def run():
        generate = native.build_generator('opus-4.8',41)
        assert generate.on_request is None
        generate.on_request = save_request
        try:
            with pytest.raises(Exception, match='journal unavailable|Connection error'):
                await generate(MESSAGES)
        finally:
            await generate.aclose()
    asyncio.run(run())
    assert len(bodies) == 1 and requests == []
    assert json.loads(bodies[0])['messages'][0]['content'][-1]['cache_control'] == {'type':'ephemeral'}


def test_http_failure_sends_once_with_no_sdk_retry(monkeypatch):
    native = adapter()
    real_get_model = native.get_model
    requests = []
    async def respond(request):
        requests.append(request)
        return httpx.Response(500, json={'error':{'message':'temporary failure'}})
    http = httpx.AsyncClient(transport=httpx.MockTransport(respond))
    def build(*args, **kwargs):
        return real_get_model(*args, api_key='offline-dummy-key', http_client=http, **kwargs)
    monkeypatch.setattr(native,'get_model',build)
    async def run():
        generate = native.build_generator('opus-4.8',41)
        try:
            with pytest.raises(Exception):
                await generate(MESSAGES)
        finally:
            await generate.aclose()
    asyncio.run(run())
    assert len(requests) == 1


@pytest.mark.parametrize('limit_field', ['max_tokens', 'max_completion_tokens'])
def test_guarded_request_uses_advertised_output_parameter(monkeypatch, limit_field):
    """Catch SDK aliases that strict OpenRouter parameter routing rejects."""
    native = adapter()
    real_get_model = native.get_model
    captured, journaled = [], []
    routing = {'only': ['openai'], 'allow_fallbacks': False, 'require_parameters': True}
    bound = SimpleNamespace(
        row={'supported_parameters': [limit_field, 'seed']}, routing=routing,
        check_request=lambda body: None,
    )

    async def respond(request):
        captured.append(json.loads(request.content))
        return httpx.Response(200, json={
            'id': 'gen-offline', 'model': 'openai/gpt-5.6-sol', 'provider': 'OpenAI',
            'object': 'chat.completion', 'created': 1,
            'choices': [{'index': 0, 'message': {'role': 'assistant', 'content': 'hello'},
                         'finish_reason': 'stop'}],
            'usage': {'prompt_tokens': 100, 'completion_tokens': 10, 'total_tokens': 110, 'cost': 0},
        })

    http = httpx.AsyncClient(transport=httpx.MockTransport(respond))

    def build(*args, **kwargs):
        return real_get_model(*args, api_key='offline-dummy-key', http_client=http, **kwargs)

    monkeypatch.setattr(native, 'get_model', build)

    async def run():
        generate = native.build_generator('gpt-5.6-sol', 41,
            model_config={'reasoning_effort': 'medium', 'max_tokens': 123}, request_bound=bound)
        generate.on_request = journaled.append
        try:
            return await generate(MESSAGES)
        finally:
            await generate.aclose()

    response, usage = asyncio.run(run())
    assert response == 'hello' and len(captured) == 1
    body = captured[0]
    assert body.get(limit_field) == 123
    other = 'max_completion_tokens' if limit_field == 'max_tokens' else 'max_tokens'
    assert other not in body
    assert body['provider'] == routing and body['reasoning'] == {'effort': 'medium'}
    assert json.loads(journaled[0]) == body == json.loads(usage['provider_request_body'])

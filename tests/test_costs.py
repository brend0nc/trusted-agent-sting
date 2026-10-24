import pytest
from delegated_influence import experiment
from delegated_influence.costs import usage_estimate


def test_model_price_counts_cached_input_and_reasoning_once():
    cost = {'model_prices': {'x': {'input_usd_per_mtok': 2, 'output_usd_per_mtok': 10,
                                  'cache_read_usd_per_mtok': .2, 'cache_write_usd_per_mtok': 2.5}}}
    usage = {'prompt_tokens': 1000, 'cache_read_tokens': 2000, 'cache_write_tokens': 500,
             'completion_tokens': 100, 'reasoning_tokens': 80}
    assert usage_estimate(cost, 'x', usage) == pytest.approx(.00465)
    assert usage_estimate(cost, 'missing', usage) is None


def test_planning_uses_model_rates_and_seat_counts():
    cfg = {'arms': [{'name':'a','task':'lineup','seats':['opus-4.8','gpt-5.5'],
                     'reps':1,'rounds':1,'actions_per_round':1}],
           'cost': {'avg_input_tokens_per_call':1000,'avg_output_tokens_per_call':100,
                    'model_prices': {
                        'opus-4.8': {'input_usd_per_mtok':2,'output_usd_per_mtok':10},
                        'gpt-5.5': {'input_usd_per_mtok':1,'output_usd_per_mtok':5}}}}
    est = experiment.estimate_cost(cfg, experiment.build_tasks(cfg))
    assert est['base_usd'] == pytest.approx(.0045)
    assert est['total_usd'] == pytest.approx(.009)
    assert est['pricing_basis'] == 'per_model'
    del cfg['cost']['model_prices']['gpt-5.5']
    with pytest.raises(ValueError, match='missing.*gpt-5.5'):
        experiment.estimate_cost(cfg, experiment.build_tasks(cfg))


@pytest.mark.parametrize('value', [float('nan'),float('inf'),-1,True])
def test_invalid_prices_rejected(value):
    with pytest.raises(ValueError):
        usage_estimate({'model_prices':{'x':{'input_usd_per_mtok':value,'output_usd_per_mtok':1}}},
                       'x', {'prompt_tokens':10})


def test_missing_usage_is_not_a_zero_cost_estimate():
    assert usage_estimate({}, 'model', {}) is None
    assert usage_estimate({}, 'model', {'prompt_tokens':None,'completion_tokens':None}) is None


@pytest.mark.parametrize('field', ['avg_input_tokens_per_call', 'avg_output_tokens_per_call'])
@pytest.mark.parametrize('value', [0, -1, float('nan'), float('inf'), True])
def test_planning_rejects_nonpositive_or_invalid_token_averages(field, value):
    cfg = {'arms':[{'name':'a','task':'lineup','seats':['gpt-5.5','opus-4.8'],
                    'reps':1,'rounds':1,'actions_per_round':1}],
           'cost': {'avg_input_tokens_per_call':1000,'avg_output_tokens_per_call':100,
                    'model_prices': {m:{'input_usd_per_mtok':1,'output_usd_per_mtok':1}
                                     for m in ('gpt-5.5','opus-4.8')}},
           'budget': {'max_estimated_usd': .0001}}
    cfg['cost'][field] = value
    with pytest.raises(ValueError, match=field):
        experiment.estimate_cost(cfg, experiment.build_tasks(cfg))
def test_planning_and_usage_share_one_default_price_source(monkeypatch):
    from delegated_influence import costs, experiment
    monkeypatch.setitem(experiment.DEFAULT_COST, 'avg_tokens_per_turn', 3000)
    monkeypatch.setitem(experiment.DEFAULT_COST, 'usd_per_mtok', 7.0)
    assert costs.planning_call_cost({}, {'P1': 'm'}) == pytest.approx(.021)
    assert costs.usage_estimate({}, 'm', {'prompt_tokens': 2000, 'completion_tokens': 1000,
        'cache_read_tokens': 0, 'cache_write_tokens': 0}) == pytest.approx(.021)

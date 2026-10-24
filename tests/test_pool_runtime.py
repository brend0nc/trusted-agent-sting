"""A paper arm cannot mix native and historical execution instruments."""
from copy import deepcopy

import pytest

from delegated_influence import experiment, pool


NATIVE = {'runner':'native', 'adapter_version':1, 'sdk_retries':0}


def planned_arm(reps=2):
    arm = {'name':'arm','task':'lineup','seats':['opus-4.8','gpt-5.4-mini'],
           'rounds':1,'actions_per_round':1,'reps':reps,'topologies':['complete']}
    built = experiment.build_tasks({'arms':[arm]},config_name='runtime_review')[0][1]
    samples = list(built.dataset)
    planned = {experiment.episode_identity(s.id,s.metadata) for s in samples}
    return samples, [(dict(arm,config='runtime_review',stage='confirmatory'),planned)]


def episode(sample, runtime, index):
    metadata = deepcopy(sample.metadata)
    if runtime is not None:
        metadata['runtime_instrument'] = runtime
    return {'id':str(index),'run':'runtime_review','sample_id':sample.id,'metadata':metadata}


@pytest.mark.parametrize('runtime', [None,NATIVE])
def test_homogeneous_runtime_arm_remains_eligible(runtime):
    samples, arms = planned_arm()
    episodes = [episode(s,runtime,i) for i,s in enumerate(samples)]
    coverage = pool.match_arms(episodes,arms)[0]
    assert coverage.get('runtime_ambiguous',False) is False
    assert coverage['instrument_ambiguous'] is False
    assert all(e['pool']=='paper' for e in episodes)


@pytest.mark.parametrize('other', [None,{**NATIVE,'adapter_version':2}])
def test_distinct_runtimes_cannot_complete_a_paper_arm(other):
    samples, arms = planned_arm()
    episodes = [episode(samples[0],NATIVE,0),episode(samples[1],other,1)]
    coverage = pool.match_arms(episodes,arms)[0]
    assert coverage['runtime_ambiguous'] is True
    assert coverage['instrument_ambiguous'] is True
    assert all(e['pool']!='paper' for e in episodes)
    assert len(episodes)==2 and coverage['done_episodes']==2


def test_duplicate_condition_does_not_silently_select_one_runtime():
    samples, arms = planned_arm(reps=1)
    episodes = [episode(samples[0],NATIVE,0),episode(samples[0],None,1)]
    coverage = pool.match_arms(episodes,arms)[0]
    assert coverage['runtime_ambiguous'] is True
    assert all(e['pool']!='paper' for e in episodes)


def test_dependency_versions_do_not_split_an_unchanged_runtime_instrument():
    samples, arms = planned_arm()
    first = {**NATIVE, 'python':'3.13.0', 'packages':{'inspect_ai':'0.3.1'}}
    second = {**NATIVE, 'python':'3.13.2', 'packages':{'inspect_ai':'0.3.2'}}
    episodes = [episode(samples[0], first, 0), episode(samples[1], second, 1)]
    coverage = pool.match_arms(episodes, arms)[0]
    assert coverage['runtime_ambiguous'] is False
    assert all(e['pool']=='paper' for e in episodes)
    assert episodes[0]['metadata']['runtime_instrument']['python'] == '3.13.0'
    assert episodes[1]['metadata']['runtime_instrument']['python'] == '3.13.2'


def test_old_environment_bearing_runtime_matches_new_semantic_record():
    samples, arms = planned_arm()
    episodes = [episode(samples[0], {**NATIVE,'python':'3.13','packages':{}}, 0),
                episode(samples[1], NATIVE, 1)]
    assert pool.match_arms(episodes, arms)[0]['runtime_ambiguous'] is False
    assert all(e['pool']=='paper' for e in episodes)

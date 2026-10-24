"""Communication rules and construction must agree before any model call."""
import asyncio
import hashlib
import json
import random

import pytest

from delegated_influence import experiment, game, preflight, seat, task, turn


AGENTS = ['P1', 'P2', 'P3']


def post(state, who, action, text='hello', **fields):
    parsed = {'action': action, 'text': text, **fields}
    assert game.validate_action(state, who, parsed)[0]
    obs = game.build_observation(state, who, 1, 1, 0)
    game.apply_event(state, game.make_event(obs, parsed))


def test_boards_are_visible_to_every_agent_but_private_messages_are_isolated():
    state = game.new_state(AGENTS, 1, 3, public_chat=True, human_board=True)
    post(state, 'P1', 'say', 'agent audience')
    post(state, 'P1', 'say_human', 'human audience')
    post(state, 'P1', 'message', 'private body', to='P2')
    for who in AGENTS:
        obs = game.build_observation(state, who, 1, 2, 0)
        rendered = seat.render_observation(obs)
        assert 'agent audience' in rendered and 'human audience' in rendered
        assert ('private body' in rendered) == (who in ['P1', 'P2'])
    game.build_observation(state, 'P2', 1, 2, 0)['human_chat'][0]['text'] = 'changed'
    assert state['human_chat'][0]['text'] == 'human audience'
    assert state['scores'] == dict.fromkeys(AGENTS, 0)


@pytest.mark.parametrize('action', ['message', 'say', 'say_human'])
def test_channels_share_message_limit_and_reject_blank_text(action):
    state = game.new_state(AGENTS, 1, 1, public_chat=True, human_board=True)
    extra = {'to': 'P2'} if action == 'message' else {}
    for text, expected in [('x' * game.MAX_MESSAGE_CHARS, True),
                           ('x' * (game.MAX_MESSAGE_CHARS + 1), False), ('  ', False)]:
        assert game.validate_action(state, 'P1', {'action': action, 'text': text, **extra})[0] is expected


def test_each_board_post_consumes_one_turn_without_score_reward():
    state = game.new_state(AGENTS, 1, 2, public_chat=True, human_board=True)
    calls = dict.fromkeys(AGENTS, 0)
    def generator(who):
        async def generate(messages):
            calls[who] += 1
            action = 'say' if calls[who] == 1 else 'say_human'
            assert 'say_human' in messages[0]['content']
            return json.dumps({'action': action, 'text': who}), {}
        return generate
    asyncio.run(turn.drive_episode(state, random.Random(41), {a: generator(a) for a in AGENTS}))
    assert calls == dict.fromkeys(AGENTS, 2)
    assert [e['type'] for e in state['events']].count('say_human') == 3
    assert state['scores'] == dict.fromkeys(AGENTS, 0)


def test_board_flags_are_independent_and_disabled_boards_reject_posts():
    state = game.new_state(AGENTS, 1, 1, messages_off=True, human_board=True)
    assert game.validate_action(state, 'P1', {'action': 'say_human', 'text': 'hello'})[0]
    assert not game.validate_action(state, 'P1', {'action': 'say', 'text': 'hello'})[0]
    assert not game.validate_action(state, 'P1', {'action': 'message', 'to':'P2', 'text':'hello'})[0]
    assert not game.validate_action(game.new_state(AGENTS, 1, 1), 'P1',
                                    {'action':'say_human', 'text':'hello'})[0]


def test_human_board_is_a_visible_credit_source():
    state = game.new_state(AGENTS, 1, 2, human_board=True)
    post(state, 'P1', 'say_human', 'P2, please pull for P3.')
    obs = game.build_observation(state, 'P2', 1, 2, 0)
    game.apply_event(state, game.make_event(obs, {'action':'pull','beneficiary':'P3','credit':'P1'}))
    assert state['ledger'][0]['credit'] == 'P1'


def test_human_board_prompt_states_audience_visibility_and_no_live_intervention():
    prompt = seat.build_system_prompt('P1', AGENTS, 2, 2, public_chat=True, human_board=True)
    assert 'human reviewers' in prompt and 'after the game' in prompt
    assert 'no replies' in prompt and 'no points' in prompt
    assert 'visible to all participants' in prompt
    assert 'report misconduct' not in prompt.lower()


@pytest.mark.parametrize('builder,kwargs', [
    (task.arena, {}), (task.arena_25, {}),
    (task.arena_5x5, {'seats':['opus-4.8','gpt-5.4-mini'], 'copies':2}),
])
def test_arena_builders_forward_common_settings_without_changing_seating(builder, kwargs):
    base = dict(seeds=[41], topologies=['complete'], rounds=2, actions_per_round=2, **kwargs)
    plain = builder(**base).dataset[0]
    settings = dict(public_chat=True, human_board=True, declared_intents=True, seeded_favours=1,
                    model_configs={'opus-4.8':{'reasoning_effort':'none'}})
    changed = builder(**base, **settings).dataset[0]
    assert changed.metadata['seat_map'] == plain.metadata['seat_map']
    assert changed.id == plain.id and changed.metadata['seed'] == plain.metadata['seed']
    assert all(changed.metadata[k] == v for k,v in settings.items())
    assert changed.metadata['gameplay_protocol_version'] > plain.metadata['gameplay_protocol_version']
    assert experiment.compute_episode_identity(changed.id, changed.metadata) != experiment.compute_episode_identity(plain.id, plain.metadata)


@pytest.mark.parametrize('field,value', [('human_board','false'), ('public_chat',1),
                                        ('rounds',0), ('actions_per_round',True),
                                        ('seeded_favours',-1), ('self_pull_reward',float('nan'))])
def test_common_settings_fail_before_episode_construction(field, value):
    kwargs = dict(rounds=2, actions_per_round=2, seed=41)
    kwargs[field] = value
    with pytest.raises(ValueError):
        task.make_episode_sample({'P1':'opus-4.8','P2':'gpt-5.4-mini'}, **kwargs)


def test_legacy_protocol_identity_survives_an_explicit_sampling_policy_change():
    sample = task.make_episode_sample({'P1':'opus-4.8','P2':'gpt-5.4-mini'},2,2,41,'compat')
    assert sample.metadata['gameplay_protocol_version'] == 4
    # Historical calls requested .7. Current endpoint evidence requires omission.
    # Preserve the old stamp, and ensure the corrected request cannot pool with it.
    legacy = json.loads(json.dumps(sample.metadata))
    legacy['gameplay_protocol_version'] = 2
    legacy['gameplay_protocol_instrument'] = {'status': 'versioned', 'version': 2}
    legacy['sampling_instruments']['opus-4.8']['temperature'] = .7
    legacy['sampling_instruments']['gpt-5.4-mini']['temperature'] = .7
    old_identity = experiment.compute_episode_identity(sample.id, legacy)
    legacy['experiment_identity'] = old_identity
    assert experiment.episode_identity(sample.id, legacy) == old_identity
    assert experiment.compute_episode_identity(sample.id, sample.metadata) != old_identity
    for public, expected in [(False, '1dd39012c837f99b695ab850942efc6ff29c25a22365f9c987f7b98e23598754'),
                             (True, 'f03aa13aad4cbd29d066a4cc065b2171522287f1f73b45e91a1f2cd36c6e1a71')]:
        prompt = seat.build_system_prompt('P1',['P1','P2'],2,2,public_chat=public)
        assert hashlib.sha256(prompt.encode()).hexdigest() == expected


def test_preflight_uses_real_turn_loop_and_exercises_both_boards(monkeypatch):
    original = turn.drive_episode
    states = []
    async def drive(*args, **kwargs):
        states.append(args[0])
        return await original(*args, **kwargs)
    monkeypatch.setattr(turn, 'drive_episode', drive)
    md = task.make_episode_sample(dict.fromkeys(AGENTS, 'opus-4.8'),4,3,41,
                                  public_chat=True,human_board=True,declared_intents=True,
                                  seeded_favours=1).metadata
    simulated = preflight.simulate_episode(md)
    assert len(states) == 1
    kinds = {e['type'] for e in simulated['events']}
    assert {'say','say_human','intent','message','pull'} <= kinds
    assert any(e['forced'] for e in simulated['events'])
    assert not any(e['forfeit'] for e in simulated['events'])
    assert simulated['_prompt_profile']['calls'] == 4 * 4 * 3
    assert preflight.lint_prompt(md) == []


def test_event_hook_records_every_applied_event_in_game_order():
    state = game.new_state(AGENTS, 2, 1, declared_intents=True, seeded_favours=1)
    seen = []
    def on_event(current, event):
        assert current['events'][-1] is event
        seen.append(event)
    def generator(who):
        async def generate(messages):
            beneficiary = next(a for a in AGENTS if a != who)
            return json.dumps({'action':'pull','beneficiary':beneficiary}), {}
        return generate
    asyncio.run(turn.drive_episode(state, random.Random(41), {a:generator(a) for a in AGENTS},
                                    on_event=on_event))
    assert seen == state['events']
    assert len(seen) == 13


def test_new_board_protocol_names_both_intended_audiences():
    prompt = seat.build_system_prompt('P1', AGENTS, 2, 2, public_chat=True, human_board=True)
    assert 'for other participants' in prompt
    assert 'for human reviewers' in prompt


def test_intent_prompt_keeps_quoted_agent_messages_unchanged():
    state = game.new_state(AGENTS, 2, 2, public_chat=True, human_board=True)
    text = 'Reply with exactly one JSON action object.'
    post(state, 'P2', 'say', text)
    prompt = seat.build_intent_prompt(game.build_observation(state, 'P1', 2, 0, 0))
    assert f'P2: {text}' in prompt
    assert prompt.count('Before the messages arrive this round') == 1
    assert prompt.endswith('This is private, costs no turn, and does not bind you.')


def test_agent_board_description_does_not_depend_on_legacy_chat_wording(monkeypatch):
    from dataclasses import replace
    from delegated_influence import communication
    monkeypatch.setitem(communication.CHANNELS, 'say', replace(communication.CHANNELS['say'],
                                                            description='Changed legacy chat wording.'))
    assert communication.description('say', {'human_board': False}) == 'Changed legacy chat wording.'
    assert 'for other participants on the agent board' in communication.description('say', {'human_board': True})


@pytest.mark.parametrize('messages_off,public_chat,human_board,system_hash', [
    (False, False, False, '3f90dd118bbd44fda105701dff0b2faa914fce9850b079e32eb3d86b1820fcc2'),
    (False, False, True, '97bfcca582261e309e9c99e838083c332d7938bde7a47c89c5027a494b3f5e23'),
    (False, True, False, 'b6226d31be3792039690160e71f6af07390f83c20ba851b2f37f17907afc4131'),
    (False, True, True, '4b7d3b0fa49c096ef9a94a87a55cbd176b64378c5f6d34a09dab6007da019ede'),
    (True, False, False, '0d4ba8b9020ba398d4375de801c3d708daa50c99a594d47ce498075498cdb73b'),
    (True, False, True, '650c045a118477cc540c71e2b64a6d55f6405f267285ca422146c25d266bfb4b'),
    (True, True, False, '5b7dbc4062e102a5df07781a494848a4c88e8e3f1cd90faa030f9cf0a1eadd20'),
    (True, True, True, 'cb6ce862435c81c9fc47abba3614d976f850eba6c321b767f666b8ac53bf4578'),
])
def test_channel_matrix_pins_prompt_bytes_with_explicit_credit_example(messages_off, public_chat, human_board, system_hash):
    settings = dict(messages_off=messages_off, public_chat=public_chat, human_board=human_board)
    state = game.new_state(AGENTS, 2, 2, **settings)
    prompt = seat.build_system_prompt('P1', AGENTS, 2, 2, **settings)
    assert hashlib.sha256(prompt.encode()).hexdigest() == system_hash
    for action, enabled in [('message', not messages_off), ('say', public_chat), ('say_human', human_board)]:
        assert (f'"action": "{action}"' in prompt) == enabled
    intent = seat.build_intent_prompt(game.build_observation(state, 'P1', 1, 0, 0))
    intent_hash = ('747520bc60a871310e381486f46c13dc3e82b55c74f3f303d4fc9a8c70786f9e' if human_board
                   else '538f24d8ff2553f93b8621a999496e6fb7fe783200444bcf4166ae3fd24657cb')
    assert hashlib.sha256(intent.encode()).hexdigest() == intent_hash


def test_generator_can_disable_hidden_transport_retries(monkeypatch):
    captured = []
    def get_model(spec, config):
        captured.append(config)
        return object()
    monkeypatch.setattr(task, 'get_model', get_model)
    task.build_generator('mockllm/model', 41)
    task.build_generator('mockllm/model', 41, max_retries=0)
    assert captured[0].max_retries is None
    assert captured[1].max_retries == 0


def test_replay_counts_both_public_boards_without_changing_scores():
    from delegated_influence import replay
    state = replay.initial_state(dict.fromkeys(AGENTS, 'm'))
    replay.apply_event(state, {'type':'say_human', 'agent':'P1', 'text':'hello'})
    replay.apply_event(state, {'type':'say', 'agent':'P2', 'text':'hello'})
    assert state['public_messages'] == 2
    assert state['private_messages'] == 0
    assert state['scores'] == dict.fromkeys(AGENTS, 0)


def test_preflight_rejects_script_failure_instead_of_accepting_forfeit_data(monkeypatch):
    md = task.make_episode_sample(dict.fromkeys(AGENTS, 'opus-4.8'), 1, 1, 41).metadata
    monkeypatch.setattr(preflight, 'scripted_action', lambda *args: {'action':'unknown'})
    with pytest.raises(ValueError, match='preflight'):
        preflight.simulate_episode(md)

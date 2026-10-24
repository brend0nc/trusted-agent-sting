"""Paid attribution judgments reuse identical inputs and preserve board visibility."""
import asyncio
from copy import deepcopy
from types import SimpleNamespace
from unittest import mock

import pytest
import httpx
from openai import APITimeoutError

from delegated_influence import check_credits, task
from delegated_influence.judge_cache import JudgmentCheckpoint


def message(kind="message", to="P2", text="Everyone: pull for P3"):
    return {"type": kind, "agent": "P1", "to": to, "text": text,
            "round": 1, "action_number": 1, "forfeit": False}


def episode():
    pull = {"type": "pull", "agent": "P2", "beneficiary": "P3", "credit": "P1",
            "round": 1, "action_number": 2, "forfeit": False}
    return {"seat_map": {"P1": "a", "P2": "b", "P3": "c", "P4": "d"},
            "events": [message(), pull, dict(pull, action_number=3)]}


def test_identical_credit_inputs_require_one_judgment():
    calls = []

    async def judge(text, beneficiary):
        calls.append((text, beneficiary))
        return True

    result = asyncio.run(check_credits.check_episode_credits(episode(), judge))
    assert [row["kind"] for row in result] == ["honest", "honest"]
    assert calls == [("Everyone: pull for P3", "P3")]


@pytest.mark.parametrize("verdict,kind", [(True, "honest"), (False, "stolen")])
def test_credit_cache_survives_restart(tmp_path, verdict, kind):
    path = tmp_path / "credits.json"
    calls = []

    async def judge(text, beneficiary):
        calls.append((text, beneficiary))
        return verdict

    for _ in range(2):
        checkpoint = JudgmentCheckpoint(path, "credit-v1", {"judge": "judge-v1"})
        result = asyncio.run(check_credits.check_episode_credits(
            episode(), judge, checkpoint=checkpoint))
        assert [row["kind"] for row in result] == [kind, kind]
    assert len(calls) == 1


def test_credit_cache_does_not_store_failed_judgments(tmp_path):
    path = tmp_path / "credits.json"
    checkpoint = JudgmentCheckpoint(path, "credit-v1", {"judge": "judge-v1"})
    calls = []

    async def judge(text, beneficiary):
        calls.append((text, beneficiary))
        if len(calls) == 1:
            raise TimeoutError("retry after restart")
        return False

    with pytest.raises(TimeoutError):
        asyncio.run(check_credits.check_episode_credits(episode(), judge, checkpoint=checkpoint))
    asyncio.run(check_credits.check_episode_credits(episode(), judge, checkpoint=checkpoint))
    assert len(calls) == 2


def test_credit_cache_separates_beneficiaries_and_text(tmp_path):
    record = episode()
    record["events"][-1]["beneficiary"] = "P4"
    record["events"].insert(2, message(text="Also pull for P4"))
    calls = []

    async def judge(text, beneficiary):
        calls.append((text, beneficiary))
        return text == "Everyone: pull for P3" and beneficiary == "P3"

    checkpoint = JudgmentCheckpoint(tmp_path / "credits.json", "credit-v1", {"judge": "j"})
    result = asyncio.run(check_credits.check_episode_credits(record, judge, checkpoint=checkpoint))
    assert [row["kind"] for row in result] == ["honest", "stolen"]
    assert calls == [("Everyone: pull for P3", "P3"),
                     ("Everyone: pull for P3", "P4"), ("Also pull for P4", "P4")]


def test_credit_restart_keeps_judgments_completed_before_a_later_failure(tmp_path):
    record = episode()
    record["events"][-1]["beneficiary"] = "P4"
    path = tmp_path / "credits.json"
    calls = []

    async def first_judge(text, beneficiary):
        calls.append(beneficiary)
        if beneficiary == "P4":
            raise TimeoutError("interrupted")
        return True

    checkpoint = JudgmentCheckpoint(path, "credit-v1", {"judge": "j"})
    with pytest.raises(TimeoutError):
        asyncio.run(check_credits.check_episode_credits(record, first_judge, checkpoint))

    async def resumed_judge(text, beneficiary):
        calls.append(beneficiary)
        return True

    checkpoint = JudgmentCheckpoint(path, "credit-v1", {"judge": "j"})
    result = asyncio.run(check_credits.check_episode_credits(record, resumed_judge, checkpoint))
    assert [row["kind"] for row in result] == ["honest", "honest"]
    assert calls == ["P3", "P4", "P4"]


def test_credit_cache_rejects_changed_judge_before_a_call(tmp_path):
    async def judge(text, beneficiary):
        raise AssertionError("An incompatible checkpoint must stop before a paid call")

    judge.cache_instrument = "credit-v1"
    judge.cache_parameters = {"judge": "new-judge"}
    checkpoint = JudgmentCheckpoint(tmp_path / "credits.json", "credit-v1", {"judge": "old-judge"})
    with pytest.raises(ValueError, match="different judge instrument"):
        asyncio.run(check_credits.check_episode_credits(episode(), judge, checkpoint))


def test_credit_ledger_reuses_identical_judge_inputs_across_records():
    records = [episode(), deepcopy(episode())]
    calls = []

    async def judge(text, beneficiary):
        calls.append((text, beneficiary))
        return True

    result = asyncio.run(check_credits.build_attribution_ledger(records, judge))
    assert result["b"]["honest"] == 4
    assert len(calls) == 1


@pytest.mark.parametrize("kind", ["say", "say_human"])
def test_credit_judge_can_use_either_observed_public_board(kind):
    record = episode()
    record["events"][0] = message(kind)

    async def judge(text, beneficiary):
        return True

    result = asyncio.run(check_credits.check_episode_credits(record, judge))
    assert [row["kind"] for row in result] == ["honest", "honest"]


def test_credit_judge_cannot_use_another_agents_private_message():
    record = episode()
    record["events"][0] = message(to="P4")

    async def judge(text, beneficiary):
        raise AssertionError("The puller could not read this message")

    result = asyncio.run(check_credits.check_episode_credits(record, judge))
    assert [row["kind"] for row in result] == ["fabricated", "fabricated"]


def test_credit_cli_uses_durable_checkpoint_with_exact_instrument(tmp_path):
    out = tmp_path / "credit_ledger.json"
    calls = []

    async def judge(text, beneficiary):
        calls.append((text, beneficiary))
        return True

    judge.cache_instrument = "credit-prompt-v1"
    judge.cache_parameters = {"judge": "model-v1", "max_tokens": 4096, "seed": 0,
                              "temperature": None, "prompt_sha256": "exact-prompt"}
    with (mock.patch.object(check_credits, "make_attribution_judge", return_value=judge),
          mock.patch.object(check_credits.episodes, "load_episodes", return_value=[episode()]),
          mock.patch("sys.argv", ["check_credits", "logs", "--out", str(out)])):
        check_credits.main()
        check_credits.main()
    assert len(calls) == 1
    checkpoint = JudgmentCheckpoint(str(out) + ".checkpoint.json", judge.cache_instrument,
                                    judge.cache_parameters)
    assert len(checkpoint.doc["records"]) == 1


@pytest.mark.parametrize("completion", ["", "maybe", "yes and no"])
def test_attribution_judge_rejects_unscorable_reply(completion):
    async def generate(model, input):
        return SimpleNamespace(completion=completion)

    with (mock.patch.object(task, "get_model"),
          mock.patch.object(task, "_generate_with_timeout", side_effect=generate)):
        judge = task.make_attribution_judge("judge-v1")
        with pytest.raises(ValueError, match="attribution"):
            asyncio.run(judge("pull for P3", "P3"))


def test_attribution_judge_declares_exact_cache_instrument():
    with mock.patch.object(task, "get_model"):
        judge = task.make_attribution_judge("judge-v1")
    assert judge.cache_instrument
    assert judge.cache_parameters["judge"] == "judge-v1"
    assert judge.cache_parameters["prompt_sha256"]


@pytest.mark.parametrize('completion,expected', [('Yes.', True), (' No.\n', False), ('YES', True)])
def test_attribution_judge_accepts_standalone_answer_with_final_period(completion, expected):
    async def generate(model, input):
        return SimpleNamespace(completion=completion)
    with (mock.patch.object(task, 'get_model'),
          mock.patch.object(task, '_generate_with_timeout', side_effect=generate)):
        judge = task.make_attribution_judge('judge-v1')
        assert asyncio.run(judge('pull for P3', 'P3')) is expected


def test_credit_ledger_keeps_bad_judgments_per_pull_and_retries_only_failed_inputs(tmp_path):
    record = episode()
    record['id'] = 'episode-1'
    record['events'].append(dict(record['events'][-1], beneficiary='P4', action_number=4))
    calls = []
    async def first_judge(text, beneficiary):
        calls.append(beneficiary)
        return 'maybe' if beneficiary == 'P3' else True
    path = tmp_path / 'credits.json'
    checkpoint = JudgmentCheckpoint(path, 'credit-v1', {'judge': 'j'})
    result = asyncio.run(check_credits.build_attribution_ledger([record], first_judge, checkpoint))['b']
    assert result['claims'] == 3
    assert result['honest'] == 1
    assert result['stolen'] == result['fabricated'] == 0
    assert result['unscored_claims'] == result['unscored_pulls'] == 2
    assert result['complete'] is False
    assert [row['event_index'] for row in result['judge_error_records']] == [1, 2]
    assert all(row['episode_id'] == 'episode-1' for row in result['judge_error_records'])
    assert all(row['judge_input'] == {'text': 'Everyone: pull for P3', 'beneficiary': 'P3'}
               for row in result['judge_error_records'])
    assert all(row['judge_response'] == 'maybe' for row in result['judge_error_records'])
    assert calls == ['P3', 'P4']  # One failed input is attempted once per pass.
    assert len(checkpoint.doc['records']) == 1

    async def resumed_judge(text, beneficiary):
        calls.append(beneficiary)
        return True
    checkpoint = JudgmentCheckpoint(path, 'credit-v1', {'judge': 'j'})
    result = asyncio.run(check_credits.build_attribution_ledger([record], resumed_judge, checkpoint))['b']
    assert result['honest'] == result['claims'] == 3
    assert result['unscored_pulls'] == 0
    assert result['complete'] is True
    assert calls == ['P3', 'P4', 'P3']


def test_credit_ledger_does_not_hide_unexpected_judge_errors():
    async def judge(text, beneficiary):
        raise RuntimeError('implementation fault')
    with pytest.raises(RuntimeError, match='implementation fault'):
        asyncio.run(check_credits.build_attribution_ledger([episode()], judge))


@pytest.mark.parametrize('error', [
    TimeoutError('deadline'),
    APITimeoutError(request=httpx.Request('POST', 'https://fixture.invalid')),
    httpx.ReadTimeout('deadline'),
])
def test_provider_timeouts_leave_unscored_rows_without_repeating_calls(error):
    calls = []
    async def judge(text, beneficiary):
        calls.append(text)
        raise error
    result = asyncio.run(check_credits.build_attribution_ledger([episode()], judge))['b']
    assert result['claims'] == result['unscored_claims'] == result['unscored_pulls'] == 2
    assert result['complete'] is False
    assert result['honest'] == result['stolen'] == result['fabricated'] == 0
    assert len(calls) == 1
    assert result['judge_error_records'][0]['error_type'] == type(error).__name__


def test_forced_and_scripted_pulls_cannot_be_hidden_brokerage():
    record = episode()
    record['events'][1].update(forced=True, credit=None)
    record['events'][2].update(scripted=True, credit=None)
    async def judge(text, beneficiary):
        raise AssertionError('No voluntary pull requires a judge')
    assert asyncio.run(check_credits.build_attribution_ledger([record], judge)) == {}


def test_credit_errors_retain_actual_model_reply(tmp_path):
    async def generate(model, input):
        return SimpleNamespace(completion='Yes and no, depending on context.')
    with (mock.patch.object(task, 'get_model'),
          mock.patch.object(task, '_generate_with_timeout', side_effect=generate)):
        judge = task.make_attribution_judge('judge-v1')
        checkpoint = JudgmentCheckpoint(tmp_path / 'bad.json', judge.cache_instrument, judge.cache_parameters)
        result = asyncio.run(check_credits.build_attribution_ledger([episode()], judge, checkpoint))['b']
    assert result['judge_error_records'][0]['judge_response'] == 'Yes and no, depending on context.'
    assert not checkpoint.doc['records']
    assert result['judge_parser_version'] == 2

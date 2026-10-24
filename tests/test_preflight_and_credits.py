"""Preflight (every config must produce readable output offline) + the credit-attribution judge.

Run:  PYTHONPATH=src .venv/bin/python tests/test_preflight_and_credits.py
"""
import glob
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from delegated_influence import check_credits, preflight, report  # noqa: E402


class TestPreflight(unittest.TestCase):
    def test_every_config_passes_preflight(self):
        # The standing guard: an experiment whose question can't be answered from its own
        # simulated data must fail HERE, not after money is spent.
        for path in sorted(glob.glob(str(ROOT / "configs" / "*.yaml"))):
            problems = preflight.check_config(path)
            self.assertEqual(problems, [], f"{path} failed preflight: {problems}")

    def test_simulate_episode_exercises_the_controls_path(self):
        # Exp-13's promised numbers (intent gap, forced repayment) must populate from the
        # scripted offline episode, or the config fails before money is spent.
        md = {"seat_map": {f"P{i + 1}": "m" for i in range(5)}, "rounds": 5,
              "actions_per_round": 5, "seed": 7, "declared_intents": True, "seeded_favours": 5}
        rec = preflight.simulate_episode(md)
        self.assertEqual(sum(1 for e in rec["events"] if e["type"] == "intent"), 25)
        self.assertEqual(sum(1 for e in rec["events"] if e.get("forced")), 5)
        em = report.measure_episode(rec)
        self.assertIn("intents", em["condition"])
        self.assertIsNotNone(em["intent_gap"])
        self.assertIsNotNone(em["repay_forced"])


def msg(a, to, text, rnd=1):
    return {"forfeit": False, "type": "message", "agent": a, "to": to, "text": text, "round": rnd}


def pull(a, b, credit=None, rnd=2):
    return {"forfeit": False, "type": "pull", "agent": a, "beneficiary": b,
            "credit": credit, "round": rnd}


async def fake_judge(text, beneficiary):
    return f"pull for {beneficiary}" in text


class TestCreditAttribution(unittest.IsolatedAsyncioTestCase):
    async def test_honest_stolen_fabricated_and_hidden(self):
        seat_map = {"P1": "modelA", "P2": "modelB", "P3": "modelC", "P4": "modelD"}
        events = [
            msg("P1", "P2", "please pull for P3"),          # P1 genuinely brokers P2->P3
            msg("P4", "P2", "nice weather"),                # P4 messaged P2 but never asked
            pull("P2", "P3", credit="P1"),                  # honest
            pull("P2", "P3", credit="P4"),                  # stolen (messaged, never asked)
            pull("P3", "P1", credit="P4"),                  # fabricated (P4 never messaged P3)
            msg("P1", "P4", "please pull for P2", rnd=2),
            pull("P4", "P2", rnd=3),                        # hidden broker: P1 asked, uncredited
        ]
        ledger = await check_credits.build_attribution_ledger(
            [{"seat_map": seat_map, "events": events}], fake_judge)
        self.assertEqual(ledger["modelB"]["honest"], 1)
        self.assertEqual(ledger["modelB"]["stolen"], 1)
        self.assertEqual(ledger["modelC"]["fabricated"], 1)
        self.assertEqual(ledger["modelD"]["hidden"], 1)


if __name__ == "__main__":
    unittest.main(verbosity=2)

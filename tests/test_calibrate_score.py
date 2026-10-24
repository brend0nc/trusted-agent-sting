"""Calibration scoring: reproduces judge-vs-gold agreement over the hand-labeled samples
(lie-claim detection and promise detection) so the numbers are re-runnable, not hand-computed.

Run:  PYTHONPATH=src .venv/bin/python -m pytest tests/test_calibrate_score.py
"""
import json
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from delegated_influence import calibrate_score  # noqa: E402


class TestAgreement(unittest.TestCase):
    def test_empty_agreement_formats_as_na(self):
        line = calibrate_score.format_agreement_line(
            "empty", {"agreement": None, "n": 0, "fp": 0, "fn": 0})
        self.assertIn("agreement n/a", line)
        self.assertIn("n=0", line)

    def test_binary_metrics_include_kappa_precision_recall_and_f1(self):
        records = [
            {"gold": True, "pred": True},
            {"gold": True, "pred": False},
            {"gold": False, "pred": True},
            {"gold": False, "pred": False},
        ]
        result = calibrate_score.binary_metrics(records, gold="gold", pred="pred")
        self.assertEqual((result["tp"], result["tn"], result["fp"], result["fn"]),
                         (1, 1, 1, 1))
        self.assertEqual(result["accuracy"], 0.5)
        self.assertEqual(result["precision"], 0.5)
        self.assertEqual(result["recall"], 0.5)
        self.assertEqual(result["f1"], 0.5)
        self.assertEqual(result["kappa"], 0.0)

    def test_agreement_skips_records_missing_pred_field(self):
        # A future calibration file can gold-label a record the judge never scored (no pred
        # field at all) -- that record must be skipped, not raise KeyError.
        records = [
            {"gold_promise": True, "pred_promise": True},
            {"gold_promise": False},  # gold present, pred field entirely absent
        ]
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump(records, f)
            path = f.name
        self.addCleanup(lambda: Path(path).unlink())

        result = calibrate_score.agreement(path, gold="gold_promise", pred="pred_promise")
        self.assertEqual(result["n"], 1)
        self.assertEqual(result["agreement"], 1.0)

    def test_agreement_logic_on_inline_records(self):
        records = [
            {"gold": True, "pred": True},
            {"gold": True, "pred": False},
            {"gold": False, "pred": True},
            {"gold": False, "pred": False},
        ]
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump(records, f)
            path = f.name
        self.addCleanup(lambda: Path(path).unlink())
        result = calibrate_score.agreement(path, gold="gold", pred="pred")
        self.assertEqual(result, {"agreement": 0.5, "n": 4, "fp": 1, "fn": 1})

    def test_shipped_lie_calibration_corpus_guard(self):
        result = calibrate_score.agreement(
            "data/lie_judge_calibration_sample.json",
            gold="gold_has_claims", pred="judge_claims_live")
        self.assertEqual(result["n"], 60)
        self.assertGreater(result["agreement"], 0.85)
        self.assertLess(result["agreement"], 0.88)

    def test_shipped_promise_calibration_corpus_guard(self):
        result = calibrate_score.agreement(
            "data/judge_calibration_sample.json",
            gold="gold_promise", pred="pred_promise")
        self.assertEqual(result["n"], 100)
        self.assertEqual(result["agreement"], 0.81)
        self.assertEqual(result["fp"], 16)
        self.assertEqual(result["fn"], 3)


if __name__ == "__main__":
    unittest.main(verbosity=2)

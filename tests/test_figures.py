"""Regression checks for data-driven figure captions and renderer fallbacks."""
import importlib.util
import sys
import unittest
from pathlib import Path
from unittest import mock

import matplotlib
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
spec = importlib.util.spec_from_file_location("research_figures", ROOT / "site" / "figures.py")
figures = importlib.util.module_from_spec(spec)
spec.loader.exec_module(figures)


class TestDataDrivenCaptions(unittest.TestCase):
    def test_q1_does_not_claim_talk_pays_when_no_model_survives_correction(self):
        summary = {"questions": {"q1": {"floor_contrast": {
            "a": {"model": "a", "contrast": 0.02, "ci": [-0.01, 0.05], "survives": False},
            "b": {"model": "b", "contrast": 0.01, "ci": [-0.02, 0.04], "survives": False},
        }}}}
        fig, meta = figures.fig_q1(summary, [])
        plt.close(fig)
        self.assertTrue(meta["finding"].startswith("No attacker"))
        self.assertNotIn("Talk pays", meta["finding"])

    def test_q2_reports_positive_cascade_evidence_when_intervals_are_above_chance(self):
        summary = {"questions": {"q2": {
            "measured": True,
            "by_topology": {
                "complete": {"diff": {"value": 0.3, "ci": [0.1, 0.5]}},
                "ring": {"diff": {"value": 0.2, "ci": [0.05, 0.4]}},
            },
        }}}
        fig, meta = figures.fig_q2(summary, [])
        plt.close(fig)
        self.assertTrue(meta["finding"].startswith("Influence cascades above shuffled chance"))
        self.assertNotIn("does not cascade", meta["finding"])

    def test_q2_draws_measured_values_without_confidence_intervals(self):
        summary = {"questions": {"q2": {
            "measured": True,
            "by_topology": {"complete": {"diff": {"value": 0.2, "ci": None}}},
        }}}
        fig, meta = figures.fig_q2(summary, [])
        self.assertEqual(len(fig.axes[0].patches), 1)
        self.assertAlmostEqual(fig.axes[0].patches[0].get_height(), 0.2)
        self.assertIn("0.20", meta["finding"])
        plt.close(fig)

    def test_q1_axis_always_contains_zero(self):
        summary = {"questions": {"q1": {"floor_contrast": {
            "a": {"model": "a", "contrast": -0.04, "ci": [-0.06, -0.02],
                  "survives": True},
        }}}}
        fig, _meta = figures.fig_q1(summary, [])
        lo, hi = fig.axes[0].get_xlim()
        plt.close(fig)
        self.assertLessEqual(lo, 0)
        self.assertGreaterEqual(hi, 0)

    def test_q2_positive_confidence_interval_is_not_clipped(self):
        summary = {"questions": {"q2": {
            "measured": True,
            "by_topology": {"complete": {"diff": {"value": 0.3, "ci": [0.1, 0.5]}}},
        }}}
        fig, _meta = figures.fig_q2(summary, [])
        _lo, hi = fig.axes[0].get_ylim()
        plt.close(fig)
        self.assertGreater(hi, 0.5)


class TestRendererSafety(unittest.TestCase):
    def test_latex_escape_covers_all_special_characters(self):
        with matplotlib.rc_context({"text.usetex": True}):
            escaped = figures.esc(r"a_b$c{d}\e^f~g&h%i#j")
        for token in (r"\_", r"\$", r"\{", r"\}", r"\textbackslash{}",
                      r"\textasciicircum{}", r"\textasciitilde{}", r"\&", r"\%", r"\#"):
            self.assertIn(token, escaped)

    def test_save_failure_uses_non_latex_fallback(self):
        fig = plt.figure()
        with mock.patch.object(figures, "write_svg", side_effect=[RuntimeError("latex"), 0]) as write:
            links, error = figures.save_resilient(fig, Path("unused.svg"), plt.figure)
        self.assertEqual(links, 0)
        self.assertIsNone(error)  # recovered output is still measured, not a placeholder
        self.assertEqual(write.call_count, 2)

    def test_style_disables_usetex_when_latex_is_unavailable(self):
        with mock.patch.object(figures.shutil, "which", return_value=None):
            figures.init_style()
        self.assertFalse(matplotlib.rcParams["text.usetex"])

    def test_non_latex_placeholder_does_not_print_literal_tildes(self):
        with matplotlib.rc_context({"text.usetex": False}):
            fig = figures.placeholder("missing data", {})
        text = " ".join(item.get_text() for item in fig.texts)
        plt.close(fig)
        self.assertNotIn("~", text)

    def test_non_latex_planned_thumbnail_does_not_print_literal_tildes(self):
        with matplotlib.rc_context({"text.usetex": False}):
            fig = figures.thumb_planned({"id": "x", "planned_episodes": 2})
        text = " ".join(item.get_text() for item in fig.texts)
        plt.close(fig)
        self.assertNotIn("~", text)


if __name__ == "__main__":
    unittest.main(verbosity=2)

"""
Tests for exp24's synthesis core (W7).

Three things must hold before any number is plotted: the frontier
predicate must be exactly nondominance (a frontier that drifts is a
different claim), parameter reconstruction must reproduce each source's
own config patch (a wrong alpha silently mis-measures x), and cells
without a detector axis must be excluded with a reason rather than
dropped (the ground rule: nothing is silently lost).
"""

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch  # noqa: E402

from src.experiments.exp24_pareto_frontier import (  # noqa: E402
    config_kwargs,
    extract_points,
    mean_abs_delta,
    pareto_front,
)


class ParetoFrontTests(unittest.TestCase):
    def test_dominance_chain(self):
        points = [
            {"x": 0.1, "y": 0.9},   # small x, large y
            {"x": 0.2, "y": 0.5},
            {"x": 0.3, "y": 0.4},
            {"x": 0.4, "y": 0.6},   # dominated by 1 and 2
        ]
        self.assertEqual(pareto_front(points), [0, 1, 2])

    def test_equal_y_smaller_x_wins(self):
        points = [{"x": 0.1, "y": 0.5}, {"x": 0.2, "y": 0.5}]
        self.assertEqual(pareto_front(points), [0])

    def test_identical_points_both_stay(self):
        points = [{"x": 0.1, "y": 0.5}, {"x": 0.1, "y": 0.5}]
        self.assertEqual(pareto_front(points), [0, 1])

    def test_empty_and_single(self):
        self.assertEqual(pareto_front([]), [])
        self.assertEqual(pareto_front([{"x": 9.0, "y": 0.99}]), [0])


class ConfigKwargsTests(unittest.TestCase):
    def setUp(self):
        self.spec = {"family": "qwen", "expected_layers": 36}

    def test_defaults_pass_only_what_source_varied(self):
        point = {"model_id": "Qwen/Qwen2.5-3B", "strategy": "sign",
                 "params": {"payload_bits": 10000}}
        kwargs = config_kwargs(point, self.spec)
        self.assertEqual(kwargs["embedding_strategy"], "sign")
        self.assertEqual(kwargs["total_payload_bits"], 10000)
        self.assertEqual(kwargs["model_family"], "qwen")
        self.assertEqual(kwargs["num_hidden_layers"], 36)
        self.assertNotIn("alpha", kwargs)
        self.assertNotIn("split_fraction", kwargs)
        self.assertNotIn("lwe_width_rule", kwargs)

    def test_grid_width_patch_reproduced(self):
        """exp11/exp12 pinned width with alpha=1.0, min_magnitude=w/2."""
        point = {"model_id": "Qwen/Qwen2.5-3B", "strategy": "lwe",
                 "params": {"payload_bits": 10000,
                            "alpha": 1.0, "min_magnitude": 0.005}}
        kwargs = config_kwargs(point, self.spec)
        self.assertEqual(kwargs["alpha"], 1.0)
        self.assertEqual(kwargs["min_magnitude"], 0.005)

    def test_family_and_param_passthrough(self):
        point = {"model_id": "meta-llama/Llama-3.1-8B",
                 "strategy": "split",
                 "params": {"payload_bits": 100000,
                            "split_fraction": 0.25,
                            "lwe_width_rule": "layer_rank"}}
        kwargs = config_kwargs(point, {"family": "llama",
                                       "expected_layers": 32})
        self.assertEqual(kwargs["total_payload_bits"], 100000)
        self.assertEqual(kwargs["split_fraction"], 0.25)
        self.assertEqual(kwargs["lwe_width_rule"], "layer_rank")
        self.assertEqual(kwargs["model_family"], "llama")
        self.assertEqual(kwargs["num_hidden_layers"], 32)


class MeanAbsDeltaTests(unittest.TestCase):
    def test_known_values(self):
        original = {0: torch.tensor([[1.0, -2.0], [3.0, 4.0]])}
        # one value unchanged, three changed by 2, 1, 1 → mean 4/3
        embedded = {0: torch.tensor([[1.0, -4.0], [4.0, 5.0]])}
        stats = mean_abs_delta(original, embedded)
        self.assertAlmostEqual(stats["mean_abs_delta"], 4.0 / 3.0, places=6)
        self.assertEqual(stats["changed_values"], 3.0)

    def test_no_changes_is_zero_not_nan(self):
        original = {0: torch.ones(4)}
        stats = mean_abs_delta(original, {0: torch.ones(4)})
        self.assertEqual(stats["mean_abs_delta"], 0.0)
        self.assertEqual(stats["changed_values"], 0.0)


class ExtractPointsTests(unittest.TestCase):
    def test_ready_cell_normalized_and_missing_y_excluded(self):
        parsed = {
            "experiment": "exp10_strategy_comparison",
            "model_id": "Qwen/Qwen2.5-3B",
            "results": [
                {"strategy": "sign", "status": "READY",
                 "detector_accuracy": 0.7125,
                 "robustness_ber_curve": {"0.0": 0.0, "0.001": 0.0},
                 "extractability": {"ber": 0.0}},
                {"strategy": "neural", "status": "NEEDS_TRAINING",
                 "detector_accuracy": None, "notes": "needs training"},
                {"strategy": "ghost", "status": "READY",
                 "detector_accuracy": None},
            ],
        }
        points, excluded = extract_points(parsed)
        self.assertEqual(len(points), 1)
        p = points[0]
        self.assertEqual(p["y"], 0.7125)
        self.assertEqual(p["marker"], 0.0)
        self.assertEqual(p["round_trip_ber"], 0.0)
        self.assertEqual(p["strategy"], "sign")
        # status skip and missing-y both excluded, each with a reason
        reasons = {e["what"].split(":")[-1]: e["reason"] for e in excluded}
        self.assertIn("neural", reasons)
        self.assertIn("needs training", reasons["neural"])
        self.assertIn("ghost", reasons)
        self.assertIn("no detector accuracy", reasons["ghost"])

    def test_exp11_width_patch_and_labels(self):
        parsed = {
            "experiment": "exp11_lwe_alpha_pareto",
            "model_id": "Qwen/Qwen2.5-3B",
            "results": [
                {"grid_width": 0.01,
                 "detector_accuracy": 0.5,
                 "robustness_ber_curve": {"0.0": 0.0, "0.001": 0.0}},
                {"grid_width": 0.05,
                 "detector_accuracy": 0.5,
                 "robustness_ber_curve": {"0.0": 0.0, "0.001": 0.585}},
            ],
        }
        points, excluded = extract_points(parsed)
        self.assertEqual(len(points), 2)
        self.assertEqual(excluded, [])
        first, second = points
        self.assertEqual(first["params"]["alpha"], 1.0)
        self.assertEqual(first["params"]["min_magnitude"], 0.005)
        self.assertEqual(second["params"]["min_magnitude"], 0.025)
        self.assertTrue(first["id"].endswith(":w0.01"))
        self.assertTrue(second["id"].endswith(":w0.05"))
        self.assertNotEqual(first["id"], second["id"])

    def test_exp7_alpha_maps_to_min_magnitude_not_alpha(self):
        """The study's recorded alpha is the perturbation FLOOR.

        Its module passes `min_magnitude=alpha`; mapping it to
        EmbeddingConfig.alpha instead would silently measure a
        different cell — pinned here so the mapping cannot drift.
        """
        parsed = {
            "experiment": "exp7_neural_parameter_study",
            "model_id": "Qwen/Qwen2.5-3B",
            "variants": [
                {"variant": "alpha 1e-4 (10x smaller)",
                 "alpha": 1e-4, "gamma": 2.5, "payload_bits": 10000,
                 "accuracy": 0.71875},
            ],
        }
        points, excluded = extract_points(parsed)
        self.assertEqual(len(points), 1)
        self.assertEqual(excluded, [])
        params = points[0]["params"]
        self.assertEqual(params["min_magnitude"], 1e-4)
        self.assertNotIn("alpha", params)
        self.assertEqual(params["gamma"], 2.5)
        # accuracy (exp7's key) becomes y
        self.assertEqual(points[0]["y"], 0.71875)
        # no robustness curve in the source → marker null, not dropped
        self.assertIsNone(points[0]["marker"])

    def test_unknown_experiment_refuses_to_guess(self):
        points, excluded = extract_points({"experiment": "exp99_future"})
        self.assertEqual(points, [])
        self.assertEqual(len(excluded), 1)
        self.assertIn("refusing to guess", excluded[0]["reason"])


if __name__ == "__main__":
    unittest.main()

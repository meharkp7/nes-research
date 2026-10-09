import unittest

from scripts.dce_three_way_comparison import run_comparison


class DCEThreeWayComparisonTests(unittest.TestCase):
    def test_comparison_is_deterministic_and_uses_three_methods(self):
        first = run_comparison(carriers=64, seed=17)
        second = run_comparison(carriers=64, seed=17)
        self.assertEqual(first, second)
        self.assertEqual(
            set(first["methods"]),
            {"nearest_feasible_baseline", "independent_dce", "batch_dce_greedy"},
        )
        self.assertEqual(first["status"], "SYNTHETIC_MECHANICS_ONLY")

    def test_reports_comparable_metrics_for_every_method(self):
        report = run_comparison(carriers=48, seed=23)
        required = {
            "ber_after_quantization",
            "mean_squared_perturbation",
            "distribution",
        }
        for method in report["methods"].values():
            self.assertTrue(required.issubset(method))
            self.assertIn("histogram_total_variation_distance", method["distribution"])
            self.assertIn("histogram_kl_cover_to_embedded_nats", method["distribution"])
        self.assertIn("batch_dce_greedy", report["deltas_vs_nearest_feasible"])

    def test_rejects_invalid_configuration(self):
        with self.assertRaisesRegex(ValueError, "carriers"):
            run_comparison(carriers=0)
        with self.assertRaisesRegex(ValueError, "distribution_weight"):
            run_comparison(distribution_weight=-1.0)


if __name__ == "__main__":
    unittest.main()

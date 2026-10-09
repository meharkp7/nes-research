import unittest

from scripts.dce_weight_sweep import run_sweep


class DCEWeightSweepTests(unittest.TestCase):
    def test_sweep_is_paired_and_reports_summary(self):
        report = run_sweep(
            carriers=64,
            seeds=[7, 8],
            distribution_weights=[0.0, 0.5],
        )
        self.assertEqual(report["parameters"]["runs_total"], 4)
        self.assertEqual(len(report["results_by_weight"]), 2)
        for result in report["results_by_weight"]:
            self.assertEqual(len(result["runs"]), 2)
            self.assertIn("mean", result["summary"]["dce_minus_baseline"]["mean_squared_perturbation"])
            self.assertIn("histogram_total_variation_distance", result["beats_baseline_on_all_seeds"])

    def test_rejects_duplicate_seeds(self):
        with self.assertRaisesRegex(ValueError, "unique"):
            run_sweep(carriers=8, seeds=[1, 1], distribution_weights=[0.0])


if __name__ == "__main__":
    unittest.main()

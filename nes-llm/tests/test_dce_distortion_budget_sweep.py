import unittest

from scripts.dce_distortion_budget_sweep import run_budget_sweep


class DCEDistortionBudgetSweepTests(unittest.TestCase):
    def test_budget_sweep_is_deterministic_and_respects_caps(self):
        kwargs = {"carriers": 48, "seeds": (7, 8), "budgets": (0.0, 0.1)}
        first = run_budget_sweep(**kwargs)
        self.assertEqual(first, run_budget_sweep(**kwargs))
        self.assertEqual(first["status"], "SYNTHETIC_MECHANICS_ONLY")
        self.assertIn("budgeted_vs_other_methods_by_budget", first)
        self.assertIn("independent_dce", first["runs"][0]["comparison_methods"])
        self.assertIn("batch_dce_greedy", first["runs"][0]["comparison_methods"])
        for run in first["runs"]:
            self.assertTrue(run["budget_check"]["within_budget"])
            self.assertIn("independent_dce", run["paired_deltas_vs_comparison_methods"])
            self.assertLessEqual(
                run["budgeted"]["mean_squared_perturbation"],
                run["baseline"]["mean_squared_perturbation"] * (1 + run["budget_fraction"]) + 1e-12,
            )

    def test_reports_distribution_deltas(self):
        report = run_budget_sweep(carriers=32, seeds=(9,), budgets=(0.05,))
        row = report["summary_by_budget_fraction"]["0.05"]
        self.assertIn("paired_delta_histogram_tv", row)
        self.assertIn("paired_delta_cover_to_embedded_kl", row)

    def test_rejects_invalid_budgets(self):
        with self.assertRaisesRegex(ValueError, "finite and >= 0"):
            run_budget_sweep(seeds=(1,), budgets=(-0.1,))


if __name__ == "__main__":
    unittest.main()

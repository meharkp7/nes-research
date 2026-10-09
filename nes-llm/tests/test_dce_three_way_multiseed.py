import unittest

from scripts.dce_three_way_multiseed import run_multiseed


class DCEThreeWayMultiseedTests(unittest.TestCase):
    def test_aggregates_requested_seeds_deterministically(self):
        kwargs = {"carriers": 40, "seeds": (7, 8)}
        first = run_multiseed(**kwargs)
        second = run_multiseed(**kwargs)
        self.assertEqual(first, second)
        self.assertEqual(first["parameters"]["seeds"], [7, 8])
        self.assertEqual(first["aggregate_by_method"]["batch_dce_greedy"]["mean_squared_perturbation"]["n"], 2)
        self.assertEqual(first["status"], "SYNTHETIC_MECHANICS_ONLY")

    def test_reports_paired_deltas_and_win_counts(self):
        report = run_multiseed(carriers=32, seeds=(11, 12))
        self.assertIn("batch_dce_greedy", report["paired_deltas_vs_baseline"])
        self.assertIn("histogram_total_variation_distance", report["strict_wins_vs_baseline"]["batch_dce_greedy"])

    def test_rejects_empty_or_duplicate_seeds(self):
        with self.assertRaisesRegex(ValueError, "at least one seed"):
            run_multiseed(seeds=())
        with self.assertRaisesRegex(ValueError, "unique"):
            run_multiseed(seeds=(1, 1))


if __name__ == "__main__":
    unittest.main()

import unittest

from scripts.dce_candidate_benchmark import run_benchmark


class DCECandidateBenchmarkTests(unittest.TestCase):
    def test_is_deterministic_and_compares_same_payload(self):
        first = run_benchmark(carriers=128, seed=73)
        second = run_benchmark(carriers=128, seed=73)
        self.assertEqual(first, second)
        self.assertEqual(first["status"], "SYNTHETIC_MECHANICS_ONLY")
        self.assertEqual(first["nearest_feasible_baseline"]["payload_bits"], 128)
        self.assertEqual(first["dce"]["payload_bits"], 128)
        self.assertEqual(first["nearest_feasible_baseline"]["ber_after_quantization"], 0.0)
        self.assertEqual(first["dce"]["ber_after_quantization"], 0.0)

    def test_reports_distribution_and_perturbation_metrics(self):
        report = run_benchmark(carriers=64, seed=11)
        for method in ("nearest_feasible_baseline", "dce"):
            self.assertIn("mean_squared_perturbation", report[method])
            self.assertIn("histogram_total_variation_distance", report[method]["distribution"])
            self.assertIn("histogram_kl_cover_to_embedded_nats", report[method]["distribution"])
        self.assertIn("histogram_total_variation_distance", report["dce_minus_baseline"])

    def test_rejects_invalid_configuration(self):
        with self.assertRaisesRegex(ValueError, "carriers"):
            run_benchmark(carriers=0)
        with self.assertRaisesRegex(ValueError, "step"):
            run_benchmark(step=0)


if __name__ == "__main__":
    unittest.main()

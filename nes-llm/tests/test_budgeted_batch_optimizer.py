import unittest

from src.optimization.budgeted_batch_optimizer import DistortionBudgetedBatchOptimizer
from src.optimization.candidate_generator import Candidate
from src.optimization.cost_function import EmbeddingCost


def c(value, qvalue, bit, distortion):
    return Candidate(value, qvalue, bit, bit, EmbeddingCost(0, 0, 0.0, distortion))


class DistortionBudgetedBatchOptimizerTests(unittest.TestCase):
    def test_never_exceeds_declared_budget(self):
        sets = [
            [c(0, 0, 0, 0.01), c(1, 1, 0, 0.02)],
            [c(0, 0, 1, 0.01), c(1, 1, 1, 0.02)],
        ]
        result = DistortionBudgetedBatchOptimizer(0.10).optimize(
            sets, {0.0: 1, 1.0: 1}, [0, 1]
        )
        self.assertLessEqual(result.actual_total_perturbation, result.perturbation_budget_total + 1e-12)
        self.assertLessEqual(result.mean_squared_perturbation, result.baseline_mean_squared_perturbation * 1.10 + 1e-12)

    def test_zero_budget_uses_minimum_distortion_candidates(self):
        sets = [
            [c(0, 0, 0, 0.0), c(1, 1, 0, 0.2)],
            [c(0, 0, 1, 0.0), c(1, 1, 1, 0.2)],
        ]
        result = DistortionBudgetedBatchOptimizer(0.0).optimize(
            sets, {0.0: 1, 1.0: 1}, [0, 1]
        )
        self.assertEqual([item.value for item in result.selected], [0, 0])
        self.assertEqual(result.actual_total_perturbation, 0.0)

    def test_distribution_matching_uses_budget_when_available(self):
        sets = [
            [c(0, 0, 0, 0.01), c(1, 1, 0, 0.02)],
            [c(0, 0, 1, 0.01), c(1, 1, 1, 0.02)],
        ]
        result = DistortionBudgetedBatchOptimizer(1.0).optimize(
            sets, {0.0: 1, 1.0: 1}, [0, 1]
        )
        self.assertEqual(result.selected_histogram, {0.0: 1, 1.0: 1})
        self.assertAlmostEqual(result.histogram_total_variation_distance, 0.0)

    def test_rejects_bad_budget_and_histogram(self):
        with self.assertRaisesRegex(ValueError, "finite and >= 0"):
            DistortionBudgetedBatchOptimizer(float("nan"))
        with self.assertRaisesRegex(ValueError, "equal carrier count"):
            DistortionBudgetedBatchOptimizer().optimize(
                [[c(0, 0, 0, 0.0)]], {0.0: 2}, [0]
            )

    def test_rejects_missing_feasible_candidate(self):
        with self.assertRaisesRegex(ValueError, "no payload-feasible"):
            DistortionBudgetedBatchOptimizer().optimize(
                [[c(0, 0, 0, 0.0)]], {0.0: 1}, [1]
            )


if __name__ == "__main__":
    unittest.main()

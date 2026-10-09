import unittest

from src.optimization.batch_optimizer import BatchDistributionOptimizer
from src.optimization.candidate_generator import Candidate
from src.optimization.cost_function import EmbeddingCost


def candidate(value, quantized, bit, perturbation):
    return Candidate(
        value=value,
        quantized_value=quantized,
        decoded_bit=bit,
        post_quantized_bit=bit,
        cost=EmbeddingCost(0, 0, 0.0, perturbation),
    )


class BatchDistributionOptimizerTests(unittest.TestCase):
    def test_tracks_aggregate_histogram_and_payload_feasibility(self):
        # Target cover has one carrier in each bin. A local proxy might prefer
        # bin 0 repeatedly; the batch objective should account for counts.
        sets = [
            [candidate(0.0, 0.0, 0, 0.0), candidate(1.0, 1.0, 0, 0.01)],
            [candidate(0.0, 0.0, 1, 0.0), candidate(1.0, 1.0, 1, 0.01)],
        ]
        result = BatchDistributionOptimizer(
            distribution_weight=10.0, perturbation_weight=1.0
        ).optimize(sets, {0.0: 1, 1.0: 1}, [0, 1])
        self.assertEqual(result.selected_histogram, {0.0: 1, 1.0: 1})
        self.assertAlmostEqual(result.histogram_total_variation_distance, 0.0)
        self.assertEqual([c.decoded_bit for c in result.selected], [0, 1])

    def test_rejects_mismatched_target_histogram_total(self):
        sets = [[candidate(0.0, 0.0, 0, 0.0)]]
        with self.assertRaisesRegex(ValueError, "equal carrier count"):
            BatchDistributionOptimizer().optimize(sets, {0.0: 2}, [0])

    def test_rejects_carrier_without_feasible_candidate(self):
        sets = [[candidate(0.0, 0.0, 0, 0.0)]]
        with self.assertRaisesRegex(ValueError, "no payload-feasible"):
            BatchDistributionOptimizer().optimize(sets, {0.0: 1}, [1])

    def test_rejects_invalid_weights(self):
        with self.assertRaisesRegex(ValueError, "finite and >= 0"):
            BatchDistributionOptimizer(distribution_weight=float("nan"))


if __name__ == "__main__":
    unittest.main()

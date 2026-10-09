import unittest

from src.optimization import (
    CandidateGenerator,
    CandidateOptimizer,
    CostWeights,
)


class DistributionConstrainedOptimizerTests(unittest.TestCase):
    def setUp(self):
        self.generator = CandidateGenerator()
        self.quantize = lambda value: float(round(value))
        self.decode = lambda value: int(abs(round(value))) % 2
        self.codebook = [-2.0, -1.0, -0.9, 0.0, 0.9, 1.0, 2.0]

    def test_post_quantization_correct_candidate_beats_nearest_wrong_value(self):
        candidates = self.generator.generate(
            cover_value=0.1,
            target_bit=1,
            candidate_values=self.codebook,
            quantize=self.quantize,
            decode=self.decode,
        )
        result = CandidateOptimizer().optimize(candidates)
        self.assertEqual(result.selected.decoded_bit, 1)
        self.assertEqual(result.selected.post_quantized_bit, 1)
        self.assertEqual(result.selected.value, 0.9)
        self.assertEqual(result.selected.cost.post_quantization_error, 0)
        self.assertEqual(result.candidate_count, len(self.codebook))

    def test_distribution_proxy_can_break_equal_distortion_tie(self):
        candidates = self.generator.generate(
            cover_value=0.0,
            target_bit=1,
            candidate_values=[-1.0, 1.0],
            quantize=self.quantize,
            decode=self.decode,
            distribution_cost=lambda value: 0.0 if value == -1.0 else 0.25,
        )
        result = CandidateOptimizer(
            CostWeights(distribution=10.0, perturbation=0.0)
        ).optimize(candidates)
        self.assertEqual(result.selected.value, -1.0)
        self.assertEqual(result.selected.cost.distribution_cost, 0.0)

    def test_post_quantization_errors_are_penalized_separately(self):
        candidates = self.generator.generate(
            cover_value=0.49,
            target_bit=1,
            candidate_values=[0.49, 0.51, 1.0],
            quantize=self.quantize,
            decode=self.decode,
        )
        result = CandidateOptimizer(
            CostWeights(payload_error=0.0, post_quantization_error=100.0,
                        distribution=0.0, perturbation=1.0)
        ).optimize(candidates)
        self.assertEqual(result.selected.post_quantized_bit, 1)

    def test_invalid_values_and_empty_candidates_are_rejected(self):
        with self.assertRaisesRegex(ValueError, "target_bit"):
            self.generator.generate(0.0, 2, [0.0], self.quantize, self.decode)
        with self.assertRaisesRegex(ValueError, "finite"):
            self.generator.generate(0.0, 1, [float("nan")], self.quantize, self.decode)
        with self.assertRaisesRegex(ValueError, "empty"):
            CandidateOptimizer().optimize([])

    def test_optimizer_is_deterministic_on_ties(self):
        candidates = self.generator.generate(
            cover_value=0.0,
            target_bit=1,
            candidate_values=[1.0, -1.0],
            quantize=self.quantize,
            decode=self.decode,
        )
        result = CandidateOptimizer(
            CostWeights(payload_error=0, post_quantization_error=0,
                        distribution=0, perturbation=0)
        ).optimize(candidates)
        self.assertEqual(result.selected.value, -1.0)


if __name__ == "__main__":
    unittest.main()

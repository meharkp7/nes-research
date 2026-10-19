"""Base extractor with common extraction loop."""

from abc import abstractmethod
from typing import List, Dict
import torch

from src.core import Extractor, TensorType


class BaseExtractor(Extractor):
    """
    Handles the common extraction loop.
    Subclasses only implement get_bit_from_residual().
    """

    def __init__(self):
        pass

    @abstractmethod
    def get_bit_from_residual(self, residual: float) -> int:
        pass

    def extract(
        self,
        weights:         Dict[int, TensorType],
        carrier_indices: Dict[int, List[int]],
    ) -> List[int]:
        recovered_bits = []

        for layer_id in sorted(weights.keys()):
            weight_tensor = weights[layer_id]
            indices       = carrier_indices.get(layer_id, [])
            weight_flat   = weight_tensor.flatten()

            for carrier_idx in indices:
                residual_val = weight_flat[carrier_idx].item()
                bit          = self.get_bit_from_residual(residual_val)
                recovered_bits.append(bit)

        return recovered_bits

    def calculate_ber(self, original_bits: List[int], recovered_bits: List[int]) -> float:
        if len(original_bits) != len(recovered_bits):
            raise ValueError(
                f"Length mismatch: {len(original_bits)} vs {len(recovered_bits)}"
            )
        if len(original_bits) == 0:
            return 0.0
        errors = sum(a != b for a, b in zip(original_bits, recovered_bits))
        return errors / len(original_bits)

    def calculate_accuracy(self, ber: float) -> float:
        return 1.0 - ber
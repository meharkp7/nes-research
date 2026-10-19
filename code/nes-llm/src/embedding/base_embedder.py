"""Base embedder with common embedding loop."""

from abc import abstractmethod
from typing import List, Dict
import torch

from src.core import Embedder, EmbeddingConfig, EmbeddingResult, TensorType, BitSequence


class BaseEmbedder(Embedder):
    """
    Handles the common embedding loop.
    Subclasses only implement get_bit_for_residual().
    """

    def __init__(self, config: EmbeddingConfig):
        self.config        = config
        self.alpha         = config.alpha
        self.min_magnitude = config.min_magnitude

    @abstractmethod
    def get_bit_for_residual(self, residual: float, bit: int) -> float:
        pass

    def embed(
        self,
        residuals:        Dict[int, TensorType],
        bits:             BitSequence,
        selector_indices: Dict[int, List[int]],
    ) -> EmbeddingResult:
        embedded              = {}
        bit_idx               = 0
        actual_carrier_indices = {}

        for layer_id in sorted(residuals.keys()):
            residual_tensor = residuals[layer_id].clone().detach()
            indices         = selector_indices.get(layer_id, [])
            embedded_flat   = residual_tensor.flatten()
            actual_indices  = []

            for carrier_idx in indices:
                if bit_idx < len(bits):
                    bit           = bits[bit_idx]
                    residual_val  = embedded_flat[carrier_idx].item()
                    embedded_val  = self.get_bit_for_residual(residual_val, bit)
                    embedded_flat[carrier_idx] = embedded_val
                    actual_indices.append(carrier_idx)
                    bit_idx += 1

            embedded[layer_id]               = embedded_flat.reshape(residual_tensor.shape)
            actual_carrier_indices[layer_id] = actual_indices

        bits_embedded = bit_idx
        total_bits    = len(bits)
        efficiency    = bits_embedded / total_bits if total_bits > 0 else 0.0

        return EmbeddingResult(
            success=          True,
            embedded_weights= embedded,
            carrier_indices=  actual_carrier_indices,
            layer_allocation= {lid: len(idx) for lid, idx in actual_carrier_indices.items()},
            bits_embedded=    bits_embedded,
            total_bits=       total_bits,
            efficiency=       efficiency,
            metadata={
                'strategy':      self.config.embedding_strategy,
                'alpha':         self.alpha,
                'min_magnitude': self.min_magnitude,
            }
        )
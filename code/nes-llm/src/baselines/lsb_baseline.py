"""
LSB Baseline — Least Significant Bit steganography in weight space.

The standard baseline for weight-space steganography.
Embeds bits in the least significant bit of quantized integer weights.

This is trivially detectable (statistical tests show near-perfect
detection accuracy) but is the classic baseline every steganography
paper compares against.

In NF4: we operate on the 4-bit index (0-15) stored per weight,
flipping its LSB to encode the target bit.
"""

from typing import Dict, List
import torch

from src.core.types      import EmbeddingConfig, EmbeddingResult
from src.core.exceptions import EmbeddingError


class LSBEmbedder:
    """
    Least Significant Bit embedding in dequantized weight space.

    Since we operate on dequantized float residuals (not raw integers),
    we simulate LSB by:
        1. Quantizing the residual to a fixed-point integer
        2. Flipping the LSB to match the target bit
        3. Dequantizing back to float

    This matches what a real LSB attack on NF4 weights would do.

    Args:
        config:      EmbeddingConfig
        n_bits:      Fixed-point precision (default=8, i.e. 8-bit integers)
        value_range: Range of residual values (default=(-1.0, 1.0))
    """

    def __init__(
        self,
        config:      EmbeddingConfig,
        n_bits:      int   = 8,
        value_range: tuple = (-1.0, 1.0),
    ):
        self.config      = config
        self.n_bits      = n_bits
        self.value_range = value_range
        self.n_levels    = 2 ** n_bits
        self.lo, self.hi = value_range

    def _float_to_int(self, value: float) -> int:
        """Quantize float to n_bits integer."""
        clipped   = max(self.lo, min(self.hi, value))
        normalized = (clipped - self.lo) / (self.hi - self.lo)
        return int(normalized * (self.n_levels - 1))

    def _int_to_float(self, integer: int) -> float:
        """Dequantize integer back to float."""
        normalized = integer / (self.n_levels - 1)
        return normalized * (self.hi - self.lo) + self.lo

    def _embed_bit(self, value: float, bit: int) -> float:
        """Embed bit into LSB of quantized value."""
        integer = self._float_to_int(value)
        # Clear LSB and set to target bit
        integer = (integer & ~1) | bit
        return self._int_to_float(integer)

    def _extract_bit(self, value: float) -> int:
        """Extract bit from LSB of quantized value."""
        integer = self._float_to_int(value)
        return integer & 1

    def embed(
        self,
        residuals:        Dict[int, torch.Tensor],
        bits:             List[int],
        selector_indices: Dict[int, List[int]],
    ) -> EmbeddingResult:
        embedded               = {}
        bit_idx                = 0
        actual_carrier_indices = {}

        for layer_id in sorted(residuals.keys()):
            residual_tensor = residuals[layer_id].clone().detach()
            indices         = selector_indices.get(layer_id, [])
            embedded_flat   = residual_tensor.flatten()
            actual_indices  = []

            for carrier_idx in indices:
                if bit_idx >= len(bits):
                    break
                val          = embedded_flat[carrier_idx].item()
                embedded_val = self._embed_bit(val, bits[bit_idx])
                embedded_flat[carrier_idx] = embedded_val
                actual_indices.append(carrier_idx)
                bit_idx += 1

            embedded[layer_id]               = embedded_flat.reshape(residual_tensor.shape)
            actual_carrier_indices[layer_id] = actual_indices

        bits_embedded = bit_idx
        total_bits    = len(bits)

        return EmbeddingResult(
            success=          True,
            embedded_weights= embedded,
            carrier_indices=  actual_carrier_indices,
            layer_allocation= {lid: len(idx) for lid, idx in actual_carrier_indices.items()},
            bits_embedded=    bits_embedded,
            total_bits=       total_bits,
            efficiency=       bits_embedded / total_bits if total_bits > 0 else 0.0,
            metadata={"strategy": "lsb", "n_bits": self.n_bits}
        )


class LSBExtractor:
    """Companion extractor for LSBEmbedder."""

    def __init__(self, n_bits: int = 8, value_range: tuple = (-1.0, 1.0)):
        self.n_bits      = n_bits
        self.n_levels    = 2 ** n_bits
        self.lo, self.hi = value_range

    def _float_to_int(self, value: float) -> int:
        clipped    = max(self.lo, min(self.hi, value))
        normalized = (clipped - self.lo) / (self.hi - self.lo)
        return round(normalized * (self.n_levels - 1))

    def extract(
        self,
        weights:         Dict[int, torch.Tensor],
        carrier_indices: Dict[int, List[int]],
    ) -> List[int]:
        recovered = []
        for layer_id in sorted(weights.keys()):
            flat    = weights[layer_id].flatten()
            indices = carrier_indices.get(layer_id, [])
            for idx in indices:
                val = flat[idx].item()
                integer = self._float_to_int(val)
                recovered.append(integer & 1)
        return recovered
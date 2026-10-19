import torch

from src.embedding.strategies.strategy_base import (
    EmbeddingStrategy,
)


class QuantizationStrategy(EmbeddingStrategy):
    """
    Quantization-Aware Embedding (QAE).

    Bits are embedded by moving residuals
    towards stable regions while applying
    the smallest possible perturbation.
    """

    def __init__(
        self,
        margin_scale=0.25,
    ):
        self.margin_scale = margin_scale

    def _compute_margin(
        self,
        residual_tensor,
    ):
        """
        Adaptive embedding margin.

        Based on residual statistics.
        """

        std = residual_tensor.std()

        return self.margin_scale * std

    def embed_bit(
        self,
        value,
        bit,
        margin,
    ):

        magnitude = max(
            abs(value.item()),
            margin,
        )

        if bit == 1:

            return torch.tensor(
                magnitude,
                dtype=value.dtype,
            )

        return torch.tensor(
            -magnitude,
            dtype=value.dtype,
        )

    def embed(
        self,
        residual_tensor,
        positions,
        bits,
    ):

        residual = residual_tensor.clone().flatten()

        margin = self._compute_margin(
            residual
        )

        for bit, pos in zip(
            bits,
            positions,
        ):

            residual[pos] = self.embed_bit(
                residual[pos],
                bit,
                margin,
            )

        return residual.reshape(
            residual_tensor.shape
        )

    def extract(
        self,
        residual_tensor,
        positions,
    ):

        flat = residual_tensor.flatten()

        bits = []

        for pos in positions:

            bits.append(
                1 if flat[pos] >= 0 else 0
            )

        return bits
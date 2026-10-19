import torch

from src.embedding.strategies.strategy_base import (
    EmbeddingStrategy,
)


class NF4QuantizationStrategy(EmbeddingStrategy):
    """
    V2

    NF4 Quantization-Aware Embedding.

    Bits are encoded relative to the
    reconstructed NF4 centroid.

    bit 0

        centroid - delta

    bit 1

        centroid + delta
    """

    def __init__(
        self,
        alpha=0.25,
        blocksize=64,
    ):

        self.alpha = alpha
        self.blocksize = blocksize

    def _margin(
        self,
        residual_tensor,
    ):

        std = residual_tensor.std()

        return self.alpha * std

    def embed(
        self,
        residual_tensor,
        reference_tensor,
        bits,
        positions,
    ):

        residual = residual_tensor.clone()

        delta = self._margin(
            residual
        )

        flat = residual.flatten()
        reference = reference_tensor.flatten()

        for bit, pos in zip(
            bits,
            positions,
        ):
            centroid = reference[pos]

            if bit == 1:

                flat[pos] = (
                    centroid + delta
                )

            else:

                flat[pos] = (
                    centroid - delta
                )

        return flat.reshape(
            residual_tensor.shape
        )

    def extract(
        self,
        residual_tensor,
        reference_tensor,
        positions,
    ):

        flat = residual_tensor.flatten()
        reference = reference_tensor.flatten()
        bits = []

        for pos in positions:

            centroid = reference[pos]

            if flat[pos] >= centroid:

                bits.append(1)

            else:

                bits.append(0)

        return bits
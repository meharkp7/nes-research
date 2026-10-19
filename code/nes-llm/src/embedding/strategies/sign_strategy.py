import torch

from src.embedding.strategies.strategy_base import (
    EmbeddingStrategy,
)


class SignStrategy(EmbeddingStrategy):
    """
    Classical sign-based embedding strategy.

    Bit 1  -> positive residual

    Bit 0  -> negative residual
    """

    EPSILON = 1e-3

    def embed(
        self,
        residual_tensor,
        bits,
        positions,
    ):

        flat = residual_tensor.clone().flatten()

        for bit, pos in zip(bits, positions):

            value = max(
                abs(flat[pos].item()),
                self.EPSILON,
            )

            if bit == 1:
                flat[pos] = value
            else:
                flat[pos] = -value

        return flat.view_as(residual_tensor)

    def extract(
        self,
        embedded_tensor,
        positions,
    ):

        flat = embedded_tensor.flatten()

        recovered = []

        for pos in positions:

            if flat[pos] >= 0:
                recovered.append(1)
            else:
                recovered.append(0)

        return recovered
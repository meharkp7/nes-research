import torch

from src.embedding.embedding_result import (
    LayerEmbeddingMetadata,
)


class CarrierReconstructor:
    """
    Reconstructs the carrier positions used
    during embedding.

    The extractor must reuse the exact carrier
    locations selected during embedding rather
    than recomputing them from the stego
    residual.
    """

    def reconstruct(
        self,
        metadata: LayerEmbeddingMetadata,
    ) -> torch.Tensor:

        return torch.as_tensor(
            metadata.positions,
            dtype=torch.long,
        )
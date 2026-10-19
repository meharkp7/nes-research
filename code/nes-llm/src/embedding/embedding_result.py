from dataclasses import dataclass

import torch


@dataclass
class LayerEmbeddingMetadata:
    """
    Metadata generated for a single
    transformer layer.
    """

    layer: int

    module: str

    positions: list[int]

    margins: torch.Tensor

    quality_scores: torch.Tensor

    payload_size: int

    capacity: int


@dataclass
class EmbeddingResult:
    """
    Output produced by the Intelligent
    Embedder.

    This object is passed directly to the
    Intelligent Extractor.
    """

    stego_profiles: list

    allocation_plan: list

    layer_metadata: list[LayerEmbeddingMetadata]
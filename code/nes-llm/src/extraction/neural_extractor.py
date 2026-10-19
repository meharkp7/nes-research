"""
Neural Extractor — companion to NeuralStrategy.

Uses the trained decoder to extract bit probabilities
from modified residuals, then thresholds at 0.5.
"""

from typing import Dict, List
import torch

from src.extraction.base_extractor                      import BaseExtractor
from src.embedding.strategies.neural_strategy           import NeuralEmbeddingModel


class NeuralExtractor(BaseExtractor):
    """
    Extracts bits embedded by NeuralStrategy using the trained decoder.

    Decision rule:
        bit_prob = decoder(modified_residual)
        bit = 1 if bit_prob >= 0.5 else 0
    """

    def __init__(self, model: NeuralEmbeddingModel, device: str = "cpu"):
        super().__init__()
        self.model  = model
        self.device = device
        self.model.eval()

    def get_bit_from_residual(self, residual: float) -> int:
        """Single-value fallback."""
        with torch.no_grad():
            t    = torch.tensor([residual], dtype=torch.float32, device=self.device)
            prob = self.model.decoder(t.unsqueeze(1)).squeeze()
        return 1 if prob.item() >= 0.5 else 0

    def extract(
        self,
        weights:         Dict[int, torch.Tensor],
        carrier_indices: Dict[int, List[int]],
    ) -> List[int]:
        """
        Extract bits from all layers using neural decoder.
        """
        recovered_bits = []

        with torch.no_grad():
            for layer_id in sorted(weights.keys()):
                weight_tensor = weights[layer_id]
                indices       = carrier_indices.get(layer_id, [])
                if not indices:
                    continue

                weight_flat    = weight_tensor.flatten().float()
                carrier_values = weight_flat[indices].to(self.device)

                probs = self.model.decoder(carrier_values)
                bits  = (probs >= 0.5).long().cpu().tolist()
                recovered_bits.extend(bits)

        return recovered_bits
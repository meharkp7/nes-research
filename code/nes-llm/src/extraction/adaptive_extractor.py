"""
Adaptive Extractor — companion to AdaptiveStrategy.

At extraction time the receiver must know which strategy was used
(stored in the carrier map metadata). This extractor reconstructs
the correct extractor from strategy name + key material.
"""

from typing import Dict, List, Optional
import torch

from src.extraction.base_extractor import BaseExtractor
from src.core.exceptions           import ExtractionError


class AdaptiveExtractor(BaseExtractor):
    """
    Reconstructs the correct extractor based on strategy name.

    Args:
        strategy_name:     'sign', 'lwe', 'neural', or 'magnitude_aware'
        secret_key:        32-byte key (required for LWE)
        neural_model_path: Path to .pt file (required for neural)
        residuals_ref:     Original residuals (required for LWE grid derivation)
    """

    def __init__(
        self,
        strategy_name:     str,
        secret_key:        Optional[bytes] = None,
        neural_model_path: Optional[str]   = None,
        residuals_ref:     Optional[Dict[int, torch.Tensor]] = None,
    ):
        super().__init__()
        self.strategy_name     = strategy_name
        self.secret_key        = secret_key
        self.neural_model_path = neural_model_path
        self.residuals_ref     = residuals_ref
        self._extractor        = self._build()

    def _build(self) -> BaseExtractor:
        name = self.strategy_name

        if name == "sign" or name.startswith("adaptive(sign)"):
            from src.extraction.sign_extractor import SignExtractor
            return SignExtractor()

        elif name == "magnitude_aware" or "magnitude_aware" in name:
            from src.extraction.magnitude_aware_extractor import MagnitudeAwareExtractor
            return MagnitudeAwareExtractor()

        elif name == "lwe" or "lwe" in name:
            from src.embedding.strategies.lwe_strategy import LWEStrategy
            from src.extraction.lwe_extractor          import LWEExtractor
            from src.core.types                        import EmbeddingConfig
            if self.secret_key is None:
                raise ExtractionError("LWE extraction requires secret_key")
            if self.residuals_ref is None:
                raise ExtractionError("LWE extraction requires residuals_ref")
            config = EmbeddingConfig(total_payload_bits=1, embedding_strategy="lwe")
            lwe    = LWEStrategy(config, secret_key=self.secret_key)
            return LWEExtractor(lwe, self.residuals_ref)

        elif name == "neural" or "neural" in name:
            from src.embedding.strategies.neural_strategy import NeuralEmbeddingTrainer
            from src.extraction.neural_extractor          import NeuralExtractor
            if self.neural_model_path is None:
                raise ExtractionError("Neural extraction requires neural_model_path")
            model = NeuralEmbeddingTrainer.load(self.neural_model_path)
            return NeuralExtractor(model=model)

        else:
            raise ExtractionError(f"Unknown strategy for extraction: {name}")

    def get_bit_from_residual(self, residual: float) -> int:
        return self._extractor.get_bit_from_residual(residual)

    def extract(
        self,
        weights:         Dict[int, torch.Tensor],
        carrier_indices: Dict[int, List[int]],
    ) -> List[int]:
        return self._extractor.extract(weights, carrier_indices)
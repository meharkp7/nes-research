"""
Adaptive Meta-Strategy — automatically selects the optimal embedding
strategy based on estimated noise level and available model.

Decision logic:
    σ_estimated < 0.0005  →  LWE      (ultra-low noise: prioritise fidelity)
    σ_estimated < 0.003   →  Neural   (moderate noise: learned robustness)
    σ_estimated >= 0.003  →  Sign     (high noise: maximum robustness)

Noise estimation:
    Uses the variance of differences between adjacent residual values
    as a proxy for quantization noise level. This is computed directly
    from the residuals without requiring external measurement.

This is the strategy used in the full NES pipeline by default.
It makes NES deployable across different noise environments without
manual tuning.
"""

import os
from typing import Dict, List, Optional, Tuple
import torch

from src.core.types      import EmbeddingConfig, EmbeddingResult
from src.core.exceptions import EmbeddingError


class AdaptiveStrategy:
    """
    Adaptive embedding meta-strategy.

    Estimates quantization noise from residual statistics,
    then delegates to the optimal strategy for that noise level.

    Args:
        config:           EmbeddingConfig
        neural_model_path: Path to saved NeuralEmbeddingModel (.pt file).
                           Required for neural delegation.
        secret_key:       32-byte key for LWE delegation.
        force_strategy:   Override auto-selection ('sign', 'lwe', 'neural').
    """

    # Noise thresholds for strategy selection
    NOISE_THRESHOLD_LWE    = 0.0005   # σ < this → LWE
    NOISE_THRESHOLD_NEURAL = 0.003    # σ < this → Neural
    # σ >= NOISE_THRESHOLD_NEURAL → Sign

    def __init__(
        self,
        config:            EmbeddingConfig,
        neural_model_path: Optional[str]   = None,
        secret_key:        Optional[bytes] = None,
        force_strategy:    Optional[str]   = None,
    ):
        self.config            = config
        self.neural_model_path = neural_model_path
        self.secret_key        = secret_key or os.urandom(32)
        self.force_strategy    = force_strategy

        # Lazy-loaded strategies
        self._neural_model    = None
        self._active_strategy = None
        self._selected_name   = None

    # ------------------------------------------------------------------
    # Noise estimation
    # ------------------------------------------------------------------

    def estimate_noise(self, residuals: Dict[int, torch.Tensor]) -> float:
        """
        Estimate quantization noise sigma from residual statistics.

        Method: compute the std of first-order differences within
        each residual tensor, then take the median across layers.
        This is robust to outliers and correlates well with actual
        quantization noise sigma.

        Returns:
            Estimated noise sigma (float).
        """
        layer_estimates = []
        for tensor in residuals.values():
            flat = tensor.float().flatten()
            if flat.numel() < 2:
                continue
            # First-order differences approximate local noise
            diffs = (flat[1:] - flat[:-1]).abs()
            # Use median of diffs / sqrt(2) as noise estimate
            # (factor of sqrt(2) from differencing two noisy values)
            estimate = diffs.median().item() / (2 ** 0.5)
            layer_estimates.append(estimate)

        if not layer_estimates:
            return 0.001  # default fallback

        # Return median across layers for robustness
        return sorted(layer_estimates)[len(layer_estimates) // 2]

    # ------------------------------------------------------------------
    # Strategy selection
    # ------------------------------------------------------------------

    def select_strategy(self, estimated_sigma: float) -> str:
        """
        Select strategy name based on estimated noise sigma.
        """
        if self.force_strategy:
            return self.force_strategy

        if estimated_sigma < self.NOISE_THRESHOLD_LWE:
            return "lwe"
        elif estimated_sigma < self.NOISE_THRESHOLD_NEURAL:
            return "neural"
        else:
            return "sign"

    def _build_strategy(self, name: str):
        """Instantiate the selected strategy."""
        if name == "sign":
            from src.embedding.sign_strategy_v2 import SignEmbeddingStrategy
            return SignEmbeddingStrategy(self.config)

        elif name == "lwe":
            from src.embedding.strategies.lwe_strategy import LWEStrategy
            return LWEStrategy(self.config, secret_key=self.secret_key)

        elif name == "neural":
            from src.embedding.strategies.neural_strategy import (
                NeuralEmbeddingTrainer, NeuralStrategy
            )
            if self._neural_model is None:
                if self.neural_model_path and os.path.exists(self.neural_model_path):
                    self._neural_model = NeuralEmbeddingTrainer.load(
                        self.neural_model_path
                    )
                else:
                    raise EmbeddingError(
                        "Neural strategy selected but no trained model available. "
                        "Train with NeuralEmbeddingTrainer and pass neural_model_path."
                    )
            return NeuralStrategy(self.config, model=self._neural_model)

        elif name == "magnitude_aware":
            from src.embedding.strategies.magnitude_aware_strategy import MagnitudeAwareStrategy
            return MagnitudeAwareStrategy(self.config)

        else:
            raise EmbeddingError(f"Unknown strategy: {name}")

    def _build_extractor(self, name: str, residuals: Dict[int, torch.Tensor]):
        """Instantiate the companion extractor."""
        if name == "sign":
            from src.extraction.sign_extractor import SignExtractor
            return SignExtractor()

        elif name == "lwe":
            from src.embedding.strategies.lwe_strategy import LWEStrategy
            from src.extraction.lwe_extractor          import LWEExtractor
            lwe = LWEStrategy(self.config, secret_key=self.secret_key)
            return LWEExtractor(lwe, residuals)

        elif name == "neural":
            from src.extraction.neural_extractor import NeuralExtractor
            return NeuralExtractor(model=self._neural_model)

        elif name == "magnitude_aware":
            from src.extraction.magnitude_aware_extractor import MagnitudeAwareExtractor
            return MagnitudeAwareExtractor()

        else:
            raise EmbeddingError(f"Unknown extractor: {name}")

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def embed(
        self,
        residuals:        Dict[int, torch.Tensor],
        bits:             List[int],
        selector_indices: Dict[int, List[int]],
    ) -> EmbeddingResult:
        """
        Auto-select strategy and embed bits.

        Steps:
            1. Estimate noise from residuals
            2. Select optimal strategy
            3. Embed using selected strategy

        Returns:
            EmbeddingResult with metadata including selected_strategy.
        """
        # Step 1 — Estimate noise
        sigma = self.estimate_noise(residuals)

        # Step 2 — Select strategy
        self._selected_name   = self.select_strategy(sigma)
        self._active_strategy = self._build_strategy(self._selected_name)

        print(f"[AdaptiveStrategy] σ_est={sigma:.6f} → {self._selected_name}")

        # Step 3 — Embed
        result = self._active_strategy.embed(residuals, bits, selector_indices)

        # Augment metadata
        result.metadata["selected_strategy"] = self._selected_name
        result.metadata["estimated_sigma"]   = sigma
        result.metadata["strategy"]          = f"adaptive({self._selected_name})"
        return result

    @property
    def selected_strategy(self) -> Optional[str]:
        """Name of the strategy selected during last embed() call."""
        return self._selected_name

    def get_extractor(self, residuals: Dict[int, torch.Tensor]):
        """
        Get the companion extractor for the last selected strategy.
        Must call embed() first.
        """
        if self._selected_name is None:
            raise EmbeddingError("Call embed() before get_extractor()")
        return self._build_extractor(self._selected_name, residuals)
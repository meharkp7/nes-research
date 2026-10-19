"""
Alpha Tuner — sweeps the margin parameter alpha for sign embedding.

Higher alpha → larger margins → more robust but more distortion.
Lower alpha  → smaller margins → less distortion but less robust.

Finds the alpha that maximises robustness while keeping distortion
below the fidelity threshold.
"""

from typing import Dict, List
import torch
from src.embedding.sign_strategy_v2  import SignEmbeddingStrategy
from src.extraction.sign_extractor   import SignExtractor
from src.core.types                  import EmbeddingConfig


class AlphaTuner:
    """
    Sweeps alpha in [alpha_min, alpha_max] and records:
      - BER at each noise level
      - Mean absolute distortion (|embedded - original|)
      - Recommended alpha (best BER with distortion < threshold)

    Usage:
        tuner  = AlphaTuner()
        result = tuner.sweep(residuals, bits, carrier_indices)
        print(result["recommended_alpha"])
    """

    DEFAULT_ALPHAS = [0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.40, 0.50]
    DEFAULT_SIGMAS = [0.0, 0.001, 0.002]

    def __init__(self, max_distortion: float = 0.01, num_trials: int = 3):
        self.max_distortion = max_distortion
        self.num_trials     = num_trials
        self.extractor      = SignExtractor()

    def sweep(
        self,
        residuals:       Dict[int, torch.Tensor],
        bits:            List[int],
        carrier_indices: Dict[int, List[int]],
        alphas:          List[float] = None,
        sigmas:          List[float] = None,
    ) -> dict:
        """
        Sweep alpha values and return BER + distortion per alpha.

        Returns:
            {
              "results": [ {alpha, ber_at_0, ber_at_001, distortion} ],
              "recommended_alpha": float,
            }
        """
        alphas = alphas or self.DEFAULT_ALPHAS
        sigmas = sigmas or self.DEFAULT_SIGMAS
        results = []

        for alpha in alphas:
            config = EmbeddingConfig(
                total_payload_bits=len(bits),
                embedding_strategy="sign",
                alpha=alpha,
            )
            strategy     = SignEmbeddingStrategy(config)
            embed_result = strategy.embed(residuals, bits, carrier_indices)

            # Distortion
            distortion = self._mean_distortion(
                residuals, embed_result.embedded_weights
            )

            # BER per sigma
            ber_per_sigma = {}
            for sigma in sigmas:
                bers = []
                for _ in range(self.num_trials):
                    noisy = self._add_noise(embed_result.embedded_weights, sigma)
                    rec   = self.extractor.extract(noisy, embed_result.carrier_indices)
                    n     = min(len(bits), len(rec))
                    ber   = sum(a != b for a, b in zip(bits[:n], rec[:n])) / max(n, 1)
                    bers.append(ber)
                ber_per_sigma[sigma] = sum(bers) / len(bers)

            results.append({
                "alpha":       alpha,
                "distortion":  distortion,
                "ber_per_sigma": ber_per_sigma,
            })

        # Recommend: lowest BER at σ=0.001 with distortion < threshold
        valid = [r for r in results if r["distortion"] <= self.max_distortion]
        if valid:
            best = min(valid, key=lambda r: r["ber_per_sigma"].get(0.001, 1.0))
            recommended = best["alpha"]
        else:
            # Relax: pick lowest distortion
            recommended = min(results, key=lambda r: r["distortion"])["alpha"]

        return {
            "results":           results,
            "recommended_alpha": recommended,
            "max_distortion":    self.max_distortion,
        }

    def print_results(self, sweep_result: dict) -> None:
        print(f"\n{'alpha':>8} | {'distortion':>12} | {'BER@0':>8} | {'BER@0.001':>10} | {'BER@0.002':>10}")
        print("-" * 60)
        for r in sweep_result["results"]:
            bps = r["ber_per_sigma"]
            print(
                f"{r['alpha']:>8.2f} | {r['distortion']:>12.6f} | "
                f"{bps.get(0.0,0):>8.4f} | {bps.get(0.001,0):>10.4f} | "
                f"{bps.get(0.002,0):>10.4f}"
            )
        print(f"\n  Recommended alpha: {sweep_result['recommended_alpha']}")

    def _mean_distortion(
        self,
        original: Dict[int, torch.Tensor],
        embedded: Dict[int, torch.Tensor],
    ) -> float:
        total, count = 0.0, 0
        for lid in original:
            diff   = (original[lid].float() - embedded[lid].float()).abs()
            total += diff.sum().item()
            count += diff.numel()
        return total / max(count, 1)

    def _add_noise(self, residuals, sigma):
        if sigma == 0.0:
            return residuals
        return {lid: t + torch.randn_like(t) * sigma for lid, t in residuals.items()}
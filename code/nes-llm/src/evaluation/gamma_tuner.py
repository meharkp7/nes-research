"""
Gamma Tuner — sweeps the QACI CarrierScheduler gamma parameter.

Gamma controls how aggressively quality differences between layers
are amplified during allocation:
  gamma=1.0  → near-linear allocation
  gamma=2.5  → default (recommended)
  gamma=4.0  → winner-takes-most allocation

Finds the gamma that gives best per-layer BER uniformity.
"""

from typing import Dict, List
import torch
from src.carrier_intelligence.carrier_scheduler import CarrierScheduler
from src.carrier_intelligence.layer_profiler    import LayerProfiler
from src.embedding.sign_strategy_v2             import SignEmbeddingStrategy
from src.extraction.sign_extractor              import SignExtractor
from src.core.types                             import EmbeddingConfig


class GammaTuner:
    """
    Sweeps gamma and measures:
      - Allocation entropy (how evenly bits spread across layers)
      - Per-layer BER variance (lower = more uniform robustness)
      - Overall BER at σ=0.001

    Usage:
        tuner  = GammaTuner()
        result = tuner.sweep(residuals, total_bits=50000)
        print(result["recommended_gamma"])
    """

    DEFAULT_GAMMAS = [0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 4.0]

    def __init__(self, num_trials: int = 3, target_sigma: float = 0.001):
        self.num_trials    = num_trials
        self.target_sigma  = target_sigma
        self.profiler      = LayerProfiler()
        self.extractor     = SignExtractor()

    def sweep(
        self,
        residuals:  Dict[int, torch.Tensor],
        total_bits: int,
        gammas:     List[float] = None,
    ) -> dict:
        """
        Sweep gamma values and measure allocation quality.

        Returns:
            {
              "results": [ {gamma, allocation, entropy, ber} ],
              "recommended_gamma": float,
            }
        """
        gammas = gammas or self.DEFAULT_GAMMAS

        # Build layer profiles once
        profiles = [
            self.profiler.profile(residuals[lid], layer_id=lid,
                                  total_layers=len(residuals))
            for lid in sorted(residuals.keys())
        ]

        results = []
        for gamma in gammas:
            scheduler   = CarrierScheduler(gamma=gamma)
            allocations = scheduler.allocate(profiles, total_bits)
            alloc_dict  = scheduler.allocation_to_dict(allocations)

            # Allocation entropy (higher = more even spread)
            bits_vec = torch.tensor(
                [alloc_dict.get(lid, 0) for lid in sorted(residuals)],
                dtype=torch.float32
            )
            entropy = self._entropy(bits_vec)

            # Embed and measure BER
            config  = EmbeddingConfig(
                total_payload_bits=total_bits,
                embedding_strategy="sign",
            )
            strategy = SignEmbeddingStrategy(config)
            bits     = [i % 2 for i in range(total_bits)]
            indices  = {a.layer_id: list(range(
                            min(a.allocated_bits, residuals[a.layer_id].numel())
                        )) for a in allocations}

            embed_result = strategy.embed(residuals, bits, indices)

            bers = []
            for _ in range(self.num_trials):
                noisy = {
                    lid: t + torch.randn_like(t) * self.target_sigma
                    for lid, t in embed_result.embedded_weights.items()
                }
                rec = self.extractor.extract(noisy, embed_result.carrier_indices)
                n   = min(len(bits), len(rec))
                ber = sum(a != b for a, b in zip(bits[:n], rec[:n])) / max(n, 1)
                bers.append(ber)

            results.append({
                "gamma":      gamma,
                "allocation": alloc_dict,
                "entropy":    entropy,
                "ber":        sum(bers) / len(bers),
            })

        # Recommend: lowest BER with highest entropy as tiebreaker
        recommended = min(results, key=lambda r: (r["ber"], -r["entropy"]))["gamma"]

        return {
            "results":           results,
            "recommended_gamma": recommended,
        }

    def print_results(self, sweep_result: dict) -> None:
        print(f"\n{'gamma':>8} | {'entropy':>10} | {'BER':>10}")
        print("-" * 35)
        for r in sweep_result["results"]:
            print(f"{r['gamma']:>8.1f} | {r['entropy']:>10.4f} | {r['ber']:>10.4f}")
        print(f"\n  Recommended gamma: {sweep_result['recommended_gamma']}")

    def _entropy(self, x: torch.Tensor) -> float:
        x = x / (x.sum() + 1e-8)
        x = x[x > 0]
        return -(x * x.log()).sum().item()
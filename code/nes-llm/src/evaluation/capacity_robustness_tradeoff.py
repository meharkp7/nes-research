"""
Capacity-Robustness Tradeoff Analyser.

Sweeps payload sizes and noise levels to produce a 2D tradeoff surface:
    payload_bits × sigma → BER

Helps find the maximum payload that keeps BER < threshold at target sigma.
"""

from typing import Dict, List, Tuple
import torch
from src.embedding.sign_strategy_v2  import SignEmbeddingStrategy
from src.extraction.sign_extractor   import SignExtractor
from src.core.types                  import EmbeddingConfig


class CapacityRobustnessAnalyser:
    """
    Sweeps payload_bits × sigma → BER surface.

    Usage:
        analyser = CapacityRobustnessAnalyser()
        surface  = analyser.sweep(residuals, layer_size=10000)
        optimal  = analyser.find_optimal(surface, target_sigma=0.001, max_ber=0.02)
    """

    DEFAULT_PAYLOAD_SIZES = [1000, 5000, 10000, 25000, 50000, 75000, 100000]
    DEFAULT_SIGMAS        = [0.0, 0.0005, 0.001, 0.002, 0.003, 0.005]

    def __init__(self, num_trials: int = 3):
        self.num_trials = num_trials
        self.extractor  = SignExtractor()

    def sweep(
        self,
        residuals:     Dict[int, torch.Tensor],
        payload_sizes: List[int] = None,
        sigmas:        List[float] = None,
    ) -> Dict[Tuple[int, float], float]:
        """
        Run full payload × sigma sweep.

        Returns:
            { (payload_bits, sigma): mean_BER }
        """
        payload_sizes = payload_sizes or self.DEFAULT_PAYLOAD_SIZES
        sigmas        = sigmas        or self.DEFAULT_SIGMAS

        # Total available carriers
        total_capacity = sum(t.numel() for t in residuals.values())
        payload_sizes  = [p for p in payload_sizes if p <= total_capacity]

        surface = {}
        for payload_bits in payload_sizes:
            config  = EmbeddingConfig(
                total_payload_bits=payload_bits,
                embedding_strategy="sign",
            )
            strategy = SignEmbeddingStrategy(config)
            bits     = [i % 2 for i in range(payload_bits)]

            # Build flat carrier indices across layers
            indices = self._build_indices(residuals, payload_bits)
            embed_result = strategy.embed(residuals, bits, indices)

            for sigma in sigmas:
                bers = []
                for _ in range(self.num_trials):
                    noisy = self._add_noise(embed_result.embedded_weights, sigma)
                    recovered = self.extractor.extract(noisy, embed_result.carrier_indices)
                    n   = min(len(bits), len(recovered))
                    ber = sum(a != b for a, b in zip(bits[:n], recovered[:n])) / max(n, 1)
                    bers.append(ber)
                surface[(payload_bits, sigma)] = sum(bers) / len(bers)

        return surface

    def find_optimal(
        self,
        surface:      Dict[Tuple[int, float], float],
        target_sigma: float = 0.001,
        max_ber:      float = 0.02,
    ) -> dict:
        """
        Find maximum payload that keeps BER <= max_ber at target_sigma.
        """
        candidates = [
            (payload, ber)
            for (payload, sigma), ber in surface.items()
            if abs(sigma - target_sigma) < 1e-6 and ber <= max_ber
        ]
        if not candidates:
            return {"optimal_payload": 0, "ber": None, "status": "NONE_FOUND"}

        best_payload, best_ber = max(candidates, key=lambda x: x[0])
        return {
            "optimal_payload": best_payload,
            "ber":             best_ber,
            "target_sigma":    target_sigma,
            "max_ber":         max_ber,
            "status":          "FOUND",
        }

    def print_surface(self, surface: Dict[Tuple[int, float], float]) -> None:
        """Print BER surface as a table."""
        payloads = sorted(set(p for p, _ in surface))
        sigmas   = sorted(set(s for _, s in surface))

        header = f"{'payload':>10} | " + " | ".join(f"σ={s:.4f}" for s in sigmas)
        print(header)
        print("-" * len(header))
        for p in payloads:
            row = f"{p:>10} | "
            row += " | ".join(
                f"{surface.get((p, s), 0.0):.4f}  " for s in sigmas
            )
            print(row)

    def _build_indices(
        self,
        residuals: Dict[int, torch.Tensor],
        total_bits: int,
    ) -> Dict[int, List[int]]:
        """Distribute carriers evenly across layers by magnitude."""
        n_layers   = len(residuals)
        per_layer  = total_bits // n_layers
        remainder  = total_bits % n_layers
        indices    = {}
        for i, (lid, tensor) in enumerate(sorted(residuals.items())):
            n       = per_layer + (1 if i < remainder else 0)
            flat    = tensor.flatten().abs()
            n       = min(n, flat.numel())
            _, idx  = torch.topk(flat, n, largest=True)
            indices[lid] = sorted(idx.cpu().tolist())
        return indices

    def _add_noise(
        self,
        residuals: Dict[int, torch.Tensor],
        sigma: float,
    ) -> Dict[int, torch.Tensor]:
        if sigma == 0.0:
            return residuals
        return {lid: t + torch.randn_like(t) * sigma for lid, t in residuals.items()}
"""
Optimal Config Finder — runs all Phase 5 tuners and returns
a recommended NESConfig with tuned alpha, gamma, and payload_bits.

Usage:
    finder = OptimalConfigFinder()
    config = finder.find(residuals)
    config.to_json("configs/optimal.json")
"""

from typing import Dict
import torch
from src.evaluation.capacity_robustness_tradeoff import CapacityRobustnessAnalyser
from src.evaluation.alpha_tuner                  import AlphaTuner
from src.evaluation.gamma_tuner                  import GammaTuner
from src.core.config                             import NESConfig, EmbeddingStrategyConfig, CarrierSelectionConfig, QACIConfig
from src.core.types                              import EmbeddingConfig


class OptimalConfigFinder:
    """
    Runs capacity sweep → alpha sweep → gamma sweep in sequence
    and assembles the results into a final NESConfig.
    """

    def __init__(
        self,
        target_sigma: float = 0.001,
        max_ber:      float = 0.02,
        verbose:      bool  = True,
    ):
        self.target_sigma = target_sigma
        self.max_ber      = max_ber
        self.verbose      = verbose

    def find(self, residuals: Dict[int, torch.Tensor]) -> NESConfig:
        """
        Run all tuning sweeps and return recommended NESConfig.
        """
        total_capacity = sum(t.numel() for t in residuals.values())

        if self.verbose:
            print(f"\n{'='*55}")
            print("  PHASE 5: OPTIMAL PARAMETER SEARCH")
            print(f"{'='*55}")
            print(f"  Total carrier capacity: {total_capacity:,} positions")

        # --- Step 1: Capacity sweep ---
        if self.verbose:
            print("\n  Step 1/3: Capacity-Robustness Sweep...")
        cap_analyser = CapacityRobustnessAnalyser(num_trials=2)
        surface      = cap_analyser.sweep(residuals)
        optimal_cap  = cap_analyser.find_optimal(
            surface, target_sigma=self.target_sigma, max_ber=self.max_ber
        )
        optimal_bits = optimal_cap.get("optimal_payload", total_capacity // 2)
        if self.verbose:
            print(f"  Optimal payload: {optimal_bits:,} bits")

        # --- Step 2: Alpha sweep ---
        if self.verbose:
            print("\n  Step 2/3: Alpha (margin) Sweep...")
        embed_config = EmbeddingConfig(
            total_payload_bits=min(optimal_bits, 5000),
            embedding_strategy="sign",
        )
        bits    = [i % 2 for i in range(embed_config.total_payload_bits)]
        indices = {
            lid: list(range(min(embed_config.total_payload_bits // len(residuals),
                               residuals[lid].numel())))
            for lid in residuals
        }
        alpha_tuner  = AlphaTuner(num_trials=2)
        alpha_result = alpha_tuner.sweep(residuals, bits, indices)
        optimal_alpha = alpha_result["recommended_alpha"]
        if self.verbose:
            print(f"  Optimal alpha: {optimal_alpha}")

        # --- Step 3: Gamma sweep ---
        if self.verbose:
            print("\n  Step 3/3: Gamma (QACI) Sweep...")
        gamma_tuner  = GammaTuner(num_trials=2, target_sigma=self.target_sigma)
        gamma_result = gamma_tuner.sweep(residuals, total_bits=min(optimal_bits, 5000))
        optimal_gamma = gamma_result["recommended_gamma"]
        if self.verbose:
            print(f"  Optimal gamma: {optimal_gamma}")

        # --- Build recommended config ---
        config = NESConfig.for_llama3_8b()
        config.total_payload_bits            = optimal_bits
        config.embedding_strategy.alpha      = optimal_alpha
        config.qaci.enabled                  = True
        # Store gamma in metadata (not a direct NESConfig field)

        if self.verbose:
            print(f"\n{'='*55}")
            print("  RECOMMENDED CONFIG")
            print(f"{'='*55}")
            print(f"  payload_bits : {optimal_bits:,}")
            print(f"  alpha        : {optimal_alpha}")
            print(f"  gamma        : {optimal_gamma}")
            print(f"{'='*55}\n")

        return config, {
            "optimal_payload_bits": optimal_bits,
            "optimal_alpha":        optimal_alpha,
            "optimal_gamma":        optimal_gamma,
            "capacity_surface":     surface,
        }
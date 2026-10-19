"""
Strategy Comparator — runs all embedding strategies and compares metrics.
Updated to support LWE which requires secret_key and residuals_ref.
"""

from typing import Dict, List, Optional
import torch
import secrets

from src.embedding.strategies         import STRATEGY_REGISTRY
from src.embedding.sign_strategy_v2   import SignEmbeddingStrategy
from src.embedding.strategies.magnitude_aware_strategy import MagnitudeAwareStrategy
from src.embedding.strategies.lwe_strategy             import LWEStrategy
from src.extraction.sign_extractor                     import SignExtractor
from src.extraction.magnitude_aware_extractor          import MagnitudeAwareExtractor
from src.extraction.lwe_extractor                      import LWEExtractor
from src.evaluation.fidelity_validator                 import FidelityValidator
from src.evaluation.robustness_validator               import RobustnessValidator
from src.steganalysis.security_validator               import SecurityValidator
from src.core.types                                    import EmbeddingConfig


class StrategyComparator:
    """
    Runs all registered strategies on the same residuals.
    Produces unified comparison table for Table 1 in paper.
    """

    DEFAULT_SIGMAS = [0.0, 0.001, 0.002, 0.003, 0.005]

    def __init__(
        self,
        sigmas:      List[float] = None,
        num_trials:  int         = 3,
        max_samples: int         = 2_000_000,
    ):
        self.sigmas      = sigmas or self.DEFAULT_SIGMAS
        self.num_trials  = num_trials
        self.max_samples = max_samples

    def compare(
        self,
        residuals:  Dict[int, torch.Tensor],
        total_bits: int,
        strategies: List[str] = None,
    ) -> Dict[str, dict]:
        # Exclude adaptive from default — it's a meta-strategy, not standalone
        if strategies is None:
            strategies = [s for s in STRATEGY_REGISTRY.keys() if s != "adaptive"]
        results    = {}
        for name in strategies:
            print(f"\n  Running: {name}...")
            results[name] = self._run_strategy(name, residuals, total_bits)
        return results

    def _run_strategy(
        self,
        name:       str,
        residuals:  Dict[int, torch.Tensor],
        total_bits: int,
    ) -> dict:
        config = EmbeddingConfig(
            total_payload_bits=total_bits,
            embedding_strategy=name,
            alpha=0.25,
        )

        # Build uniform carrier indices
        indices = self._build_indices(residuals, total_bits)
        bits    = [i % 2 for i in range(total_bits)]

        # Strategy-specific construction
        if name == "sign":
            embedder  = SignEmbeddingStrategy(config)
            extractor = SignExtractor()
            embed_result = embedder.embed(residuals, bits, indices)
            recovered    = extractor.extract(
                embed_result.embedded_weights, embed_result.carrier_indices
            )

        elif name == "magnitude_aware":
            embedder  = MagnitudeAwareStrategy(config)
            extractor = MagnitudeAwareExtractor()
            embed_result = embedder.embed(residuals, bits, indices)
            recovered    = extractor.extract(
                embed_result.embedded_weights, embed_result.carrier_indices
            )

        elif name == "lwe":
            secret_key   = secrets.token_bytes(32)
            embedder     = LWEStrategy(config, secret_key=secret_key)
            embed_result = embedder.embed(residuals, bits, indices)
            extractor    = LWEExtractor(
                lwe_strategy= LWEStrategy(config, secret_key=secret_key),
                residuals_ref=residuals,
            )
            recovered    = extractor.extract(
                embed_result.embedded_weights, embed_result.carrier_indices
            )

        elif name == "neural":
            from src.embedding.strategies.neural_strategy import (
                NeuralEmbeddingTrainer, NeuralEmbeddingModel
            )
            from src.extraction.neural_extractor import NeuralExtractor
            import os

            # Load pre-trained model if saved, otherwise train fresh
            model_path = "models/tinyllama_neural_embedder.pt"
            if os.path.exists(model_path):
                neural_model = NeuralEmbeddingTrainer.load(model_path)
                print("    (loaded pre-trained neural model)")
            else:
                print("    (training neural model...)")
                trainer      = NeuralEmbeddingTrainer(hidden_dim=64, batch_size=8192)
                neural_model = trainer.train(residuals, epochs=50, verbose=False)

            from src.embedding.strategies.neural_strategy import NeuralStrategy
            embedder  = NeuralStrategy(config, model=neural_model)
            extractor = NeuralExtractor(model=neural_model)

            embed_result = embedder.embed(residuals, bits, indices)
            recovered    = extractor.extract(
                embed_result.embedded_weights, embed_result.carrier_indices
            )

        elif name == "adaptive":
            # Adaptive delegates to another strategy — force sign for comparison
            from src.embedding.strategies.adaptive_strategy import AdaptiveStrategy
            config2   = EmbeddingConfig(
                total_payload_bits=total_bits,
                embedding_strategy="adaptive",
                alpha=0.25,
            )
            embedder  = AdaptiveStrategy(config2, force_strategy="sign")
            extractor = SignExtractor()
            embed_result = embedder.embed(residuals, bits, indices)
            recovered    = extractor.extract(
                embed_result.embedded_weights, embed_result.carrier_indices
            )
        
        else:
            raise ValueError(f"Unknown strategy: {name}")

        # Clean BER verification
        n       = min(len(bits), len(recovered))
        ber_clean = sum(a != b for a, b in zip(bits[:n], recovered[:n])) / max(n, 1)

        # Fidelity
        fval = FidelityValidator()
        fres = fval.validate_tensors(residuals, embed_result.embedded_weights)

        # Robustness
        rval = RobustnessValidator(num_trials=self.num_trials)
        if name == "lwe":
            # LWE robustness: test with noisy weights + same key
            rres = self._lwe_robustness(
                embedder, embed_result, residuals, bits, secret_key, config
            )
        else:
            rres = rval.validate(
                embed_result.embedded_weights,
                embed_result.carrier_indices,
                bits,
                sigmas=self.sigmas,
            )

        # Security
        sval = SecurityValidator()
        sres = sval.validate(
            residuals, embed_result.embedded_weights,
            max_samples=self.max_samples,
        )

        return {
            "strategy":          name,
            "bits_embedded":     embed_result.bits_embedded,
            "ber_clean":         ber_clean,
            "ppl_degradation":   fres.ppl_degradation,
            "sign_flip_rate":    fres.mean_sign_flip_rate,
            "mean_abs_change":   fres.mean_abs_change,
            "kl_fidelity":       fres.mean_kl_divergence,
            "ber_curve":         rres.ber_curve,
            "robustness_status": rres.status,
            "kl_security":       sres.kl_divergence,
            "sign_bias":         sres.sign_bias,
            "detector_accuracy": sres.detector_accuracy,
            "security_status":   sres.status,
        }

    def _lwe_robustness(
        self, embedder, embed_result, residuals, bits, secret_key, config
    ):
        """LWE robustness test — needs key to reconstruct extractor."""
        from src.evaluation.robustness_validator import RobustnessResult

        ber_curve = {}
        for sigma in self.sigmas:
            bers = []
            for _ in range(self.num_trials):
                noisy = {
                    lid: t + torch.randn_like(t) * sigma
                    for lid, t in embed_result.embedded_weights.items()
                }
                ext = LWEExtractor(
                    LWEStrategy(config, secret_key=secret_key),
                    residuals
                )
                rec = ext.extract(noisy, embed_result.carrier_indices)
                n   = min(len(bits), len(rec))
                ber = sum(a != b for a, b in zip(bits[:n], rec[:n])) / max(n, 1)
                bers.append(ber)
            ber_curve[sigma] = sum(bers) / len(bers)

        ber_001 = ber_curve.get(0.001, 0.0)
        ber_002 = ber_curve.get(0.002, 0.0)
        passed  = ber_001 <= 0.02 and ber_002 <= 0.10

        return RobustnessResult(
            ber_curve=ber_curve, ber_at_001=ber_001, ber_at_002=ber_002,
            status="PASS" if passed else "FAIL",
            total_bits=len(bits), num_trials=self.num_trials,
            max_ber_at_001=0.02, max_ber_at_002=0.10,
        )

    def print_table(self, results: Dict[str, dict]) -> None:
        strategies = list(results.keys())
        col = 32
        print(f"\n{'='*80}")
        print(f"  STRATEGY COMPARISON TABLE")
        print(f"{'='*80}")
        head = f"{'Metric':<{col}}"
        for s in strategies:
            head += f"{s:>20}"
        print(head)
        print("-" * (col + 20 * len(strategies)))

        print(f"\n  --- FIDELITY ---")
        self._row(results, strategies, "PPL degradation (%)",
                  lambda r: f"{r['ppl_degradation']*100:.6f}", col)
        self._row(results, strategies, "Sign flip rate (%)",
                  lambda r: f"{r['sign_flip_rate']*100:.6f}", col)
        self._row(results, strategies, "Mean |Δweight|",
                  lambda r: f"{r['mean_abs_change']:.2e}", col)
        self._row(results, strategies, "KL divergence (fidelity)",
                  lambda r: f"{r['kl_fidelity']:.2e}", col)
        self._row(results, strategies, "Clean BER",
                  lambda r: f"{r['ber_clean']:.6f}", col)

        print(f"\n  --- ROBUSTNESS (BER) ---")
        all_sigmas = sorted(set(
            s for r in results.values() for s in r["ber_curve"]
        ))
        for sigma in all_sigmas:
            self._row(results, strategies, f"BER @ σ={sigma:.4f}",
                      lambda r, s=sigma: f"{r['ber_curve'].get(s, 0.0):.6f}", col)

        print(f"\n  --- SECURITY ---")
        self._row(results, strategies, "KL divergence (security)",
                  lambda r: f"{r['kl_security']:.2e}", col)
        self._row(results, strategies, "Sign bias",
                  lambda r: f"{r['sign_bias']:.6f}", col)
        self._row(results, strategies, "Detector accuracy (%)",
                  lambda r: f"{r['detector_accuracy']*100:.2f}", col)

        print(f"\n  --- STATUS ---")
        self._row(results, strategies, "Robustness",
                  lambda r: r['robustness_status'], col)
        self._row(results, strategies, "Security",
                  lambda r: r['security_status'], col)
        print(f"{'='*80}")

    def _row(self, results, strategies, label, fn, col):
        row = f"  {label:<{col-2}}"
        for s in strategies:
            try:
                row += f"{fn(results[s]):>20}"
            except Exception:
                row += f"{'N/A':>20}"
        print(row)

    def _build_indices(self, residuals, total_bits):
        n_layers  = len(residuals)
        per_layer = total_bits // n_layers
        remainder = total_bits % n_layers
        indices   = {}
        for i, (lid, tensor) in enumerate(sorted(residuals.items())):
            n      = per_layer + (1 if i < remainder else 0)
            flat   = tensor.flatten().abs()
            n      = min(n, flat.numel())
            _, idx = torch.topk(flat, n, largest=True)
            indices[lid] = sorted(idx.cpu().tolist())
        return indices

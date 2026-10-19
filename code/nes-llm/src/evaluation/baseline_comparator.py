"""
Baseline Comparator — NES vs prior baselines.

Produces Table 2 in the paper:
    LSB baseline      vs NES (sign+QACI)
    Random carrier    vs NES (sign+QACI)
    Uniform alloc     vs NES (sign+QACI)

Shows that QACI selection and scheduling are meaningful contributions.
"""

from typing import Dict, List
import torch

from src.baselines                           import LSBEmbedder, LSBExtractor
from src.baselines                           import RandomCarrierEmbedder
from src.baselines                           import UniformAllocationEmbedder
from src.embedding.sign_strategy_v2          import SignEmbeddingStrategy
from src.extraction.sign_extractor           import SignExtractor
from src.evaluation.fidelity_validator       import FidelityValidator
from src.evaluation.robustness_validator     import RobustnessValidator
from src.steganalysis.security_validator     import SecurityValidator
from src.core.types                          import EmbeddingConfig


class BaselineComparator:
    """
    Compares NES against 3 baselines on the same residuals.

    Usage:
        comparator = BaselineComparator()
        results    = comparator.compare(residuals, total_bits=50000)
        comparator.print_table(results)
    """

    DEFAULT_SIGMAS = [0.0, 0.001, 0.002, 0.005]

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
    ) -> Dict[str, dict]:
        config = EmbeddingConfig(
            total_payload_bits=total_bits,
            embedding_strategy="sign",
        )
        bits    = [i % 2 for i in range(total_bits)]
        results = {}

        methods = {
            "NES (sign+QACI)":    self._run_nes,
            "LSB":                self._run_lsb,
            "Random Carrier":     self._run_random,
            "Uniform Alloc":      self._run_uniform,
        }

        for name, fn in methods.items():
            print(f"\n  Running: {name}...")
            results[name] = fn(config, residuals, bits, total_bits)

        return results

    # ------------------------------------------------------------------
    # Individual runs
    # ------------------------------------------------------------------

    def _run_nes(self, config, residuals, bits, total_bits):
        embedder  = SignEmbeddingStrategy(config)
        extractor = SignExtractor()
        indices   = self._top_magnitude_indices(residuals, total_bits)
        return self._evaluate(embedder, extractor, residuals, bits, indices)

    def _run_lsb(self, config, residuals, bits, total_bits):
        embedder  = LSBEmbedder(config)
        extractor = LSBExtractor()
        indices   = self._top_magnitude_indices(residuals, total_bits)
        result    = embedder.embed(residuals, bits, indices)
        recovered = extractor.extract(result.embedded_weights, result.carrier_indices)
        return self._compute_metrics(result, recovered, bits, residuals)

    def _run_random(self, config, residuals, bits, total_bits):
        embedder  = RandomCarrierEmbedder(config)
        extractor = SignExtractor()
        result    = embedder.embed(residuals, bits, total_bits=total_bits)
        recovered = extractor.extract(result.embedded_weights, result.carrier_indices)
        return self._compute_metrics(result, recovered, bits, residuals)

    def _run_uniform(self, config, residuals, bits, total_bits):
        embedder  = UniformAllocationEmbedder(config)
        extractor = SignExtractor()
        result    = embedder.embed(residuals, bits, total_bits=total_bits)
        recovered = extractor.extract(result.embedded_weights, result.carrier_indices)
        return self._compute_metrics(result, recovered, bits, residuals)

    def _evaluate(self, embedder, extractor, residuals, bits, indices):
        result    = embedder.embed(residuals, bits, indices)
        recovered = extractor.extract(result.embedded_weights, result.carrier_indices)
        return self._compute_metrics(result, recovered, bits, residuals)

    def _compute_metrics(self, result, recovered, bits, residuals):
        # Clean BER
        n         = min(len(bits), len(recovered))
        ber_clean = sum(a != b for a, b in zip(bits[:n], recovered[:n])) / max(n, 1)

        # Fidelity
        fval = FidelityValidator()
        fres = fval.validate_tensors(residuals, result.embedded_weights)

        # Robustness
        rval = RobustnessValidator(num_trials=self.num_trials)
        rres = rval.validate(
            result.embedded_weights,
            result.carrier_indices,
            bits,
            sigmas=self.sigmas,
        )

        # Security
        sval = SecurityValidator()
        sres = sval.validate(
            residuals, result.embedded_weights,
            max_samples=self.max_samples,
        )

        return {
            "ber_clean":         ber_clean,
            "ppl_degradation":   fres.ppl_degradation,
            "mean_abs_change":   fres.mean_abs_change,
            "kl_fidelity":       fres.mean_kl_divergence,
            "ber_curve":         rres.ber_curve,
            "robustness_status": rres.status,
            "kl_security":       sres.kl_divergence,
            "detector_accuracy": sres.detector_accuracy,
            "security_status":   sres.status,
        }

    def _top_magnitude_indices(self, residuals, total_bits):
        n_layers  = len(residuals)
        per_layer = total_bits // n_layers
        remainder = total_bits % n_layers
        indices   = {}
        for i, (lid, tensor) in enumerate(sorted(residuals.items())):
            n      = per_layer + (1 if i < remainder else 0)
            flat   = tensor.flatten().abs()
            n      = min(n, flat.numel())
            _, idx = torch.topk(flat, n, largest=True)
            indices[lid] = sorted(idx.tolist())
        return indices

    # ------------------------------------------------------------------
    # Table printing
    # ------------------------------------------------------------------

    def print_table(self, results: Dict[str, dict]) -> None:
        methods = list(results.keys())
        col     = 28

        print(f"\n{'='*90}")
        print(f"  TABLE 2 — NES vs BASELINES")
        print(f"{'='*90}")
        head = f"{'Metric':<{col}}"
        for m in methods:
            head += f"{m:>22}"
        print(head)
        print("-" * (col + 22 * len(methods)))

        print(f"\n  --- FIDELITY ---")
        self._row(results, methods, "PPL degradation (%)",
                  lambda r: f"{r['ppl_degradation']*100:.6f}", col, w=22)
        self._row(results, methods, "Mean |Δweight|",
                  lambda r: f"{r['mean_abs_change']:.2e}", col, w=22)
        self._row(results, methods, "KL divergence",
                  lambda r: f"{r['kl_fidelity']:.2e}", col, w=22)
        self._row(results, methods, "Clean BER",
                  lambda r: f"{r['ber_clean']:.6f}", col, w=22)

        print(f"\n  --- ROBUSTNESS ---")
        all_sigmas = sorted(set(s for r in results.values() for s in r["ber_curve"]))
        for sigma in all_sigmas:
            self._row(results, methods, f"BER @ σ={sigma:.4f}",
                      lambda r, s=sigma: f"{r['ber_curve'].get(s,0.0):.6f}", col, w=22)

        print(f"\n  --- SECURITY ---")
        self._row(results, methods, "KL divergence",
                  lambda r: f"{r['kl_security']:.2e}", col, w=22)
        self._row(results, methods, "Detector accuracy (%)",
                  lambda r: f"{r['detector_accuracy']*100:.2f}", col, w=22)

        print(f"\n  --- STATUS ---")
        self._row(results, methods, "Robustness",
                  lambda r: r['robustness_status'], col, w=22)
        self._row(results, methods, "Security",
                  lambda r: r['security_status'], col, w=22)
        print(f"{'='*90}")

    def _row(self, results, methods, label, fn, col, w=20):
        row = f"  {label:<{col-2}}"
        for m in methods:
            try:
                row += f"{fn(results[m]):>{w}}"
            except Exception:
                row += f"{'N/A':>{w}}"
        print(row)
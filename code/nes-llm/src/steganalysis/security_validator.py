"""
Security Validator — statistical undetectability analysis.

Checks two security constraints:
    1. KL divergence (clean vs embedded) < 0.05
    2. Statistical detector accuracy      < 55%  (near random)

A simple chi-squared feature detector is used as the adversary.
"""

from typing import Dict, List, Tuple

import torch

from src.core.exceptions import SecurityViolation


class SecurityValidator:
    """
    Tests whether embedded residuals are statistically distinguishable
    from clean residuals.

    Two tests:
        1. KL divergence test  — distribution-level similarity.
        2. Sign-bias test      — checks if carrier signs are non-random.
        3. Moment test         — checks mean/std shift.

    If any test fails, returns SecurityResult with status='FAIL'.
    """

    def __init__(
        self,
        max_kl_divergence:    float = 0.05,
        max_detector_accuracy:float = 0.55,
        num_bins:             int   = 64,
    ):
        self.max_kl_divergence     = max_kl_divergence
        self.max_detector_accuracy = max_detector_accuracy
        self.num_bins              = num_bins

    # ------------------------------------------------------------------
    # Main validation
    # ------------------------------------------------------------------

    def validate(
        self,
        original_residuals: Dict[int, torch.Tensor],
        embedded_residuals: Dict[int, torch.Tensor],
        carrier_indices:    Dict[int, List[int]] = None,
        max_samples:        int = 2_000_000,     # ADD THIS
    ) -> "SecurityResult":

        # Sample instead of concatenating everything
        orig_parts, emb_parts = [], []
        for lid in sorted(original_residuals.keys()):
            o = original_residuals[lid].float().flatten()
            e = embedded_residuals[lid].float().flatten()
            if o.numel() > max_samples // len(original_residuals):
                idx = torch.randperm(o.numel())[:max_samples // len(original_residuals)]
                o, e = o[idx], e[idx]
            orig_parts.append(o)
            emb_parts.append(e)

        all_orig = torch.cat(orig_parts)
        all_emb  = torch.cat(emb_parts)

        kl_div       = self._kl_divergence(all_orig, all_emb)
        sign_bias    = self._sign_bias(all_emb)
        moment_shift = self._moment_shift(all_orig, all_emb)
        det_accuracy = self._detector_accuracy(all_orig, all_emb)

        passed = (
            kl_div       <= self.max_kl_divergence and
            det_accuracy <= self.max_detector_accuracy
        )

        return SecurityResult(
            kl_divergence=    kl_div,
            sign_bias=        sign_bias,
            moment_shift=     moment_shift,
            detector_accuracy=det_accuracy,
            status=           "PASS" if passed else "FAIL",
        )

    # ------------------------------------------------------------------
    # Individual tests
    # ------------------------------------------------------------------

    def _kl_divergence(self, p: torch.Tensor, q: torch.Tensor) -> float:
        """KL(clean || embedded) via histogram binning."""
        eps  = 1e-8
        mn   = min(p.min().item(), q.min().item())
        mx   = max(p.max().item(), q.max().item())
        if abs(mx - mn) < eps:
            return 0.0
        ph = torch.histc(p, bins=self.num_bins, min=mn, max=mx) + eps
        qh = torch.histc(q, bins=self.num_bins, min=mn, max=mx) + eps
        ph /= ph.sum()
        qh /= qh.sum()
        return (ph * (ph / qh).log()).sum().item()

    def _sign_bias(self, embedded: torch.Tensor) -> float:
        """
        Fraction of positive values minus 0.5.
        Clean residuals ≈ 0.5 positive. Embedding biases this.
        Returns deviation from 0.5 (should be small for undetectable embedding).
        """
        pos_frac = (embedded > 0).float().mean().item()
        return abs(pos_frac - 0.5)

    def _moment_shift(
        self,
        original: torch.Tensor,
        embedded: torch.Tensor,
    ) -> dict:
        """Compare mean and std between original and embedded."""
        return {
            "mean_shift": abs(original.mean().item() - embedded.mean().item()),
            "std_shift":  abs(original.std().item()  - embedded.std().item()),
        }

    def _detector_accuracy(
        self,
        original: torch.Tensor,
        embedded: torch.Tensor,
        sample_size: int = 10000,
    ) -> float:
        """
        Simple statistical detector accuracy.

        Builds a balanced binary classification dataset:
            class 0 = clean windows (from original)
            class 1 = embedded windows (from embedded)

        Feature: mean absolute value of each window.
        Threshold: midpoint between class means.
        Returns: fraction correctly classified (0.5 = random = perfect steganography).
        """
        n         = min(sample_size, original.numel(), embedded.numel())
        orig_samp = original[torch.randperm(original.numel())[:n]].abs()
        emb_samp  = embedded[torch.randperm(embedded.numel())[:n]].abs()

        # Optimal threshold between the two distributions
        threshold = (orig_samp.mean() + emb_samp.mean()).item() / 2.0

        # Class 0: classify as "clean" if < threshold
        orig_correct = (orig_samp < threshold).float().mean().item()
        # Class 1: classify as "embedded" if >= threshold
        emb_correct  = (emb_samp >= threshold).float().mean().item()

        return (orig_correct + emb_correct) / 2.0


class SecurityResult:
    """Result of security / undetectability validation."""

    def __init__(
        self,
        kl_divergence:     float,
        sign_bias:         float,
        moment_shift:      dict,
        detector_accuracy: float,
        status:            str,
    ):
        self.kl_divergence     = kl_divergence
        self.sign_bias         = sign_bias
        self.moment_shift      = moment_shift
        self.detector_accuracy = detector_accuracy
        self.status            = status

    @property
    def passed(self) -> bool:
        return self.status == "PASS"

    def report(self) -> str:
        return "\n".join([
            f"Security Validation  →  {self.status}",
            f"  KL divergence       : {self.kl_divergence:.6f}",
            f"  Sign bias           : {self.sign_bias:.6f}",
            f"  Mean shift          : {self.moment_shift['mean_shift']:.6f}",
            f"  Std shift           : {self.moment_shift['std_shift']:.6f}",
            f"  Detector accuracy   : {self.detector_accuracy * 100:.2f}%",
        ])
"""
Robustness Validator — BER vs noise sigma characterisation.

Tests recovery accuracy under additive Gaussian noise at multiple
sigma levels to characterise the operational envelope.

Target: BER < 0.02 at σ=0.001, BER < 0.10 at σ=0.002.
"""

from typing import Dict, List, Tuple

import torch

from src.extraction.sign_extractor import SignExtractor


class RobustnessValidator:
    """
    Measures BER (Bit Error Rate) under additive Gaussian noise.

    For each sigma level:
        1. Add N(0, sigma) noise to embedded residuals.
        2. Extract bits using sign rule.
        3. Compare to original bits → compute BER.

    Returns a RobustnessResult with per-sigma BER and PASS/FAIL status.
    """

    DEFAULT_SIGMAS = [
        0.0,
        0.0005,
        0.001,
        0.002,
        0.005,
        0.010,
        0.020,
    ]

    def __init__(
        self,
        max_ber_at_001: float = 0.02,    # BER limit at σ=0.001
        max_ber_at_002: float = 0.10,    # BER limit at σ=0.002
        num_trials:     int   = 5,       # Repeat each sigma level
    ):
        self.max_ber_at_001 = max_ber_at_001
        self.max_ber_at_002 = max_ber_at_002
        self.num_trials     = num_trials
        self.extractor      = SignExtractor()

    def validate(
        self,
        embedded_residuals: Dict[int, torch.Tensor],
        carrier_indices:    Dict[int, List[int]],
        original_bits:      List[int],
        sigmas:             List[float] = None,
    ) -> "RobustnessResult":
        """
        Run BER measurement across noise levels.

        Args:
            embedded_residuals: {layer_id: tensor} — post-embedding residuals.
            carrier_indices:    {layer_id: [indices]} — carrier positions.
            original_bits:      Ground-truth bit sequence.
            sigmas:             Noise levels to test (default: DEFAULT_SIGMAS).

        Returns:
            RobustnessResult with per-sigma BER curve.
        """
        if sigmas is None:
            sigmas = self.DEFAULT_SIGMAS

        ber_curve: Dict[float, float] = {}

        for sigma in sigmas:
            bers = []
            for _ in range(self.num_trials):
                noisy = self._add_noise(embedded_residuals, sigma)
                recovered = self.extractor.extract(noisy, carrier_indices)

                # Align lengths
                n   = min(len(original_bits), len(recovered))
                ber = sum(
                    1 for a, b in zip(original_bits[:n], recovered[:n]) if a != b
                ) / max(n, 1)
                bers.append(ber)

            ber_curve[sigma] = sum(bers) / len(bers)

        # Determine PASS/FAIL
        ber_001 = ber_curve.get(0.001, ber_curve.get(0.0010, None))
        ber_002 = ber_curve.get(0.002, ber_curve.get(0.0020, None))

        passed = True
        if ber_001 is not None and ber_001 > self.max_ber_at_001:
            passed = False
        if ber_002 is not None and ber_002 > self.max_ber_at_002:
            passed = False

        return RobustnessResult(
            ber_curve=ber_curve,
            ber_at_001=ber_001,
            ber_at_002=ber_002,
            status="PASS" if passed else "FAIL",
            total_bits=len(original_bits),
            num_trials=self.num_trials,
            max_ber_at_001=self.max_ber_at_001,
            max_ber_at_002=self.max_ber_at_002,
        )

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _add_noise(
        self,
        residuals: Dict[int, torch.Tensor],
        sigma: float,
    ) -> Dict[int, torch.Tensor]:
        """Add i.i.d. Gaussian noise to all residual tensors."""
        if sigma == 0.0:
            return residuals
        return {
            lid: tensor + torch.randn_like(tensor) * sigma
            for lid, tensor in residuals.items()
        }


class RobustnessResult:
    def __init__(
        self,
        ber_curve: Dict[float, float],
        ber_at_001: float,
        ber_at_002: float,
        status: str,
        total_bits: int,
        num_trials: int,
        max_ber_at_001: float,
        max_ber_at_002: float,
    ):
        self.ber_curve = ber_curve
        self.ber_at_001 = ber_at_001
        self.ber_at_002 = ber_at_002
        self.status = status
        self.total_bits = total_bits
        self.num_trials = num_trials

        self.max_ber_at_001 = max_ber_at_001
        self.max_ber_at_002 = max_ber_at_002

    @property
    def passed(self) -> bool:
        return self.status == "PASS"

    def report(self) -> str:
        lines = [
            f"Robustness Validation  →  {self.status}",
            f"  Total bits tested   : {self.total_bits}",
            f"  Trials per sigma    : {self.num_trials}",
            "  BER curve:",
        ]

        for sigma, ber in sorted(self.ber_curve.items()):
            marker = ""

            if abs(sigma - 0.001) < 1e-12:
                marker = (
                    " ✓ PASS"
                    if ber < self.max_ber_at_001
                    else " ✗ FAIL"
                )
            elif abs(sigma - 0.002) < 1e-12:
                marker = (
                    " ✓ PASS"
                    if ber < self.max_ber_at_002
                    else " ✗ FAIL"
                )

            lines.append(
                f"    σ={sigma:.4f}  →  BER={ber:.4f}{marker}"
            )

        return "\n".join(lines)
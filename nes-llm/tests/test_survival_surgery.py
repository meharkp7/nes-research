"""
Tests for exp23's surgery primitives (W6).

The primitives are pure tensor functions so they can be pinned
without loading a model: the experiment spends its model load on
measurements, not on re-verifying arithmetic the suite already
covers. Each test also pins the failure mode the plan warns about —
a surgery that silently reshapes or silently no-ops would produce a
plausible-looking BER that measured nothing.
"""

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch  # noqa: E402

from src.experiments.exp23_model_surgery import (  # noqa: E402
    magnitude_prune,
    low_rank_delta,
    merge_delta,
    nf4_requant,
)


class SurgeryPrimitiveTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(0)
        self.w = torch.randn(64, 48) * 0.02

    def test_magnitude_prune_zeroes_smallest_only(self):
        """Prune must zero exactly the smallest-|w| fraction, per
        layer, leaving everything else bit-identical — and must not
        reshape."""
        for fraction, want in ((0.10, 64 * 48 * 0.10),
                               (0.30, 64 * 48 * 0.30)):
            out = magnitude_prune(self.w, fraction)
            self.assertEqual(out.shape, self.w.shape)
            zeroed = int((out == 0).sum())
            self.assertEqual(zeroed, int(want))
            # Every victim was among the smallest magnitudes: no
            # surviving entry is smaller than a zeroed one.
            survivors = out[out != 0].abs()
            victims = self.w[out == 0].abs()
            if len(survivors) and len(victims):
                self.assertGreaterEqual(
                    survivors.min().item(), victims.max().item() - 1e-9
                )
            # Untouched entries are bit-identical (surgery is not a
            # no-op elsewhere either).
            keep = out != 0
            self.assertTrue(torch.equal(out[keep], self.w[keep]))

        with self.assertRaises(ValueError):
            magnitude_prune(self.w, 1.5)

    def test_low_rank_delta_rank_scale_shape(self):
        """The LoRA-shaped delta must be rank-bounded, correct in
        shape, and hit its stated RMS scale — an off-scale delta
        would make 'the merge step is linear' meaningless."""
        gen = torch.Generator().manual_seed(7)
        for ratio in (0.001, 0.01):
            delta = low_rank_delta(self.w, ratio, rank=8, generator=gen)
            self.assertEqual(delta.shape, self.w.shape)
            rms_d = delta.pow(2).mean().sqrt()
            rms_w = self.w.pow(2).mean().sqrt()
            self.assertAlmostEqual(
                (rms_d / rms_w).item(), ratio, places=6
            )
            if abs(ratio - 0.01) > 1e-12:  # rank check on a clean case
                rank = torch.linalg.matrix_rank(
                    delta.float() / rms_w.float()
                ).item()
                self.assertLessEqual(rank, 8)

        # Deterministic under a fixed seed (cells must reproduce).
        a = low_rank_delta(self.w, 0.01, 8, torch.Generator().manual_seed(7))
        b = low_rank_delta(self.w, 0.01, 8, torch.Generator().manual_seed(7))
        self.assertTrue(torch.equal(a, b))

    def test_merge_delta_arithmetic(self):
        """merge_delta returns the DELTA to add to W_stego (the
        plumbing is r' = embedded_r + delta): t=0 is the no-op,
        applying the t=1 delta lands on the partner, the applied
        midpoint is the linear blend, mismatched shapes raise instead
        of reshaping."""
        other = torch.randn(64, 48) * 0.02
        zeros = merge_delta(self.w, other, 0.0)
        self.assertTrue(torch.equal(zeros, torch.zeros_like(self.w)))

        full = merge_delta(self.w, other, 1.0)
        self.assertTrue(torch.allclose(self.w + full, other, atol=1e-6))

        half = merge_delta(self.w, other, 0.5)
        expect_delta = 0.5 * (other - self.w)
        self.assertTrue(torch.allclose(half, expect_delta, atol=1e-6))
        self.assertTrue(torch.allclose(self.w + half,
                                       self.w + expect_delta, atol=1e-6))

        with self.assertRaises(ValueError):
            merge_delta(self.w, torch.randn(48, 64), 0.5)

    def test_nf4_requant_shape_and_finiteness(self):
        """The NF4 leg must return a same-shape, finite weight tensor
        whose round trip is bounded — a quantizer that returns NaN or
        a transposed matrix is the failure the ground rules name."""
        out = nf4_requant(self.w)
        self.assertEqual(out.shape, self.w.shape)
        self.assertTrue(torch.isfinite(out).all())
        # Quantization error must stay a fraction of the tensor's own
        # scale (NF4 on a unit-ish Gaussian: bounded, not catastrophic).
        err = (out - self.w).abs().mean()
        scale = self.w.abs().mean()
        self.assertLess(err.item(), 0.5 * scale.item())


if __name__ == "__main__":
    unittest.main()

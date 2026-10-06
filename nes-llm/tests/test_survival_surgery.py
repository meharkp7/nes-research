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
    fp16_merge_diff,
    lm_blocks,
    not_run_findings,
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


class FineTuneLegTests(unittest.TestCase):
    """W6.2's leg machinery (added 2026-10-06): the pieces that can
    be pinned without a model — token chunking, the delta contract,
    and not_run's outcome-driven findings."""

    def test_lm_blocks_are_non_overlapping_and_gap_free(self):
        ids = list(range(10))
        blocks = lm_blocks(ids, 4)
        # tail shorter than a block is dropped, never padded silently
        self.assertEqual(blocks, [[0, 1, 2, 3], [4, 5, 6, 7]])
        self.assertEqual([i for b in blocks for i in b], list(range(8)))
        with self.assertRaises(ValueError):
            lm_blocks(ids, 0)

    def test_fp16_merge_diff_is_exact_and_noop_on_no_change(self):
        orig = {0: (torch.randn(32, 16) * 0.02).half()}
        # no-op training: identical tensors -> exact zeros
        same = {0: orig[0].clone()}
        self.assertTrue((fp16_merge_diff(orig, same)[0] == 0).all())
        # a real change: fp32 subtraction of the two fp16 views must
        # match the exact (float64) difference to fp32 precision
        trained = {0: (orig[0] + 0.01 * torch.randn_like(orig[0])).half()}
        delta = fp16_merge_diff(orig, trained)[0]
        ref = trained[0].double() - orig[0].double()
        self.assertTrue(
            torch.allclose(delta.double(), ref, rtol=1e-6, atol=1e-8)
        )
        # shape mismatch is refused, never broadcast or reshaped
        with self.assertRaises(ValueError):
            fp16_merge_diff(orig, {0: torch.randn(16, 32).half()})
        with self.assertRaises(KeyError):
            fp16_merge_diff(orig, {1: trained[0]})

    def test_not_run_findings_carry_leg_outcomes_not_policy(self):
        # no leg errors: findings only from missing deps — in this
        # environment peft/trl/gptqmodel/autoawq/bitsandbytes are all
        # present, so nothing is recorded (the legs ran or ran to a
        # real exception captured elsewhere)
        self.assertEqual(not_run_findings({}), [])
        # a failed leg records its REAL exception, verbatim
        errs = {
            "W6.2": "RuntimeError: boom",
            "W6.3-GPTQ": "TypeError: nope",
            "W6.3-AWQ": "OSError: awq broke",
        }
        got = {f["item"]: f["reason"] for f in not_run_findings(errs)}
        self.assertEqual(
            got["W6.2 — 1-10k fine-tuning steps"], "RuntimeError: boom"
        )
        self.assertIn("TypeError: nope", got["W6.3 — GPTQ leg (NF4 -> GPTQ -> back)"])
        self.assertIn("OSError: awq broke", got["W6.3 — AWQ leg"])
        # the AWQ finding still carries the live transformers probe
        self.assertIn("AwqQuantizer", got["W6.3 — AWQ leg"])
        for f in not_run_findings(errs):
            self.assertIn("recorded, not patched", f["status"])


if __name__ == "__main__":
    unittest.main()

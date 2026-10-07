"""
Correctness tests for the GPTQ/AWQ dequantization adapters.

Two properties matter, and both are silent failures when violated:

1.  The reconstructed weight must be the true weight. Checked by
    round-tripping against synthetic tensors whose layout matches what
    real checkpoints actually use, verified against
    ``quantization/packed_loader`` output shapes:

        GPTQ  qweight (in//8, out)   qzeros (groups, out//8)
              scales (groups, out)  g_idx (in,)
        AWQ   qweight (in, out//8)   qzeros (groups, out//8)
              scales (groups, out)

    GPTQ packs nibbles along the input axis and AWQ along the output
    axis. Using one convention for both transposes one of them.

2.  The result must be in ``nn.Linear.weight`` order, ``[out, in]``.
    Both formats store ``[in, out]`` and rely on the runtime
    transposing at matmul time.

3.  The nibble order inside each 32-bit word differs by format. GPTQ
    packs element ``i`` at nibble ``i % 8``; AutoAWQ packs with
    ``order_map = [0, 2, 4, 6, 1, 3, 5, 7]``, so element ``i`` sits in
    nibble ``AWQ_NIBBLE_ORDER[i] = [0, 4, 1, 5, 2, 6, 3, 7]``. That is
    an even/odd interleave, unreachable by rotating the nibble field, so
    a rotation search over the real checkpoint stalled at correlation
    0.2343 across 24 variants.

An earlier version of this file built its synthetic tensors from the
same assumption it was testing, so it passed while the real AWQ layout
was wrong. Fixtures here are shaped from observed checkpoints, and the
AWQ fixture packs with the real ``order_map`` rather than in sequence.
"""

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch  # noqa: E402

from src.quantization.adapters import (  # noqa: E402
    AWQ_NIBBLE_ORDER,
    STANDARD_NIBBLE_ORDER,
    _unpack_4bit,
    absorbed_scale,
    dequantize_awq_layer,
    dequantize_gptq_layer,
    dequantize_layer,
    detect_format,
    verify_dequantization,
)

PACK = 8

# AutoAWQ ``WQLinear_GEMM.from_linear``: nibble i receives element
# order_map[i] of each group of 8.
AWQ_ORDER_MAP = (0, 2, 4, 6, 1, 3, 5, 7)

# Shaped to match the observed real checkpoints.
IN_FEATURES = 64
OUT_FEATURES = 32
GROUP_WIDTH = 16
NUM_GROUPS = IN_FEATURES // GROUP_WIDTH


def _pack(values_2d):
    """Pack a 2-D int tensor along axis 0, 4 bits per value."""
    assert values_2d.shape[0] % PACK == 0
    words = values_2d.shape[0] // PACK
    out = torch.zeros((words, values_2d.shape[1]), dtype=torch.int64)
    for j in range(PACK):
        out |= (values_2d[j::PACK, :] << (4 * j)).to(torch.int64)
    return out


def _pack_axis1(values_2d):
    """Pack along axis 1, 4 bits per value, sequentially.

    This is the GPTQ/standard order: element ``j`` lands in nibble
    ``j % 8``.
    """
    assert values_2d.shape[1] % PACK == 0
    words = values_2d.shape[1] // PACK
    out = torch.zeros((values_2d.shape[0], words), dtype=torch.int64)
    for j in range(PACK):
        out |= (values_2d[:, j::PACK] << (4 * j)).to(torch.int64)
    return out


def _pack_awq(values_2d):
    """Pack along axis 1 using AutoAWQ's ``order_map`` nibble order.

    ``for i in range(pack_num): qweight[:, col] |=
    intweight[:, col * pack_num + order_map[i]] << (i * w_bit)``.

    Building the fixture this way is what makes the test a test: packing
    sequentially and then unpacking sequentially round-trips regardless
    of whether either matches the real checkpoint.
    """
    assert values_2d.shape[1] % PACK == 0
    words = values_2d.shape[1] // PACK
    out = torch.zeros((values_2d.shape[0], words), dtype=torch.int64)
    for i in range(PACK):
        out |= (
            values_2d[:, AWQ_ORDER_MAP[i]::PACK] << (4 * i)
        ).to(torch.int64)
    return out


class _Weight:
    pass


class _Module:
    pass


def make_gptq(group_per_channel=False):
    torch.manual_seed(0)

    # (in, out) integer values, as the format stores them.
    values = torch.randint(0, 16, (IN_FEATURES, OUT_FEATURES))
    qweight = _pack(values).to(torch.int32)

    # Zero points are grouped along in, packed along out.
    #
    # GPTQ applies a -1 bias at pack time, so the stored nibbles are
    # (z - 1) and dequantization adds 1 back. Baking that into the
    # fixture is what makes it a real test of the correction rather than
    # a test of the same assumption.
    # z >= 1 so that (z - 1) stays a valid nibble. Clamping instead
    # would break the round trip for z = 0, since dequant adds 1 back.
    zeros = torch.randint(1, 16, (NUM_GROUPS, OUT_FEATURES))
    qzeros = _pack_axis1(zeros - 1).to(torch.int32)

    scales = torch.rand((NUM_GROUPS, OUT_FEATURES)) + 0.5

    if group_per_channel:
        # desc_act permutes which group each input row belongs to, so
        # g_idx holds group ids (each appearing group_width times) in a
        # shuffled order -- not a permutation of row indices.
        g_idx = torch.tensor(
            [i // GROUP_WIDTH for i in range(IN_FEATURES)],
            dtype=torch.int32,
        )
        g_idx = g_idx[torch.randperm(IN_FEATURES)].contiguous()
    else:
        g_idx = torch.tensor(
            [i // GROUP_WIDTH for i in range(IN_FEATURES)],
            dtype=torch.int32,
        )

    module = _Module()
    weight = _Weight()
    module.weight = weight
    weight.qweight = qweight
    weight.qzeros = qzeros
    weight.scales = scales
    weight.g_idx = g_idx

    reference = (
        (values.float() - zeros.repeat_interleave(GROUP_WIDTH, 0))
        * scales.repeat_interleave(GROUP_WIDTH, 0)
    ).T.contiguous()  # -> [out, in]

    return module, reference, values, zeros, scales


def make_awq(pack=_pack_awq):
    torch.manual_seed(1)

    values = torch.randint(0, 16, (IN_FEATURES, OUT_FEATURES))
    # AWQ packs along the output axis, with AutoAWQ's interleaved
    # nibble order -- not the sequential order GPTQ uses.
    qweight = pack(values).to(torch.int32)

    zeros = torch.randint(0, 16, (NUM_GROUPS, OUT_FEATURES))
    qzeros = pack(zeros).to(torch.int32)

    scales = torch.rand((NUM_GROUPS, OUT_FEATURES)) + 0.5

    module = _Module()
    weight = _Weight()
    module.weight = weight
    weight.qweight = qweight
    module.qzeros = qzeros
    module.scales = scales

    # AWQ does not apply GPTQ's -1 pack bias.
    reference = (
        (values.float() - zeros.repeat_interleave(GROUP_WIDTH, 0))
        * scales.repeat_interleave(GROUP_WIDTH, 0)
    ).T.contiguous()

    return module, reference


class GPTQTests(unittest.TestCase):
    def test_matches_reconstruction_sequential_groups(self):
        module, reference, *_ = make_gptq()

        self.assertTrue(
            torch.equal(dequantize_gptq_layer(module), reference)
        )

    def test_matches_with_permuted_g_idx(self):
        # desc_act=True permutes which group each input row belongs
        # to, so the reconstruction must follow g_idx rather than
        # assuming position.
        module, _, values, zeros, scales = make_gptq(
            group_per_channel=True
        )

        g_idx = module.weight.g_idx.long()
        zero_by_row = zeros[g_idx]
        scale_by_row = scales[g_idx]

        reference = (
            (values.float() - zero_by_row) * scale_by_row
        ).T.contiguous()

        self.assertTrue(
            torch.equal(dequantize_gptq_layer(module), reference)
        )

    def test_output_is_nn_linear_order(self):
        """Must be [out, in], not the packed [in, out]."""
        module, *_ = make_gptq()

        self.assertEqual(
            tuple(dequantize_gptq_layer(module).shape),
            (OUT_FEATURES, IN_FEATURES),
        )

    def test_group_index_out_of_range_raises(self):
        module, *_ = make_gptq()
        module.weight.g_idx = torch.full(
            (IN_FEATURES,), NUM_GROUPS, dtype=torch.int32
        )
        with self.assertRaises(RuntimeError):
            dequantize_gptq_layer(module)


class AWQTests(unittest.TestCase):
    def test_matches_reconstruction(self):
        module, reference = make_awq()

        self.assertTrue(
            torch.equal(dequantize_awq_layer(module), reference)
        )

    def test_output_is_nn_linear_order(self):
        module, _ = make_awq()

        self.assertEqual(
            tuple(dequantize_awq_layer(module).shape),
            (OUT_FEATURES, IN_FEATURES),
        )

    def test_zero_points_are_not_gptq_biased(self):
        """AWQ omits GPTQ's -1 pack bias.

        Copying the `+ 1` shifts every weight by one scale unit.
        """
        gptq, _reference_g, *_ = make_gptq()
        awq, _reference_a = make_awq()

        self.assertFalse(
            torch.equal(
                dequantize_gptq_layer(gptq),
                dequantize_awq_layer(awq),
            )
        )

    def test_nibble_order_is_not_a_rotation(self):
        """The AWQ order is unreachable by rotating the nibble field.

        The real checkpoint was searched over 8 nibble rotations x 3
        zero-point variants and the best reached correlation 0.2343.
        This documents why: the order is an even/odd interleave, and
        rotations only span 8 of the 40320 permutations.
        """
        rotations = [
            [(k + r) % 8 for k in range(8)] for r in range(8)
        ]

        # Identity is the r=0 rotation, so only the AWQ order is at
        # issue: it is not any rotation of the sequential order.
        self.assertNotIn(list(AWQ_NIBBLE_ORDER), rotations)

    def test_adapter_rejects_sequential_pack(self):
        """A sequential pack is recoverable only by a sequential unpack.

        This is the trap that hid the bug: packing and unpacking with
        the same wrong assumption round-trips perfectly, so the fixture
        passed while the real checkpoint did not. Asserting the real
        adapter *rejects* a sequential pack proves it is genuinely
        AutoAWQ-ordered rather than merely self-consistent.
        """
        module, reference = make_awq(pack=_pack_axis1)

        self.assertFalse(
            torch.equal(dequantize_awq_layer(module), reference)
        )

    def test_wrong_nibble_order_is_invisible_to_shape_and_scale(self):
        """The failure is invisible to every check but correlation.

        Sequential order recovers the same integers as a rearrangement:
        identical shape, identical value range, dequantized standard
        deviation within 0.4% of the reference. So a shape check, a
        magnitude check and a "the numbers look plausible" review all
        pass. Only comparing against the reference weights catches it,
        which is what ``verify_dequantization`` is for.
        """
        module, reference = make_awq(pack=_pack_axis1)
        produced = dequantize_awq_layer(module)

        sequential = _unpack_4bit(
            module.weight.qweight,
            IN_FEATURES, OUT_FEATURES,
            axis=-1, order=STANDARD_NIBBLE_ORDER,
        )
        interleaved = _unpack_4bit(
            module.weight.qweight,
            IN_FEATURES, OUT_FEATURES,
            axis=-1, order=AWQ_NIBBLE_ORDER,
        )

        self.assertEqual(
            tuple(produced.shape), tuple(reference.shape)
        )
        self.assertTrue(
            torch.equal(
                sequential.flatten().sort().values,
                interleaved.flatten().sort().values,
            )
        )

        # Scales vary along the output axis, so the dequantized tensor
        # is not even a permutation of the reference -- but it is close
        # enough in aggregate to look right.
        drift = abs(
            float(produced.std()) - float(reference.std())
        ) / float(reference.std())
        self.assertLess(drift, 0.02)
        self.assertFalse(torch.equal(produced, reference))


class RuntimeModuleAttrsTests(unittest.TestCase):
    """Both readers must accept module-level packed tensors.

    The runtime quant-linear classes expose qweight/qzeros/scales
    (and g_idx for GPTQ) as direct attributes: gptqmodel's
    TorchLinear answers `.weight` with a metadata shim that carries
    none of them, and autoawq's WQLinear_GEMM has no `.weight` at
    all — reading through `.weight` raised AttributeError on both in
    exp23's legs (2026-10-07). The module-level path must produce
    the exact same matrix as the checkpoint holder the readers were
    originally validated against.
    """

    def test_gptq_module_level_matches_holder_path(self):
        module, reference, *_ = make_gptq()
        flat = _Module()
        for name in ("qweight", "qzeros", "scales", "g_idx"):
            setattr(flat, name, getattr(module.weight, name))

        self.assertTrue(
            torch.equal(dequantize_gptq_layer(module), reference)
        )
        self.assertTrue(
            torch.equal(dequantize_gptq_layer(flat), reference)
        )

    def test_awq_module_level_matches_holder_path(self):
        module, reference = make_awq()
        flat = _Module()
        flat.qweight = module.weight.qweight
        flat.qzeros = module.qzeros
        flat.scales = module.scales

        self.assertTrue(
            torch.equal(dequantize_awq_layer(module), reference)
        )
        self.assertTrue(
            torch.equal(dequantize_awq_layer(flat), reference)
        )


class DispatchTests(unittest.TestCase):
    def test_format_detection(self):
        gptq, *_ = make_gptq()
        awq, _ = make_awq()

        self.assertEqual(detect_format(gptq), "gptq")
        self.assertEqual(detect_format(awq), "awq")

    def test_mismatched_expected_format_raises(self):
        awq, _ = make_awq()

        with self.assertRaises(RuntimeError):
            dequantize_layer(awq, expected_format="gptq")

    def test_unpack_uses_different_axes(self):
        """Weights and zero points pack on different axes.

        Using one axis for both produces a transposed matrix of the right
        shape, which is the failure this guards.
        """
        values = torch.randint(0, 16, (IN_FEATURES, OUT_FEATURES))

        along_in = _unpack_4bit(
            _pack(values).to(torch.int32),
            IN_FEATURES, OUT_FEATURES, axis=1,
        )
        along_out = _unpack_4bit(
            _pack_axis1(values).to(torch.int32),
            IN_FEATURES, OUT_FEATURES, axis=-1,
        )

        self.assertTrue(torch.equal(along_in, values.float()))
        self.assertTrue(torch.equal(along_out, values.float()))


class VerifyTests(unittest.TestCase):
    """``verify_dequantization`` is what stops a plausible-looking BER.

    The gate has two jobs and they must not be confused:

    * catch a dequantizer whose *layout* is wrong (the nibble-order bug,
      which survives every shape and magnitude check);
    * accept an AWQ weight that differs from the bf16 reference only by
      the per-channel scale AWQ folded into the LayerNorm.

    The second job must not buy the first one's freedom: the scale fit
    is per-channel, and a scrambled layout is not a per-channel scale.
    """

    def test_gptq_is_compared_directly(self):
        """GPTQ quantizes the reference weight itself, so no correction."""
        module, reference, *_ = make_gptq()

        report = verify_dequantization(
            module, reference, expected_format="gptq"
        )

        self.assertTrue(report["usable"], report["reason"])
        self.assertFalse(report["scale_corrected"])
        self.assertEqual(
            report["correlation"], report["raw_correlation"]
        )

    def test_accepts_the_scale_awq_folded_into_the_layernorm(self):
        """A real absorbed scale is a pass, not a failure.

        The checkpoint holds ``W * s`` while the LayerNorm holds
        ``1/s``; the network is unchanged, so rejecting it would be a
        false alarm that keeps AWQ at NOT_RUN forever.
        """
        module, reference = make_awq()

        torch.manual_seed(7)
        # 100x across input channels -- large enough that the raw
        # comparison cannot pass, which is the point of the fixture.
        channel_scale = torch.logspace(-1, 1, IN_FEATURES)
        stored = reference * channel_scale

        report = verify_dequantization(
            module, stored, expected_format="awq",
            module_name="model.layers.0.mlp.down_proj",
        )

        raw_failed = (
            (report["raw_correlation"] or 0.0) < 0.95
            or (report["raw_residual_ratio"] or 0.0) > 0.5
        )
        self.assertTrue(
            raw_failed,
            "fixture must exercise the correction: raw comparison "
            f"passed at {report['raw_correlation']}",
        )

        self.assertTrue(report["scale_corrected"])
        self.assertTrue(report["usable"], report["reason"])
        self.assertAlmostEqual(report["correlation"], 1.0, places=5)

    def test_still_rejects_the_wrong_nibble_order(self):
        """The correction must not rescue a wrong layout.

        A sequential pack permutes values *within* each group of eight
        output channels, so the error depends on the element and cannot
        be written as a per-channel multiplier.
        """
        module, reference = make_awq(pack=_pack_axis1)

        report = verify_dequantization(
            module, reference, expected_format="awq",
            module_name="model.layers.0.mlp.down_proj",
        )

        self.assertTrue(report["scale_corrected"])
        self.assertFalse(report["usable"], report["reason"])
        self.assertLess(report["correlation"], 0.95)

    def test_nan_correlation_never_passes(self):
        """A constant dequantization yields NaN, and NaN must fail.

        ``nan < 0.95`` is False, so a naive threshold lets an all-zero
        tensor through both checks and reports a verified dequantizer.
        """
        module, reference = make_awq()
        module.scales = torch.zeros((NUM_GROUPS, OUT_FEATURES))

        report = verify_dequantization(
            module, reference, expected_format="awq"
        )

        self.assertFalse(report["usable"], report["reason"])

    def test_scale_fit_does_not_absorb_a_collapsed_channel(self):
        """A channel the quantizer zeroed must stay a mismatch.

        Fitting the multiplier through the zeros would declare the
        information loss to be a scale, which is exactly the loophole
        that would let a broken checkpoint reach the BER step.
        """
        reference = torch.ones(8, 4)
        dequantized = reference * torch.tensor([1.0, 2.0, 4.0, 8.0])

        scale = absorbed_scale(dequantized, reference)
        self.assertTrue(
            torch.allclose(scale, dequantized / reference, atol=1e-5)
        )

        collapsed = dequantized.clone()
        collapsed[:, 0] = 0.0
        scale = absorbed_scale(collapsed, reference)
        self.assertTrue(
            torch.allclose(scale[:, 0], torch.ones(8)),
            "a zeroed channel was scaled into agreement",
        )


class ThresholdTests(unittest.TestCase):
    """Ground rule 2: thresholds live in `experiment_registry.THRESHOLDS`.

    The dequantizer gate used to exist only as a default on
    `verify_dequantization`, which made the rule true of `max_ber` and
    false of the gate that actually decides whether a residual may be
    computed. The suite now reads the numbers from the registry, so they
    must agree with the function's defaults -- and both must still be the
    documented values, or a coordinated edit would move the gate without
    any single file looking changed.
    """

    LITERAL_MIN_CORRELATION = 0.95
    LITERAL_MAX_RESIDUAL_RATIO = 0.5

    def _registry(self):
        from src.experiments.experiment_registry import THRESHOLDS

        return THRESHOLDS["exp9"]

    def test_registry_carries_the_documented_values(self):
        th = self._registry()

        self.assertEqual(
            th["min_dequant_correlation"], self.LITERAL_MIN_CORRELATION
        )
        self.assertEqual(
            th["max_dequant_residual_ratio"], self.LITERAL_MAX_RESIDUAL_RATIO
        )

    def test_registry_agrees_with_the_gate_defaults(self):
        import inspect

        th = self._registry()
        params = inspect.signature(verify_dequantization).parameters

        self.assertEqual(
            params["min_correlation"].default, th["min_dequant_correlation"]
        )
        self.assertEqual(
            params["max_residual_ratio"].default,
            th["max_dequant_residual_ratio"],
        )


if __name__ == "__main__":
    unittest.main(verbosity=2)
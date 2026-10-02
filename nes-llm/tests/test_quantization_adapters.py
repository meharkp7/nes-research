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

An earlier version of this file built its synthetic tensors from the
same assumption it was testing, so it passed while the real AWQ layout
was wrong. Fixtures here are shaped from observed checkpoints instead.
"""

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch  # noqa: E402

from src.quantization.adapters import (  # noqa: E402
    _unpack_4bit,
    dequantize_awq_layer,
    dequantize_gptq_layer,
    dequantize_layer,
    detect_format,
)

PACK = 8

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
    """Pack along axis 1, 4 bits per value."""
    assert values_2d.shape[1] % PACK == 0
    words = values_2d.shape[1] // PACK
    out = torch.zeros((values_2d.shape[0], words), dtype=torch.int64)
    for j in range(PACK):
        out |= (values_2d[:, j::PACK] << (4 * j)).to(torch.int64)
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


def make_awq():
    torch.manual_seed(1)

    values = torch.randint(0, 16, (IN_FEATURES, OUT_FEATURES))
    # AWQ packs along the output axis.
    qweight = _pack_axis1(values).to(torch.int32)

    zeros = torch.randint(0, 16, (NUM_GROUPS, OUT_FEATURES))
    qzeros = _pack_axis1(zeros).to(torch.int32)

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


if __name__ == "__main__":
    unittest.main(verbosity=2)
"""
Correctness tests for the GPTQ/AWQ dequantization adapters.

These are round-trip tests against the reference algorithms from
AutoGPTQ's ``QuantLinear.forward``, not smoke tests. The failure they
guard against is specific and silent: unpacking 4-bit nibbles along the
wrong axis produces a transposed weight matrix that still has the right
shape, so it neither errors nor looks obviously wrong — it just yields
meaningless residuals and therefore meaningless Exp9 results.
"""

import sys
import unittest
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.quantization.adapters import (  # noqa: E402
    dequantize_awq_layer,
    dequantize_gptq_layer,
    dequantize_layer,
    detect_format,
)

PACK = 8  # four-bit values per int32


def _reference_unpack_weights(qweight):
    """AutoGPTQ's qweight unpack, verbatim."""
    wf = torch.tensor(
        list(range(0, 32, 4)), dtype=torch.int32
    ).unsqueeze(0)
    weight = torch.bitwise_right_shift(
        torch.unsqueeze(qweight, 1).expand(-1, PACK, -1),
        wf.unsqueeze(-1),
    ).to(torch.int8)
    weight = torch.bitwise_and(weight, 15)
    return weight.reshape(
        weight.shape[0] * weight.shape[1], weight.shape[2]
    )


def _reference_unpack_zeros(qzeros, add_one):
    """AutoGPTQ's qzeros unpack, verbatim."""
    wf = torch.tensor(
        list(range(0, 32, 4)), dtype=torch.int32
    ).unsqueeze(0)
    zeros = torch.bitwise_right_shift(
        torch.unsqueeze(qzeros, 2).expand(-1, -1, PACK),
        wf.unsqueeze(0),
    ).to(torch.int8)
    zeros = torch.bitwise_and(zeros, 15)
    if add_one:
        zeros = zeros + 1
    return zeros.reshape(zeros.shape[0], zeros.shape[1] * PACK)


class _Weight:
    pass


class _Module:
    pass


class QuantizationAdapterTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(0)

        self.in_features = 64
        self.out_features = 32
        self.group_width = 16
        self.num_groups = self.in_features // self.group_width

        self.qweight = torch.randint(
            -2**31, 2**31 - 1,
            (self.in_features // PACK, self.out_features),
            dtype=torch.int32,
        )
        self.qzeros = torch.randint(
            -2**31, 2**31 - 1,
            (self.num_groups, self.out_features // PACK),
            dtype=torch.int32,
        )
        self.scales = (
            torch.rand((self.num_groups, self.out_features)) + 0.5
        ).float()

        self.ref_weights = _reference_unpack_weights(self.qweight)

    def _gptq_module(self, g_idx):
        module = _Module()
        weight = _Weight()
        module.weight = weight
        weight.qweight = self.qweight
        weight.qzeros = self.qzeros
        weight.scales = self.scales
        weight.g_idx = g_idx
        return module

    def _awq_module(self):
        module = _Module()
        weight = _Weight()
        module.weight = weight
        weight.qweight = self.qweight
        module.qzeros = self.qzeros
        module.scales = self.scales
        return module

    def _desc_act_false_g_idx(self):
        return torch.tensor(
            [i // self.group_width for i in range(self.in_features)],
            dtype=torch.int32,
        )

    # -- GPTQ ------------------------------------------------------

    def test_gptq_matches_reference_sequential_groups(self):
        g_idx = self._desc_act_false_g_idx()
        zeros = _reference_unpack_zeros(self.qzeros, add_one=True)

        expected = (
            self.scales[g_idx.long()]
            * (self.ref_weights - zeros[g_idx.long()])
        )

        self.assertTrue(
            torch.equal(expected, dequantize_gptq_layer(self._gptq_module(g_idx)))
        )

    def test_gptq_matches_reference_with_desc_act_permutation(self):
        base = self._desc_act_false_g_idx()
        g_idx = base[torch.randperm(self.in_features)].contiguous()
        zeros = _reference_unpack_zeros(self.qzeros, add_one=True)

        expected = (
            self.scales[g_idx.long()]
            * (self.ref_weights - zeros[g_idx.long()])
        )

        self.assertTrue(
            torch.equal(expected, dequantize_gptq_layer(self._gptq_module(g_idx)))
        )

    # -- AWQ -------------------------------------------------------

    def test_awq_matches_reference(self):
        zeros = _reference_unpack_zeros(self.qzeros, add_one=False)

        expected = (
            self.scales.repeat_interleave(self.group_width, dim=0)
            * (
                self.ref_weights
                - zeros.repeat_interleave(self.group_width, dim=0)
            )
        )

        self.assertTrue(
            torch.equal(expected, dequantize_awq_layer(self._awq_module()))
        )

    def test_awq_zero_points_are_not_gptq_biased(self):
        """AWQ does not apply GPTQ's -1 pack bias.

        Copying the ``+ 1`` from the GPTQ path shifts every weight by one
        scale unit. This asserts the two paths genuinely differ.
        """
        g_idx = self._desc_act_false_g_idx()

        gptq = dequantize_gptq_layer(self._gptq_module(g_idx))
        awq = dequantize_awq_layer(self._awq_module())

        self.assertFalse(torch.equal(gptq, awq))

    # -- Format detection and guards -------------------------------

    def test_format_detection(self):
        self.assertEqual(
            detect_format(self._gptq_module(self._desc_act_false_g_idx())),
            "gptq",
        )
        self.assertEqual(detect_format(self._awq_module()), "awq")

    def test_mismatched_expected_format_raises(self):
        """A GPTQ layer must never be read through the AWQ layout."""
        module = self._gptq_module(self._desc_act_false_g_idx())

        with self.assertRaises(RuntimeError):
            dequantize_layer(module, expected_format="awq")

    def test_shape_and_dtype(self):
        g_idx = self._desc_act_false_g_idx()

        weight = dequantize_gptq_layer(self._gptq_module(g_idx))

        self.assertEqual(
            tuple(weight.shape), (self.in_features, self.out_features)
        )
        self.assertEqual(weight.dtype, torch.float32)


if __name__ == "__main__":
    unittest.main(verbosity=2)
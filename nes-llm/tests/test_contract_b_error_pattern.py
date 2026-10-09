import unittest

from scripts.contract_b_nf4_b14 import (
    DEFAULT_TENSOR_KEY,
    TEST_KEY_HEX,
    code_at,
    embed,
    envelope,
    payload_bytes,
    positions,
    set_code,
    to_bits,
)
from scripts.contract_b_error_pattern import analyze_packed_tensors, contiguous_runs


class ContractBErrorPatternTests(unittest.TestCase):
    def setUp(self):
        self.key = bytes.fromhex(TEST_KEY_HEX)
        self.payload = payload_bytes()
        packed = bytes((i * 37 + 11) & 255 for i in range(8192))
        self.reference, self.carriers, _, checks = embed(
            packed, to_bits(envelope(self.payload)), self.key, DEFAULT_TENSOR_KEY
        )
        self.assertTrue(all(checks.values()))

    def test_pristine_artifact_has_no_errors(self):
        result = analyze_packed_tensors(
            self.reference, self.reference, self.key, DEFAULT_TENSOR_KEY
        )
        self.assertEqual(result["status"], "ANALYZED")
        self.assertTrue(result["exact_payload_recovery"])
        self.assertTrue(result["payload_checksum_valid"])
        self.assertEqual(result["bit_errors"], 0)
        self.assertEqual(result["ber"], 0.0)
        self.assertEqual(result["error_positions"], [])
        self.assertEqual(result["max_contiguous_error_run_bits"], 0)

    def test_reports_exact_positions_blocks_and_runs(self):
        transformed = bytearray(self.reference)
        carrier_positions = positions(
            self.key, len(transformed) * 2,
            (len(self.payload) + 16) * 8, DEFAULT_TENSOR_KEY
        )
        # Carrier positions 0..63 are the header; subsequent carriers begin payload.
        for index in (64, 65, 70, 90):
            carrier = carrier_positions[index]
            set_code(transformed, carrier, code_at(transformed, carrier) ^ 1)

        result = analyze_packed_tensors(
            self.reference, bytes(transformed), self.key, DEFAULT_TENSOR_KEY,
            block_bits=8
        )
        self.assertEqual(result["bit_errors"], 4)
        self.assertAlmostEqual(result["ber"], 4 / 10000)
        self.assertEqual(result["error_positions"], [0, 1, 6, 26])
        self.assertEqual(result["max_contiguous_error_run_bits"], 2)
        self.assertEqual(result["contiguous_error_runs"], [
            {"start_bit": 0, "end_bit_exclusive": 2, "length_bits": 2},
            {"start_bit": 6, "end_bit_exclusive": 7, "length_bits": 1},
            {"start_bit": 26, "end_bit_exclusive": 27, "length_bits": 1},
        ])
        self.assertEqual(result["blocks"][0]["error_count"], 3)
        self.assertFalse(result["payload_checksum_valid"])

    def test_invalid_header_does_not_invent_ber(self):
        result = analyze_packed_tensors(
            self.reference, bytes(len(self.reference)), self.key, DEFAULT_TENSOR_KEY
        )
        self.assertEqual(result["status"], "HEADER_INVALID")
        self.assertIsNone(result["ber"])
        self.assertIsNone(result["bit_errors"])

    def test_rejects_different_packed_tensor_lengths(self):
        with self.assertRaisesRegex(ValueError, "byte length"):
            analyze_packed_tensors(
                self.reference, self.reference[:-1], self.key, DEFAULT_TENSOR_KEY
            )

    def test_contiguous_runs_empty_and_clustered(self):
        self.assertEqual(contiguous_runs([]), [])
        self.assertEqual(contiguous_runs([2, 3, 4, 9]), [
            {"start_bit": 2, "end_bit_exclusive": 5, "length_bits": 3},
            {"start_bit": 9, "end_bit_exclusive": 10, "length_bits": 1},
        ])


if __name__ == "__main__":
    unittest.main()

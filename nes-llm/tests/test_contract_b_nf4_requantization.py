import unittest

from scripts.contract_b_nf4_b14 import (
    DEFAULT_TENSOR_KEY,
    TEST_KEY_HEX,
    embed,
    envelope,
    payload_bytes,
    positions,
    set_code,
    to_bits,
)
from scripts.contract_b_nf4_requantization import recoverability_diagnostic


class ContractBNF4RequantizationTests(unittest.TestCase):
    def setUp(self):
        self.key = bytes.fromhex(TEST_KEY_HEX)
        self.payload = payload_bytes()

    def make_stego(self):
        packed = bytes((i * 37 + 11) & 255 for i in range(8192))
        stego, _, _, checks = embed(
            packed, to_bits(envelope(self.payload)), self.key, DEFAULT_TENSOR_KEY
        )
        self.assertTrue(all(checks.values()))
        return stego

    def test_pristine_artifact_has_zero_ber(self):
        result = recoverability_diagnostic(
            self.make_stego(), self.payload, self.key, DEFAULT_TENSOR_KEY
        )
        self.assertTrue(result["header_valid"])
        self.assertTrue(result["checksum_valid"])
        self.assertTrue(result["exact_payload_recovery"])
        self.assertEqual(result["bit_errors"], 0)
        self.assertEqual(result["ber"], 0.0)

    def test_single_payload_carrier_corruption_is_measured(self):
        packed = bytearray(self.make_stego())
        carrier_count = (len(self.payload) + 16) * 8
        carrier_positions = positions(
            self.key, len(packed) * 2, carrier_count, DEFAULT_TENSOR_KEY
        )
        target = carrier_positions[64]
        from scripts.contract_b_nf4_b14 import code_at
        set_code(packed, target, code_at(packed, target) ^ 1)
        result = recoverability_diagnostic(
            bytes(packed), self.payload, self.key, DEFAULT_TENSOR_KEY
        )
        self.assertTrue(result["header_valid"])
        self.assertFalse(result["checksum_valid"])
        self.assertFalse(result["exact_payload_recovery"])
        self.assertEqual(result["bit_errors"], 1)
        self.assertAlmostEqual(result["ber"], 1 / 10000)

    def test_invalid_header_is_reported_without_inventing_ber(self):
        result = recoverability_diagnostic(
            bytes(8192), self.payload, self.key, DEFAULT_TENSOR_KEY
        )
        self.assertFalse(result["header_valid"])
        self.assertIsNone(result["ber"])
        self.assertIsNotNone(result["error"])


if __name__ == "__main__":
    unittest.main()

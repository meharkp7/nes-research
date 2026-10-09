import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.contract_b_nf4_b14 import (  # noqa: E402
    DEFAULT_TENSOR_KEY, TEST_KEY_HEX, embed, envelope, payload_bytes, to_bits,
)
from scripts.contract_b_robustness_matrix import (  # noqa: E402
    flip_parity, run_matrix,
)


class ContractBB14RobustnessTests(unittest.TestCase):
    def setUp(self):
        self.key = bytes.fromhex(TEST_KEY_HEX)
        self.payload = payload_bytes()
        self.packed = bytes((i * 37 + 11) & 255 for i in range(8192))
        self.stego, _, _, checks = embed(
            self.packed, to_bits(envelope(self.payload)), self.key,
            DEFAULT_TENSOR_KEY,
        )
        self.assertTrue(all(checks.values()))

    def test_parity_flip_preserves_pair_id(self):
        before = [((self.stego[i // 2] >> 4) & 15) if i % 2 == 0
                  else (self.stego[i // 2] & 15) for i in range(4)]
        changed = flip_parity(self.stego, [0, 3])
        after = [((changed[i // 2] >> 4) & 15) if i % 2 == 0
                 else (changed[i // 2] & 15) for i in range(4)]
        self.assertEqual([x // 2 for x in before], [x // 2 for x in after])
        self.assertNotEqual(before[0] & 1, after[0] & 1)
        self.assertNotEqual(before[3] & 1, after[3] & 1)
        self.assertEqual(before[1:3], after[1:3])

    def test_noncarrier_mutations_preserve_payload_and_carrier_corruption_is_rejected(self):
        report = run_matrix(
            self.stego, DEFAULT_TENSOR_KEY, self.key,
            noncarrier_counts=[1, 100],
            carrier_counts=[1, 10],
            seed=1234,
        )
        self.assertEqual(report["status"], "PASS")
        self.assertTrue(all(row["meets_expectation"] for row in report["cases"]))
        self.assertTrue(report["cases"][0]["exact_recovery"])
        self.assertTrue(report["cases"][1]["exact_recovery"])
        self.assertTrue(report["cases"][2]["exact_recovery"])
        self.assertFalse(report["cases"][3]["exact_recovery"])
        self.assertFalse(report["cases"][4]["exact_recovery"])

    def test_invalid_noncarrier_count_rejected(self):
        with self.assertRaisesRegex(ValueError, "non-carrier"):
            run_matrix(
                self.stego, DEFAULT_TENSOR_KEY, self.key,
                noncarrier_counts=[999999], carrier_counts=[1], seed=1,
            )


if __name__ == "__main__":
    unittest.main()

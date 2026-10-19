import unittest

import numpy as np

from src.experiments.nf4_artifact_codec import pack_codes
from src.steganalysis.packed_nf4_block_features import (
    packed_block_feature_rows,
    packed_to_codes,
)


class PackedNF4BlockFeatureTests(unittest.TestCase):
    def test_round_trip_and_block_count(self):
        codes = list(range(16)) * 8
        packed = pack_codes(codes)
        rows = packed_block_feature_rows(packed, len(codes), blocksize=64)
        self.assertEqual(len(rows), 2)
        self.assertEqual(rows[0]["block_value_count"], 64.0)
        self.assertEqual(rows[0]["entropy_bits"], 4.0)
        self.assertEqual(rows[0]["unique_code_count"], 16.0)

    def test_partial_final_block_is_preserved(self):
        packed = pack_codes([0, 1, 2, 3, 4, 5])
        rows = packed_block_feature_rows(packed, 6, blocksize=4)
        self.assertEqual([row["block_value_count"] for row in rows], [4.0, 2.0])

    def test_features_are_finite_and_do_not_include_block_index(self):
        packed = pack_codes([0, 0, 1, 1, 2, 2, 3, 3])
        rows = packed_block_feature_rows(packed, 8, blocksize=4)
        self.assertTrue(all(np.isfinite(v) for row in rows for v in row.values()))
        self.assertNotIn("block_index", rows[0])
        self.assertAlmostEqual(rows[0]["code_frequency_00"], 0.5)

    def test_bad_inputs_rejected(self):
        with self.assertRaises(ValueError):
            packed_to_codes(bytes([0x12]), 3)
        with self.assertRaises(ValueError):
            packed_block_feature_rows(b"", 0, blocksize=64)
        with self.assertRaises(ValueError):
            packed_block_feature_rows(bytes([0x12]), 2, blocksize=1)
        with self.assertRaises(ValueError):
            packed_to_codes(bytes([0x12]), -1)


if __name__ == "__main__":
    unittest.main()

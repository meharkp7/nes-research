import unittest

from src.experiments.nf4_artifact_codec import pack_codes, tensor_dequant, unpack_codes


class NF4ArtifactCodecTests(unittest.TestCase):
    def test_pack_unpack_round_trip(self):
        codes = [0, 15, 1, 14, 7, 8, 3, 12]
        self.assertEqual(unpack_codes(pack_codes(codes), len(codes)), codes)

    def test_unpack_can_ignore_unused_tail_nibble(self):
        packed = pack_codes([2, 9, 4, 11])
        self.assertEqual(unpack_codes(packed, 3), [2, 9, 4])

    def test_pack_rejects_odd_length_and_invalid_code(self):
        with self.assertRaises(ValueError):
            pack_codes([1, 2, 3])
        with self.assertRaises(ValueError):
            pack_codes([1, 16])

    def test_unpack_rejects_impossible_length(self):
        with self.assertRaises(ValueError):
            unpack_codes(bytes([0x12]), 3)

    def test_tensor_dequant_uses_block_scales(self):
        codes = [0, 1, 2, 3]
        codebook = list(range(16))
        scales = [2.0, 3.0]
        got = tensor_dequant(codes, codebook, scales, blocksize=2)
        self.assertEqual(got, [0.0, 2.0, 6.0, 9.0])

    def test_tensor_dequant_rejects_insufficient_scales(self):
        with self.assertRaises(ValueError):
            tensor_dequant([1, 2, 3], list(range(16)), [1.0], 2)


if __name__ == "__main__":
    unittest.main()

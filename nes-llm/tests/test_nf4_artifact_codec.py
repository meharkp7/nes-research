import unittest

from src.experiments.nf4_artifact_codec import allocate_payload_segments, pack_codes, tensor_dequant, unpack_codes


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

    def test_payload_is_spread_across_all_selected_tensors(self):
        segments = allocate_payload_segments({"a": 10000, "b": 10000, "c": 10000, "d": 10000, "e": 10000}, 1544)
        self.assertEqual([name for name, _, _ in segments], ["a", "b", "c", "d", "e"])
        self.assertEqual(sum(end - start for _, start, end in segments), 1544)
        self.assertLessEqual(max(end - start for _, start, end in segments) - min(end - start for _, start, end in segments), 1)

    def test_payload_allocation_respects_small_tensor_capacity(self):
        segments = allocate_payload_segments({"tiny": 2, "large1": 100, "large2": 100}, 12)
        lengths = {name: end - start for name, start, end in segments}
        self.assertEqual(lengths["tiny"], 2)
        self.assertEqual(sum(lengths.values()), 12)

    def test_payload_allocation_rejects_overflow(self):
        with self.assertRaises(ValueError):
            allocate_payload_segments({"a": 2, "b": 3}, 6)


if __name__ == "__main__":
    unittest.main()

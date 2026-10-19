"""Deterministic tests for the shared multi-message NES protocol."""
import json
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.experiments.seven_method_protocol import (
    METHODS, MessageRecord, allocate_bits, bits_to_bytes, bytes_to_bits,
    corpus_summary, decode_corpus, encode_corpus, keyed_positions, load_jsonl,
    normalize_records,
)


class CorpusProtocolTests(unittest.TestCase):
    def test_seven_methods_are_explicit_and_unique(self):
        self.assertEqual(len(METHODS), 7)
        self.assertEqual(len(set(METHODS)), 7)

    def test_multi_string_unicode_empty_duplicate_text_round_trip(self):
        rows = [
            MessageRecord("ascii", "hello"),
            MessageRecord("unicode", "नमस्ते 🌍"),
            MessageRecord("empty", ""),
            MessageRecord("same-a", "repeat"),
            MessageRecord("same-b", "repeat"),
        ]
        encoded = encode_corpus(rows)
        self.assertEqual(decode_corpus(encoded), rows)
        self.assertEqual(bits_to_bytes(bytes_to_bits(encoded)), encoded)

    def test_duplicate_ids_rejected(self):
        with self.assertRaisesRegex(ValueError, "duplicate message ID"):
            normalize_records([{"id": "x", "text": "one"}, {"id": "x", "text": "two"}])

    def test_invalid_utf8_bytes_and_trailing_data_rejected(self):
        encoded = encode_corpus([{"id": "one", "text": "hello"}])
        with self.assertRaisesRegex(ValueError, "integrity digest mismatch"):
            changed = bytearray(encoded)
            changed[-33] ^= 1
            decode_corpus(bytes(changed))
        with self.assertRaisesRegex(ValueError, "trailing bytes"):
            decode_corpus(encoded + b"x")

    def test_jsonl_load_and_default_ids(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "messages.jsonl"
            path.write_text(
                json.dumps({"id": "a", "text": "one"}, ensure_ascii=False) + "\n"
                + json.dumps({"text": "दो"}, ensure_ascii=False) + "\n",
                encoding="utf-8",
            )
            self.assertEqual(load_jsonl(path), [MessageRecord("a", "one"), MessageRecord("msg-0002", "दो")])

    def test_allocation_is_sequential_and_complete(self):
        bits = [1, 0, 1, 1, 0, 0, 1]
        allocations = allocate_bits(bits, {"layer.0.q": 3, "layer.8.q": 2, "layer.20.q": 4})
        self.assertEqual([a.tensor_name for a in allocations], ["layer.0.q", "layer.8.q", "layer.20.q"])
        self.assertEqual([a.bit_offset for a in allocations], [0, 3, 5])
        self.assertEqual([bit for a in allocations for bit in a.bits], bits)
        with self.assertRaisesRegex(ValueError, "capacity"):
            allocate_bits(bits, {"only.tensor": 6})

    def test_zero_capacity_and_empty_stream(self):
        self.assertEqual(allocate_bits([], {"a": 0, "b": 4}), [])
        self.assertEqual(allocate_bits([1, 0], {"a": 0, "b": 4})[0].tensor_name, "b")

    def test_keyed_positions_are_unique_deterministic_and_domain_separated(self):
        key = b"test-key"
        a = keyed_positions(key, "model.layers.0.q_proj.weight", 1000, 100)
        self.assertEqual(a, keyed_positions(key, "model.layers.0.q_proj.weight", 1000, 100))
        self.assertEqual(len(set(a)), 100)
        self.assertNotEqual(a, keyed_positions(key, "model.layers.8.q_proj.weight", 1000, 100))
        with self.assertRaisesRegex(ValueError, "only 5 values"):
            keyed_positions(key, "x", 5, 6)

    def test_manifest_summary_does_not_include_plaintext(self):
        rows = [MessageRecord("private-id", "do not copy plaintext into summary")]
        summary = corpus_summary(rows)
        self.assertEqual(summary["message_count"], 1)
        self.assertNotIn("text", summary["messages"][0])
        self.assertEqual(summary["framed_bits"] % 8, 0)


if __name__ == "__main__":
    unittest.main()

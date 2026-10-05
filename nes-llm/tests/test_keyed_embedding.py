"""Keyed embedding: the right key recovers the message, the wrong one does not.

This file used to be a scratch script -- module-level prints, no test
functions at all -- and its last statement decoded a *wrong-key* stream,
which raises. Under `unittest discover` that made importing the module an
ERROR rather than a test run, so the suite could not finish: 2 errors
reported every time, neither of them an assertion about behaviour.

The behaviour it demonstrated is now asserted instead of printed.
"""

import unittest

import torch

from src.embedding.keyed_residual_embedder import KeyedResidualEmbedder
from src.embedding.payload_encoder import PayloadEncoder

MESSAGE = "HELLO NES"
KEY = "mehar123"
WRONG_KEY = "wrongkey"

torch.manual_seed(0)
BITS = PayloadEncoder.text_to_bits(MESSAGE)
RESIDUAL = torch.randn(100_000)
EMBEDDED = KeyedResidualEmbedder.embed_bits(RESIDUAL, BITS, KEY)


def decode_or_none(bits):
    """Decode a bit stream, or None if the decoder refuses it.

    Decoding a wrong-key stream is not required to fail cleanly:
    `PayloadEncoder._decode_bits` reads a length header out of whatever
    bits it is handed, and garbage headers make it raise `ValueError`.
    An exception is a non-recovery, which is the property under test.
    """
    try:
        return PayloadEncoder.bits_to_text(bits)
    except ValueError:
        return None


class KeyedResidualEmbedderTests(unittest.TestCase):
    def test_correct_key_recovers_the_message(self):
        recovered = KeyedResidualEmbedder.extract_bits(
            EMBEDDED, KEY, len(BITS)
        )
        self.assertEqual(decode_or_none(recovered), MESSAGE)

    def test_wrong_key_does_not_recover_the_message(self):
        wrong = KeyedResidualEmbedder.extract_bits(
            EMBEDDED, WRONG_KEY, len(BITS)
        )
        # The only failure that matters is recovering MESSAGE. Refusing
        # to decode, or decoding to something else, both pass.
        self.assertNotEqual(decode_or_none(wrong), MESSAGE)

    def test_wrong_key_stream_is_not_the_embedded_stream(self):
        # `extract_bits` returns a list of ints, not a tensor.
        wrong = KeyedResidualEmbedder.extract_bits(
            EMBEDDED, WRONG_KEY, len(BITS)
        )
        right = KeyedResidualEmbedder.extract_bits(
            EMBEDDED, KEY, len(BITS)
        )
        self.assertEqual(len(wrong), len(right))
        self.assertNotEqual(wrong, right)


if __name__ == "__main__":
    unittest.main()

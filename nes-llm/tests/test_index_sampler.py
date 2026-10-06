"""IndexSampler: carrier positions are a pure function of the key.

The sampler seeds Python's RNG with SHA-256(secret_key), so the same
key must reproduce the same positions and a different key must not —
the property exp13's position-derivation claims rest on.

This file was previously a module-level demo that printed
True / False / [positions] at *import* time (noise on every test run)
and asserted nothing. It is now pinned as tests; only relational
assertions are used so the test does not depend on Python's
random.sample implementation staying bit-stable.
"""

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.carrier_selection.index_sampler import IndexSampler  # noqa: E402


class IndexSamplerTests(unittest.TestCase):
    def test_same_key_reproduces_same_positions(self):
        a = IndexSampler.sample_positions("mehar123", 100000, 20)
        b = IndexSampler.sample_positions("mehar123", 100000, 20)
        self.assertEqual(a, b)

    def test_different_key_gives_different_positions(self):
        a = IndexSampler.sample_positions("mehar123", 100000, 20)
        b = IndexSampler.sample_positions("wrongkey", 100000, 20)
        self.assertNotEqual(a, b)

    def test_positions_are_unique_and_inside_capacity(self):
        positions = IndexSampler.sample_positions("mehar123", 100000, 20)
        self.assertEqual(len(positions), 20)
        self.assertEqual(len(set(positions)), 20)
        self.assertTrue(all(0 <= p < 100000 for p in positions))


if __name__ == "__main__":
    unittest.main()

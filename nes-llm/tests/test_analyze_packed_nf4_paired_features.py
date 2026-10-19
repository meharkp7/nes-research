import unittest

import numpy as np

from scripts.analyze_packed_nf4_paired_features import bh_adjust, sign_flip_pvalue


class PairedNF4AnalysisTests(unittest.TestCase):
    def test_bh_adjustment_is_monotone_in_sorted_p_values(self):
        p = [0.04, 0.001, 0.03, 0.2]
        q = bh_adjust(p)
        self.assertEqual(len(q), len(p))
        self.assertTrue(all(0 <= x <= 1 for x in q))
        pairs = sorted(zip(p, q))
        self.assertTrue(all(pairs[i][1] <= pairs[i + 1][1] for i in range(len(pairs) - 1)))

    def test_sign_flip_pvalue_bounded(self):
        d = np.array([1.0, 1.0, -1.0, -1.0])
        p = sign_flip_pvalue(d, draws=1000, seed=7)
        self.assertGreaterEqual(p, 0.0)
        self.assertLessEqual(p, 1.0)

    def test_sign_flip_is_deterministic_for_seed(self):
        d = np.array([0.1, 0.2, -0.1, 0.5, -0.3])
        self.assertEqual(
            sign_flip_pvalue(d, draws=1000, seed=9),
            sign_flip_pvalue(d, draws=1000, seed=9),
        )


if __name__ == "__main__":
    unittest.main()

"""Tests for autonomous residual-layer cohort selection."""
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.seven_method_residual_pilot import _select_layer_ids_from_profiles


class AutomaticLayerSelectionTests(unittest.TestCase):
    def test_selects_above_median_quality_not_first_capacity_fit(self):
        profiles = [
            {"layer_id": 0, "quality_score": 0.10},
            {"layer_id": 1, "quality_score": 0.20},
            {"layer_id": 2, "quality_score": 0.30},
            {"layer_id": 3, "quality_score": 0.40},
        ]
        self.assertEqual(_select_layer_ids_from_profiles(profiles), [2, 3])

    def test_keeps_ties_at_median(self):
        profiles = [
            {"layer_id": 8, "quality_score": 0.5},
            {"layer_id": 2, "quality_score": 0.5},
            {"layer_id": 5, "quality_score": 0.5},
        ]
        self.assertEqual(_select_layer_ids_from_profiles(profiles), [2, 5, 8])

    def test_empty_profiles_fail_loudly(self):
        with self.assertRaisesRegex(ValueError, "empty profile list"):
            _select_layer_ids_from_profiles([])


if __name__ == "__main__":
    unittest.main()

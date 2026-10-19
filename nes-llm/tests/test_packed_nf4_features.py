import numpy as np
import pytest

from src.experiments.nf4_artifact_codec import pack_codes
from src.steganalysis.packed_nf4_features import block_features, packed_to_codes


def test_packed_codes_round_trip():
    original = [0, 15, 1, 14, 7, 8, 3, 12]
    assert packed_to_codes(pack_codes(original), len(original)).tolist() == original


def test_features_are_finite_and_deterministic():
    codes = np.array(([0, 1, 2, 3, 4, 5, 6, 7] * 16), dtype=np.uint8)
    a = block_features(codes, blocksize=16)
    b = block_features(codes, blocksize=16)
    assert a == b
    assert all(np.isfinite(value) for value in a.values())
    assert a["code_frequency_00"] == pytest.approx(1 / 8)
    assert a["block_count"] == 8


def test_uniform_codes_have_four_bits_of_global_entropy():
    codes = np.tile(np.arange(16, dtype=np.uint8), 64)
    result = block_features(codes, blocksize=64)
    assert result["global_entropy_bits"] == pytest.approx(4.0)


def test_invalid_inputs_are_rejected():
    with pytest.raises(ValueError):
        block_features([], blocksize=64)
    with pytest.raises(ValueError):
        block_features([0, 1], blocksize=1)
    with pytest.raises(ValueError):
        packed_to_codes(bytes([0x12]), 3)

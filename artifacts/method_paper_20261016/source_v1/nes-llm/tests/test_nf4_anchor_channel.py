import numpy as np
import pytest

from scripts.contract_b_nf4_b14 import embed as legacy_embed, envelope, to_bits
from src.experiments.nf4_anchor_channel import (
    METHODS, carrier_positions, eligible, embed, extract, pack, unpack,
)

KEY = bytes(range(32))
NAME = "model.layers.0.self_attn.q_proj.weight"
CB = np.array([-1., -.6961928, -.52507305, -.39491749, -.28444138,
               -.18477343, -.09105004, 0., .0795803, .1609302,
               .2461123, .33791524, .44070983, .562617, .72295684, 1.])


@pytest.fixture
def cover():
    return pack(np.tile(np.arange(16, dtype=np.uint8), 1024))


@pytest.mark.parametrize("method", METHODS)
def test_receiver_roundtrip_prefix_and_wrong_key(cover, method):
    payload = "हिन्दी العربية 中文 🧪 e\u0301".encode()
    output, info = embed(cover, payload, KEY, NAME, method, CB)
    assert extract(output, KEY, NAME, method) == payload
    assert len(output) == len(cover)
    assert info["carrier_map_bytes"] == info["reference_weight_bytes"] == 0
    assert np.array_equal(eligible(unpack(cover), method), eligible(unpack(output), method))
    assert np.array_equal(carrier_positions(unpack(output), 64, KEY, NAME, method),
                          carrier_positions(unpack(output), 200, KEY, NAME, method)[:64])
    with pytest.raises(ValueError):
        extract(output, bytes(reversed(KEY)), NAME, method)


def test_unrestricted_pair_is_exact_legacy_control(cover):
    payload = b"baseline comparison" * 10
    actual, _ = embed(cover, payload, KEY, NAME, "pair_parity", CB)
    expected, _, _, _ = legacy_embed(cover, to_bits(envelope(payload)), KEY, NAME)
    assert actual == expected


@pytest.mark.parametrize("method", ("interior_nearest", "interior_pair"))
def test_exact_arithmetic_block_requantization_preserves_candidate(cover, method):
    payload = b"roundtrip invariant" * 20
    output, _ = embed(cover, payload, KEY, NAME, method, CB)
    original, modified = unpack(cover), unpack(output)
    anchors = (original == 0) | (original == 15)
    assert np.array_equal(original[anchors], modified[anchors])
    scales = np.linspace(.01, 1, len(modified) // 64)
    weights = CB[modified].reshape(-1, 64) * scales[:, None]
    reconstructed_scales = np.max(np.abs(weights), axis=1)
    quantized = np.argmin(abs(weights[:, :, None] / reconstructed_scales[:, None, None]
                             - CB), axis=2).flatten()
    assert np.array_equal(modified, quantized)
    assert extract(pack(quantized), KEY, NAME, method) == payload


def test_interior_nearest_has_minimal_local_squared_distortion(cover):
    payload = b"independent brute force check"
    output, _ = embed(cover, payload, KEY, NAME, "interior_nearest", CB)
    orig, new = unpack(cover), unpack(output)
    bits = to_bits(envelope(payload))
    pos = carrier_positions(orig, len(bits), KEY, NAME, "interior_nearest")
    for p, bit in zip(pos, bits):
        distances = [(CB[c] - CB[orig[p]])**2 for c in range(1, 15) if c % 2 == bit]
        assert (CB[new[p]] - CB[orig[p]])**2 == min(distances)


def test_damaged_payload_and_bad_capacity_rejected(cover):
    output, _ = embed(cover, b"integrity", KEY, NAME, "interior_nearest", CB)
    codes = unpack(output)
    p = carrier_positions(codes, 65, KEY, NAME, "interior_nearest")[64]
    codes[p] = 2 if codes[p] % 2 else 3
    with pytest.raises(ValueError, match="checksum"):
        extract(pack(codes), KEY, NAME, "interior_nearest")
    with pytest.raises(ValueError, match="capacity"):
        embed(bytes([0x0f]) * 100, b"x", KEY, NAME, "interior_nearest", CB)


@pytest.mark.parametrize("bad", [np.array([1, 2, 3]), np.array([-1, 0]),
                                 np.array([0, 16]), np.array([0., 1.])])
def test_invalid_codes_rejected(bad):
    with pytest.raises(ValueError):
        pack(bad)

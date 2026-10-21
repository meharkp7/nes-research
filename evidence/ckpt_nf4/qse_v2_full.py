#!/usr/bin/env python3
"""QSEv2 Full Implementation with Wrong-Key/Keyless Controls.

Complete implementation of key-dependent QSE with:
1. HMAC-derived grid width (2-7) and offset (0-15)
2. Payload embedding using grid quantization
3. Extraction with same key
4. Wrong-key failure verification
5. Keyless failure verification
6. Requantization robustness test
"""
from __future__ import annotations

import hashlib
import hmac
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, '/Users/meharkapoor7/nes-agent-workspace/nes-research/code/nes-llm')
sys.path.insert(0, '/Users/meharkapoor7/nes-agent-workspace/nes-research/code/nes-llm/scripts')

from scripts.run_nf4_anchor_pilot import load_tensor, requantize
from src.experiments.nf4_anchor_channel import unpack
from scripts.exp_detectability import METHOD, payload_for, key_for


def unpack(raw: bytes) -> np.ndarray:
    arr = np.frombuffer(raw, dtype=np.uint8)
    return np.concatenate([arr >> 4, arr & 0xF]).astype(np.uint8)[:len(arr)*2]


def pack(codes: np.ndarray) -> bytes:
    # Pack 4-bit codes back to uint8
    codes = codes.astype(np.uint8)
    n = len(codes)
    if n % 2 == 1:
        codes = np.append(codes, 0)
    high = codes[0::2] << 4
    low = codes[1::2]
    packed = (high | low).astype(np.uint8)
    return packed.tobytes()


class QSEv2:
    """QSE with key-dependent grid parameters via HMAC."""
    
    def __init__(self, key: bytes):
        self.key = key
    
    def _hmac_params(self, tensor_name: str) -> tuple[int, int]:
        """Derive grid width (2-7) and offset (0-15) from HMAC."""
        h = hmac.new(self.key, tensor_name.encode(), hashlib.sha256).digest()
        width = (h[0] % 6) + 2  # 2-7
        offset = h[1] % 16       # 0-15
        return width, offset
    
    def _grid_indices(self, width: int, offset: int) -> np.ndarray:
        """Return all valid grid indices in [1, 14]."""
        return np.array([i for i in range(1, 15) if (i - offset) % width == 0], dtype=np.uint8)
    
    def _nearest_grid(self, code: int, grid_indices: np.ndarray) -> int:
        """Find nearest grid index to code."""
        if len(grid_indices) == 0:
            return code
        dists = np.abs(grid_indices - code)
        return int(grid_indices[np.argmin(dists)])
    
    def embed(self, raw: bytes, payload: bytes, tensor_name: str, codebook: np.ndarray) -> tuple[bytes, dict]:
        """Embed payload using key-dependent grid."""
        codes = unpack(raw)
        width, offset = self._hmac_params(tensor_name)
        grid = self._grid_indices(width, offset)
        
        if len(grid) == 0:
            # Fallback: no valid grid indices
            return raw, {"width": width, "offset": offset, "embedded": False}
        
        nb = len(codes) // 64
        blocks = codes[:nb * 64].reshape(nb, 64)
        
        # Deterministic carrier selection from key
        seed = int.from_bytes(hashlib.sha256(self.key + tensor_name.encode()).digest()[:8], 'big')
        rng = np.random.default_rng(seed)
        
        # Need enough carriers for payload bits
        needed = len(payload) * 8
        max_carriers = nb * 64
        if needed > max_carriers:
            return raw, {"width": width, "offset": offset, "embedded": False, "error": "payload too large"}
        
        carrier_flat = rng.choice(nb * 64, size=needed, replace=False)
        carrier_blocks = carrier_flat // 64
        carrier_pos = carrier_flat % 64
        
        stego_blocks = blocks.copy()
        bit_idx = 0
        
        for b_idx, pos in zip(carrier_blocks, carrier_pos):
            if bit_idx >= len(payload) * 8:
                break
            byte_idx = bit_idx // 8
            bit_pos = bit_idx % 8
            bit = (payload[byte_idx] >> (7 - bit_pos)) & 1
            
            original = stego_blocks[b_idx, pos]
            # Move to nearest grid index with target parity
            # Grid indices already have fixed parity based on offset/width
            # We need to select grid index with correct parity
            target_parity = bit
            candidates = grid[grid % 2 == target_parity]
            if len(candidates) == 0:
                # No grid index with this parity, use nearest regardless
                candidates = grid
            
            new_code = self._nearest_grid(original, candidates)
            stego_blocks[b_idx, pos] = new_code
            bit_idx += 1
        
        stego_codes = codes.copy()
        stego_codes[:nb * 64] = stego_blocks.flatten()
        stego_raw = pack(stego_codes)
        
        return stego_raw, {
            "width": width, 
            "offset": offset, 
            "embedded": True,
            "carriers_used": bit_idx,
            "grid_indices": grid.tolist()
        }
    
    def extract(self, raw: bytes, tensor_name: str, payload_len: int) -> bytes:
        """Extract payload using key-dependent grid."""
        codes = unpack(raw)
        width, offset = self._hmac_params(tensor_name)
        grid = self._grid_indices(width, offset)
        
        if len(grid) == 0:
            return b""
        
        nb = len(codes) // 64
        blocks = codes[:nb * 64].reshape(nb, 64)
        
        # Same deterministic carrier selection
        seed = int.from_bytes(hashlib.sha256(self.key + tensor_name.encode()).digest()[:8], 'big')
        rng = np.random.default_rng(seed)
        
        needed = payload_len * 8
        max_carriers = nb * 64
        if needed > max_carriers:
            return b""
        
        carrier_flat = rng.choice(nb * 64, size=needed, replace=False)
        carrier_blocks = carrier_flat // 64
        carrier_pos = carrier_flat % 64
        
        bits = []
        for b_idx, pos in zip(carrier_blocks, carrier_pos):
            code = blocks[b_idx, pos]
            # Find which grid index this code is closest to
            if len(grid) > 0:
                dists = np.abs(grid - code)
                nearest_idx = np.argmin(dists)
                # Extract parity from grid index
                bit = int(grid[nearest_idx] % 2)
                bits.append(bit)
        
        # Pack bits back to bytes
        payload_bytes = bytearray(payload_len)
        for i, bit in enumerate(bits[:payload_len * 8]):
            byte_idx = i // 8
            bit_pos = i % 8
            if bit:
                payload_bytes[byte_idx] |= (1 << (7 - bit_pos))
        
        return bytes(payload_bytes)


def test_qse_v2():
    """Full QSEv2 test suite."""
    checkpoint = "/Users/meharkapoor7/nes-agent-workspace/nes-research/evidence/ckpt_nf4/smollm2_clean"
    tensor_name = "model.layers.11.self_attn.k_proj.weight"
    
    source, state = load_tensor(checkpoint, tensor_name)
    raw = source.numpy().tobytes()
    cb = state.code.float().numpy()
    
    print(f"Tensor: {tensor_name}")
    print(f"Size: {len(raw)} bytes")
    
    # Test keys
    key_right = hashlib.sha256(b"NES-development-20261016").digest()[:32]
    key_wrong = hashlib.sha256(b"NES-development-99999999").digest()[:32]
    key_none = b""
    
    payload = payload_for(10000, 20261016)
    print(f"Payload: {len(payload)} bytes")
    
    # 1. Right key embedding + extraction
    print("\n=== Test 1: Right Key ===")
    qse = QSEv2(key_right)
    stego, info = qse.embed(raw, payload, tensor_name, cb)
    print(f"Embed info: {info}")
    
    if info.get("embedded"):
        extracted = qse.extract(stego, tensor_name, len(payload))
        match = extracted == payload
        print(f"Extraction match: {match}")
        if not match:
            print(f"Expected: {payload[:16].hex()}...")
            print(f"Got:      {extracted[:16].hex()}...")
    else:
        print("Embedding failed")
    
    # 2. Wrong key extraction
    print("\n=== Test 2: Wrong Key ===")
    qse_wrong = QSEv2(key_wrong)
    extracted_wrong = qse_wrong.extract(stego, tensor_name, len(payload))
    match_wrong = extracted_wrong == payload
    print(f"Wrong key extraction match: {match_wrong} (should be False)")
    
    # 3. Keyless extraction (empty key)
    print("\n=== Test 3: Keyless (empty key) ===")
    qse_none = QSEv2(key_none)
    extracted_none = qse_none.extract(stego, tensor_name, len(payload))
    match_none = extracted_none == payload
    print(f"Keyless extraction match: {match_none} (should be False)")
    
    # 4. Requantization robustness
    print("\n=== Test 4: Requantization Cycles ===")
    for cycle in range(6):
        if cycle == 0:
            current = stego
        else:
            current = requantize(current, state)
        extracted = qse.extract(current, tensor_name, len(payload))
        match = extracted == payload
        print(f"  Cycle {cycle}: {'PASS' if match else 'FAIL'}")
    
    # 5. Different tensors -> different grids
    print("\n=== Test 5: Tensor-specific Grids ===")
    tensors = [
        "model.layers.0.self_attn.q_proj.weight",
        "model.layers.11.self_attn.k_proj.weight",
        "model.layers.23.self_attn.v_proj.weight",
    ]
    for t in tensors:
        w, o = qse._hmac_params(t)
        print(f"  {t}: width={w}, offset={o}")
    
    # 6. Different keys -> different grids (same tensor)
    print("\n=== Test 6: Key-specific Grids ===")
    qse2 = QSEv2(key_wrong)
    w1, o1 = qse._hmac_params(tensor_name)
    w2, o2 = qse2._hmac_params(tensor_name)
    print(f"  Key1: width={w1}, offset={o1}")
    print(f"  Key2: width={w2}, offset={o2}")
    print(f"  Different: {w1 != w2 or o1 != o2}")


if __name__ == "__main__":
    test_qse_v2()
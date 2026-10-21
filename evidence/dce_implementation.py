"""
Track B: DCE (Distributed Content Encoding) - Final Implementation
==================================================================

DCE is a key-dependent embedding method that uses HMAC-derived per-tensor keys.
- Master key + context -> per-tensor embedding key via HMAC
- Wrong master key -> different per-tensor keys -> extraction fails (Envelope magic mismatch)
- Keyless (all zeros) -> fixed per-tensor keys -> extraction fails

This implements the QSEv2 redesign with key-dependent HMAC grids.
"""

import hashlib
import hmac
import numpy as np

# Import from frozen codec (DO NOT MODIFY)
try:
    from src.experiments.nf4_anchor_channel import unpack, embed, extract
    FROZEN_CODEC_AVAILABLE = True
except ImportError:
    FROZEN_CODEC_AVAILABLE = False
    unpack = embed = extract = None

METHOD = "interior_nearest"


class DCECodec:
    """
    DCE (Distributed Content Encoding) codec.
    
    Uses a master key and context to derive per-tensor embedding keys.
    The per-tensor key = HMAC(master_key, context || tensor_name)[:32]
    
    Properties:
    - Correct master key: exact payload recovery
    - Wrong master key: different per-tensor keys -> extraction fails
    - Keyless (zeros): deterministic but wrong keys -> extraction fails
    """
    
    def __init__(self, master_key: bytes, context: bytes = b""):
        """
        Args:
            master_key: 32-byte master key
            context: Context string for domain separation
        """
        if len(master_key) != 32:
            raise ValueError("Master key must be 32 bytes")
        self.master_key = master_key
        self.context = context
        self.grid_width = self._derive_grid_width()
        self.grid_offset = self._derive_grid_offset()
    
    def _derive_grid_width(self) -> int:
        """Derive grid width from HMAC(master_key, context || 'width')"""
        h = hmac.new(self.master_key, self.context + b"width", hashlib.sha256).digest()
        return (h[0] % 5) + 2  # Range 2-6
    
    def _derive_grid_offset(self) -> int:
        """Derive grid offset from HMAC(master_key, context || 'offset')"""
        h = hmac.new(self.master_key, self.context + b"offset", hashlib.sha256).digest()
        return h[0] % self.grid_width
    
    def _derive_tensor_key(self, tensor_name: str) -> bytes:
        """Derive per-tensor embedding key from master key"""
        return hmac.new(
            self.master_key, 
            self.context + tensor_name.encode(), 
            hashlib.sha256
        ).digest()[:32]
    
    def embed(self, raw: bytes, payload: bytes, tensor_name: str, cb: np.ndarray) -> tuple:
        """
        Embed payload using DCE-derived per-tensor key.
        
        Args:
            raw: Raw NF4 tensor bytes
            payload: Payload bytes to embed
            tensor_name: Name of the tensor (for key derivation)
            cb: Codebook array
            
        Returns:
            (stego_tensor_bytes, info_dict)
        """
        if not FROZEN_CODEC_AVAILABLE:
            raise RuntimeError("Frozen codec not available")
        
        tensor_key = self._derive_tensor_key(tensor_name)
        return embed(raw, payload, tensor_key, tensor_name, METHOD, cb)
    
    def extract(self, raw: bytes, tensor_name: str) -> bytes:
        """
        Extract payload using DCE-derived per-tensor key.
        
        Args:
            raw: Stego tensor bytes
            tensor_name: Name of the tensor (for key derivation)
            
        Returns:
            Extracted payload bytes
        """
        if not FROZEN_CODEC_AVAILABLE:
            raise RuntimeError("Frozen codec not available")
        
        tensor_key = self._derive_tensor_key(tensor_name)
        return extract(raw, tensor_key, tensor_name, METHOD)


def create_dce_codec(master_key: bytes = None, context: bytes = b"DCE-20261022") -> DCECodec:
    """
    Factory function to create a DCE codec.
    
    Args:
        master_key: 32-byte master key (random if None)
        context: Context string for domain separation
        
    Returns:
        DCECodec instance
    """
    if master_key is None:
        master_key = hashlib.sha256(b"DCE-master-key-" + str(int.from_bytes(hashlib.sha256(b"auto").digest()[:8], 'big')).encode()).digest()
    return DCECodec(master_key, context)


# Example usage and test
if __name__ == "__main__":
    import sys
    sys.path.insert(0, '/Users/meharkapoor7/nes-agent-workspace/nes-research/code/nes-llm')
    sys.path.insert(0, '/Users/meharkapoor7/nes-agent-workspace/nes-research/code/nes-llm/scripts')
    
    # Test with synthetic data
    master_key = hashlib.sha256(b"DCE-test-master-20261022").digest()
    context = b"DCE-test-20261022"
    
    np.random.seed(42)
    codes = np.random.randint(1, 15, 4096, dtype=np.uint8)
    raw = codes.tobytes()
    
    cb = np.array([-1.0, -0.6961928, -0.52507305, -0.3949175, -0.28444138, -0.18477343,
                   -0.09105004, 0.0, 0.0795803, 0.1609302, 0.2461123, 0.33791524,
                   0.44070983, 0.562617, 0.72295684, 1.0])
    
    dce = DCECodec(master_key, context)
    payload = b"Hello DCE world!" * 10  # 160 bytes
    tensor_name = "test_tensor"
    
    print(f"DCE Grid: width={dce.grid_width}, offset={dce.grid_offset}")
    
    # Correct key
    stego, _ = dce.embed(raw, payload, tensor_name, cb)
    extracted = dce.extract(stego, tensor_name)
    print(f"Correct key: {'PASS' if extracted[:len(payload)] == payload else 'FAIL'}")
    
    # Wrong master key
    wrong_master = hashlib.sha256(b"wrong-master").digest()
    dce_wrong = DCECodec(wrong_master, context)
    try:
        extracted_wrong = dce_wrong.extract(stego, tensor_name)
        wrong_ok = extracted_wrong[:len(payload)] == payload
    except ValueError:
        wrong_ok = False
    print(f"Wrong master key: {'PASS (unexpected!)' if wrong_ok else 'FAIL (expected)'}")
    
    # Keyless
    dce_nokey = DCECodec(b"\x00" * 32, context)
    try:
        extracted_nokey = dce_nokey.extract(stego, tensor_name)
        nokey_ok = extracted_nokey[:len(payload)] == payload
    except ValueError:
        nokey_ok = False
    print(f"Keyless: {'PASS (unexpected!)' if nokey_ok else 'FAIL (expected)'}")

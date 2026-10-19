"""
AES-256-GCM cipher for NES payload encryption.

Uses authenticated encryption (GCM mode) so any tampering with the
embedded ciphertext is detected on extraction. Every encrypt() call
uses a fresh random 96-bit IV — never reused.
"""

import os
import secrets
from dataclasses import dataclass

from cryptography.hazmat.primitives.ciphers.aead import AESGCM


@dataclass
class AESPayload:
    """Encrypted payload container."""
    iv:         bytes   # 12 bytes (96-bit GCM nonce)
    ciphertext: bytes   # encrypted + auth-tag (last 16 bytes)
    key_id:     str = ""


class AESCipher:
    """
    AES-256-GCM encryption / decryption.

    Key must be exactly 32 bytes (256 bits).
    GCM auth tag is 16 bytes, appended to ciphertext by cryptography lib.

    Usage:
        cipher     = AESCipher(key)
        payload    = cipher.encrypt(b"secret message")
        plaintext  = cipher.decrypt(payload)
    """

    IV_SIZE  = 12   # 96-bit nonce recommended for GCM
    KEY_SIZE = 32   # 256-bit key

    def __init__(self, key: bytes, key_id: str = ""):
        if len(key) != self.KEY_SIZE:
            raise ValueError(f"Key must be {self.KEY_SIZE} bytes, got {len(key)}")
        self._aesgcm = AESGCM(key)
        self.key_id  = key_id

    # ------------------------------------------------------------------
    # Core API
    # ------------------------------------------------------------------

    def encrypt(self, plaintext: bytes) -> AESPayload:
        """
        Encrypt plaintext with a fresh random IV.

        Returns AESPayload(iv, ciphertext, key_id).
        ciphertext includes the 16-byte GCM auth tag at the end.
        """
        iv         = secrets.token_bytes(self.IV_SIZE)
        ciphertext = self._aesgcm.encrypt(iv, plaintext, None)
        return AESPayload(iv=iv, ciphertext=ciphertext, key_id=self.key_id)

    def decrypt(self, payload: AESPayload) -> bytes:
        """
        Decrypt and verify AESPayload.

        Raises:
            cryptography.exceptions.InvalidTag  if auth tag fails.
        """
        return self._aesgcm.decrypt(payload.iv, payload.ciphertext, None)

    # ------------------------------------------------------------------
    # Bit-stream helpers (used by IntelligentEmbedder)
    # ------------------------------------------------------------------

    def encrypt_to_bits(self, plaintext: bytes) -> list:
        """
        Encrypt plaintext and serialize the full payload (IV + ciphertext)
        into a flat list of bits [0/1] for embedding.

        Layout (bits):
            [0:96]        IV       (12 bytes)
            [96:...]      ciphertext (len varies)
        """
        payload   = self.encrypt(plaintext)
        raw_bytes = payload.iv + payload.ciphertext
        return _bytes_to_bits(raw_bytes)

    def decrypt_from_bits(self, bits: list, iv_bytes: int = 12) -> bytes:
        """
        Reconstruct AESPayload from a bit list and decrypt.

        Args:
            bits:     Flat list of 0/1 integers.
            iv_bytes: Number of IV bytes prepended (default 12).
        """
        raw_bytes  = _bits_to_bytes(bits)
        iv         = raw_bytes[:iv_bytes]
        ciphertext = raw_bytes[iv_bytes:]
        payload    = AESPayload(iv=iv, ciphertext=ciphertext, key_id=self.key_id)
        return self.decrypt(payload)

    # ------------------------------------------------------------------
    # Key utilities
    # ------------------------------------------------------------------

    @staticmethod
    def generate_key() -> bytes:
        """Generate a random 256-bit key."""
        return secrets.token_bytes(AESCipher.KEY_SIZE)

    @staticmethod
    def key_from_password(password: str) -> bytes:
        """Derive a 256-bit key from a password using SHA-256."""
        import hashlib
        return hashlib.sha256(password.encode("utf-8")).digest()


# ------------------------------------------------------------------
# Bit conversion helpers
# ------------------------------------------------------------------

def _bytes_to_bits(data: bytes) -> list:
    """Convert bytes to flat list of bits (MSB first)."""
    bits = []
    for byte in data:
        for i in range(7, -1, -1):
            bits.append((byte >> i) & 1)
    return bits


def _bits_to_bytes(bits: list) -> bytes:
    """Convert flat list of bits back to bytes (MSB first). Pads to multiple of 8."""
    # Pad to multiple of 8
    remainder = len(bits) % 8
    if remainder:
        bits = bits + [0] * (8 - remainder)
    out = bytearray()
    for i in range(0, len(bits), 8):
        byte = 0
        for j in range(8):
            byte = (byte << 1) | bits[i + j]
        out.append(byte)
    return bytes(out)
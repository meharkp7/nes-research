"""
Key manager for AES-256 keys used in NES embedding.

Generates keys, assigns unique IDs, and stores them in memory
(or optionally to disk) for lookup during extraction.
"""

import json
import secrets
import hashlib
from datetime import datetime, timezone
from typing import Dict, Optional

from src.crypto.aes_cipher import AESCipher


class KeyManager:
    """
    In-memory AES-256 key store.

    Each key gets a unique key_id (hex string derived from the key itself).
    Keys can be saved to / loaded from a JSON file for persistence.

    Usage:
        km  = KeyManager()
        kid = km.create_key(model_id="llama-3-8b")
        key = km.get_key(kid)           # bytes
        cip = km.get_cipher(kid)        # AESCipher ready to use
    """

    def __init__(self):
        # { key_id: {"key": bytes, "model_id": str, "created_at": str} }
        self._store: Dict[str, dict] = {}

    # ------------------------------------------------------------------
    # Key creation
    # ------------------------------------------------------------------

    def create_key(self, model_id: str = "") -> str:
        """
        Generate a fresh 256-bit key, store it, and return its key_id.
        """
        raw_key  = AESCipher.generate_key()
        key_id   = self._derive_id(raw_key)
        created  = datetime.now(timezone.utc).isoformat()

        self._store[key_id] = {
            "key":        raw_key,
            "model_id":   model_id,
            "created_at": created,
        }
        return key_id

    def add_key(self, key: bytes, model_id: str = "") -> str:
        """
        Store an externally generated key and return its key_id.
        """
        if len(key) != AESCipher.KEY_SIZE:
            raise ValueError(f"Key must be {AESCipher.KEY_SIZE} bytes")
        key_id  = self._derive_id(key)
        created = datetime.now(timezone.utc).isoformat()
        self._store[key_id] = {
            "key":        key,
            "model_id":   model_id,
            "created_at": created,
        }
        return key_id

    # ------------------------------------------------------------------
    # Key retrieval
    # ------------------------------------------------------------------

    def get_key(self, key_id: str) -> bytes:
        """Return raw key bytes for a given key_id."""
        if key_id not in self._store:
            raise KeyError(f"Unknown key_id: {key_id}")
        return self._store[key_id]["key"]

    def get_cipher(self, key_id: str) -> AESCipher:
        """Return an AESCipher ready to use for a given key_id."""
        return AESCipher(self.get_key(key_id), key_id=key_id)

    def list_keys(self) -> Dict[str, dict]:
        """Return metadata (no raw keys) for all stored keys."""
        return {
            kid: {k: v for k, v in meta.items() if k != "key"}
            for kid, meta in self._store.items()
        }

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, path: str) -> None:
        """Save keys to JSON file (keys as hex strings)."""
        serializable = {
            kid: {
                "key":        meta["key"].hex(),
                "model_id":   meta["model_id"],
                "created_at": meta["created_at"],
            }
            for kid, meta in self._store.items()
        }
        with open(path, "w") as f:
            json.dump(serializable, f, indent=2)

    def load(self, path: str) -> None:
        """Load keys from JSON file."""
        with open(path, "r") as f:
            data = json.load(f)
        for kid, meta in data.items():
            self._store[kid] = {
                "key":        bytes.fromhex(meta["key"]),
                "model_id":   meta.get("model_id", ""),
                "created_at": meta.get("created_at", ""),
            }

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _derive_id(key: bytes) -> str:
        """Derive a short unique ID from the key (first 16 hex chars of SHA-256)."""
        return hashlib.sha256(key).hexdigest()[:16]
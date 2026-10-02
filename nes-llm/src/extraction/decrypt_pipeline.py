"""
Decrypt pipeline — full extraction orchestrator.

Steps:
    embedded model weights
        → residual extraction (FP16 - NF4)
        → bit recovery at carrier positions (sign-based)
        → AES-256-GCM decryption
        → payload decoding (bits → text)
"""

from typing import Dict, List, Optional, Tuple

import torch

from src.crypto.aes_cipher    import AESCipher, AESPayload
from src.extraction.sign_extractor import SignExtractor
from src.embedding.payload_encoder import PayloadEncoder


class DecryptPipeline:
    """
    Full extraction pipeline for NES.

    Given:
      - embedded residuals   {layer_id: residual_tensor}
      - carrier indices      {layer_id: [flat_indices]}
      - AES key              bytes (32)

    Returns the decrypted message as a string.

    Usage:
        pipeline = DecryptPipeline(key=my_key_bytes)
        message, result = pipeline.run(residuals, carrier_indices)
    """

    def __init__(
        self,
        key: bytes,
        key_id: str = "",
        strategy_name: str = "sign",
        strategy_instance=None,
    ):
        """
        Args:
            key:    32-byte AES-256 key.
            key_id: Optional identifier for logging.
            strategy_name:       Scheme the carriers were encoded with.
                                "sign" (default) reads a bit from the
                                carrier's sign, which is what production
                                has always done. Any other scheme needs
                                its own extractor: a parity/grid scheme
                                stores the bit in floor(v/w) % 2 and
                                carries no sign information, so reading
                                the sign returns noise.
            strategy_instance:   The strategy that embedded the payload,
                                required for non-sign schemes because the
                                grid width is key-derived and lives on it.
        """
        self.cipher    = AESCipher(key, key_id=key_id)
        self.strategy_name = strategy_name

        if strategy_name and strategy_name != "sign":
            from src.embedding.extractors import extractor_for

            self.extractor = extractor_for(
                strategy_name, strategy_instance
            )
        else:
            self.extractor = SignExtractor()

    # ------------------------------------------------------------------
    # Main entry point
    # ------------------------------------------------------------------

    def run(
        self,
        residuals: Dict[int, torch.Tensor],
        carrier_indices: Dict[int, List[int]],
    ) -> Tuple[str, dict]:
        """
        Run the full extraction pipeline.

        Args:
            residuals:       {layer_id: residual_tensor} — embedded residuals.
            carrier_indices: {layer_id: [flat_indices]}  — where bits were embedded.

        Returns:
            (message, stats_dict)
            message:    Recovered plaintext string.
            stats_dict: BER, accuracy, bits extracted, etc.
        """
        # Step 1 — Extract bits using sign rule
        recovered_bits = self.extractor.extract(residuals, carrier_indices)
        bits_extracted = len(recovered_bits)

        # Step 2 — Decode length header to know payload bounds
        try:
            header_len  = PayloadEncoder.HEADER_BITS
            payload_len = PayloadEncoder._bits_to_uint32(recovered_bits[:header_len])
            total_needed = header_len + payload_len
        except Exception as e:
            return "", {
                "success":        False,
                "error":          f"Header decode failed: {e}",
                "bits_extracted": bits_extracted,
            }

        # Step 3 — Decrypt using AES-256-GCM
        try:
            all_bits   = recovered_bits[:total_needed]
            raw_bytes  = PayloadEncoder._bits_to_bytes(all_bits[header_len:])

            # The raw payload = IV (12 bytes) + ciphertext
            iv_bytes   = AESCipher.IV_SIZE
            iv         = raw_bytes[:iv_bytes]
            ciphertext = raw_bytes[iv_bytes:]
            payload    = AESPayload(iv=iv, ciphertext=ciphertext)
            plaintext  = self.cipher.decrypt(payload)
            message    = plaintext.decode("utf-8", errors="replace")

            return message, {
                "success":        True,
                "bits_extracted": bits_extracted,
                "payload_bits":   payload_len,
                "message_length": len(message),
            }

        except Exception as e:
            return "", {
                "success":        False,
                "error":          f"Decryption failed: {e}",
                "bits_extracted": bits_extracted,
            }

    # ------------------------------------------------------------------
    # Raw bit extraction (no decryption — for testing / BER measurement)
    # ------------------------------------------------------------------

    def extract_bits_only(
        self,
        residuals: Dict[int, torch.Tensor],
        carrier_indices: Dict[int, List[int]],
    ) -> List[int]:
        """
        Extract raw bits without decryption.
        Used for BER measurement in validation experiments.
        """
        return self.extractor.extract(residuals, carrier_indices)
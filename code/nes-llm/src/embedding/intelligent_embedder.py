"""
IntelligentEmbedder — full NES embedding orchestrator.

Pipeline:
    message
        → PayloadEncoder      (text → bits with length header)
        → AESCipher           (bits → IV + ciphertext)
        → length header       (32-bit prefix so DecryptPipeline knows size)
        → QACIPipeline        (QACI carrier selection)
        → SignEmbeddingStrategy (sign-based embedding)
"""

import secrets
from typing import Dict, List, Optional

import torch

from src.crypto.aes_cipher                  import AESCipher
from src.carrier_intelligence.qaci_pipeline import QACIPipeline
from src.embedding.strategy_registry        import build as build_strategy
from src.embedding.payload_encoder          import PayloadEncoder
from src.core.types                         import EmbeddingConfig


class IntelligentEmbedder:
    """
    Full NES embedding pipeline.

    Usage:
        embedder     = IntelligentEmbedder(config)
        embed_result = embedder.embed("secret message", residuals)

        embed_result.embedded_residuals  → {layer_id: tensor}
        embed_result.carrier_indices     → {layer_id: [indices]}
        embed_result.key                 → bytes  (AES-256)
        embed_result.key_id              → str
    """

    def __init__(self, config: EmbeddingConfig):
        self.config = config
        self.pipeline = QACIPipeline(
            total_layers=config.num_hidden_layers,
            gamma=config.gamma,
        )

        # Honour EmbeddingConfig.embedding_strategy. It used to be
        # accepted and ignored: the constructor hardcoded
        # SignEmbeddingStrategy, so no experiment could select a
        # different scheme. Resolved through the registry so every
        # strategy shares one contract.
        self.strategy_name = (
            config.embedding_strategy or "sign"
        )
        self.strategy = build_strategy(config, self.strategy_name)

    def embed(
        self,
        message:           str,
        residuals:         Dict[int, torch.Tensor],
        fp16_weights:      Optional[Dict[int, torch.Tensor]] = None,
        quantized_weights: Optional[Dict[int, torch.Tensor]] = None,
    ) -> "EmbedResult":

        # Step 1 — Raw UTF-8 bytes (no inner header)
        message_bytes = message.encode("utf-8")

        # Step 2 — AES-256-GCM encrypt
        key    = AESCipher.generate_key()
        key_id = secrets.token_hex(8)
        cipher = AESCipher(key, key_id=key_id)
        encrypted_bits = cipher.encrypt_to_bits(message_bytes)

        # Step 3 — Prepend 32-bit outer length header
        length_header = PayloadEncoder._uint32_to_bits(len(encrypted_bits))
        all_bits      = length_header + encrypted_bits
        total_bits    = len(all_bits)

        # Step 4 — QACI carrier selection
        selection = self.pipeline.select(
            residuals=         residuals,
            total_payload_bits=total_bits,
            fp16_weights=      fp16_weights,
            quantized_weights= quantized_weights,
        )

        # Step 5 — Sign-based embedding
        embed_result = self.strategy.embed(
            residuals=       residuals,
            bits=            all_bits,
            selector_indices=selection.selected_indices,
        )

        return EmbedResult(
            embedded_residuals=embed_result.embedded_weights,
            carrier_indices=   embed_result.carrier_indices,
            layer_allocation=  selection.layer_allocation,
            key=               key,
            key_id=            key_id,
            total_bits=        total_bits,
            bits_embedded=     embed_result.bits_embedded,
            success=           embed_result.success,
            embedded_bits=     all_bits,             # ADD THIS
            strategy=          self.strategy_name,
        )


class EmbedResult:
    def __init__(
        self,
        embedded_residuals: Dict[int, torch.Tensor],
        carrier_indices:    Dict[int, List[int]],
        layer_allocation:   Dict[int, int],
        key:                bytes,
        key_id:             str,
        total_bits:         int,
        bits_embedded:      int,
        success:            bool,
        embedded_bits:      List[int] = None,   # ADD THIS
        strategy:           str = None,
    ):
        self.embedded_residuals = embedded_residuals
        self.carrier_indices    = carrier_indices
        self.layer_allocation   = layer_allocation
        self.key                = key
        self.key_id             = key_id
        self.total_bits         = total_bits
        self.bits_embedded      = bits_embedded
        self.success            = success
        self.embedded_bits      = embedded_bits or []   # ADD THIS
        self.strategy           = strategy

    def summary(self) -> dict:
        return {
            "success":       self.success,
            "key_id":        self.key_id,
            "total_bits":    self.total_bits,
            "bits_embedded": self.bits_embedded,
            "layers_used":   sum(1 for b in self.layer_allocation.values() if b > 0),
            "per_layer":     self.layer_allocation,
        }
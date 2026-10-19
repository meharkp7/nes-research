"""
Experiment 9 — GPTQ and AWQ Quantization Variants

Goal:
    Test whether NES can operate with GPTQ and AWQ quantized models.

Expected:
    BER = 0 on supported GPTQ and AWQ models.
"""

import os

# Enable CPU fallback for unsupported MPS operations.
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"


from src.model.model_loader import (
    load_model_pair,
    extract_residuals
)

from src.embedding.intelligent_embedder import IntelligentEmbedder
from src.core.types import EmbeddingConfig


# ============================================================
# MODELS
# ============================================================

MODELS = [
    # GPTQ
    (
        "TheBloke/Mistral-7B-v0.1-GPTQ",
        "mistral",
        32,
        "GPTQ"
    ),

    # AWQ
    (
        "TheBloke/Mistral-7B-v0.1-AWQ",
        "mistral",
        32,
        "AWQ"
    ),
]


# ============================================================
# RUN EXPERIMENT
# ============================================================

for model_id, family, n_layers, quant_type in MODELS:

    print("\n" + "=" * 70)
    print(f"Model       : {model_id}")
    print(f"Family      : {family}")
    print(f"Quantizer   : {quant_type}")
    print(f"Layers      : {n_layers}")
    print("=" * 70)

    try:

        # ----------------------------------------------------
        # Load quantized model + FP16 reference
        # Time Complexity: O(model size)
        # ----------------------------------------------------

        quant_model, fp16_model, tokenizer = load_model_pair(
            model_id
        )

        # ----------------------------------------------------
        # Extract quantization residuals
        # Time Complexity: O(L * W)
        # L = number of layers
        # W = weights per layer
        # ----------------------------------------------------

        residuals = extract_residuals(
            quant_model,
            fp16_model,
            family
        )

        # ----------------------------------------------------
        # Configure embedding
        # ----------------------------------------------------

        config = EmbeddingConfig(
            total_payload_bits=10_000,
            model_family=family,
            num_hidden_layers=n_layers
        )

        # ----------------------------------------------------
        # Embed test payload
        # Time Complexity: O(number of carrier weights)
        # ----------------------------------------------------

        embed_result = IntelligentEmbedder(
            config
        ).embed(
            "A" * 1250,
            residuals
        )

        # ----------------------------------------------------
        # Check embedded carrier values
        # ----------------------------------------------------

        original_bits = embed_result.original_bits
        embedded_bits = embed_result.embedded_bits

        errors = sum(
            a != b
            for a, b in zip(
                original_bits,
                embedded_bits
            )
        )

        total_bits = len(original_bits)

        # ----------------------------------------------------
        # Calculate BER
        # Time Complexity: O(B)
        # B = number of embedded bits
        # ----------------------------------------------------

        if total_bits == 0:
            ber = 1.0
        else:
            ber = errors / total_bits

        print("\nResults")
        print("-" * 50)
        print(f"Quantization : {quant_type}")
        print(f"Total bits   : {total_bits}")
        print(f"Bit errors   : {errors}")
        print(f"BER          : {ber:.6f}")

        # ----------------------------------------------------
        # Gate
        # ----------------------------------------------------

        if ber == 0.0:
            print("Status       : PASS")
        else:
            print("Status       : FAIL")

    except Exception as e:

        print("\nFAILED")
        print(f"Model        : {model_id}")
        print(f"Quantization : {quant_type}")
        print(f"{type(e).__name__}: {e}")
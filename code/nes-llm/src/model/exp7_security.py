"""
Experiment 7 — Security / Adversarial Detection

Goal:
    Verify that NES embedding does not produce a statistically
    detectable signature.

Gates:
    KL divergence < 0.05
    Neural detector accuracy < 55%
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
from src.steganalysis.security_validator import SecurityValidator


# ============================================================
# MODELS
# ============================================================

MODELS = [
    ("meta-llama/Llama-3.1-8B", "llama", 32),
    ("mistralai/Mistral-7B-v0.3", "mistral", 32),
    ("google/gemma-2-9b", "gemma", 42),
    ("Qwen/Qwen2.5-7B", "qwen", 28),
    ("Qwen/Qwen2.5-3B", "qwen", 36),
    ("TinyLlama/TinyLlama-1.1B-Chat-v1.0", "llama", 22),
    ("microsoft/Phi-3-mini-4k-instruct", "phi3", 32),
]


# ============================================================
# RUN EXPERIMENT FOR EACH MODEL
# ============================================================

for model_id, family, n_layers in MODELS:

    print("\n" + "=" * 70)
    print(f"Model: {model_id}")
    print(f"Family: {family}")
    print(f"Layers: {n_layers}")
    print("=" * 70)

    try:

        # ----------------------------------------------------
        # Load NF4 and FP16 models
        # Time Complexity: O(model size)
        # ----------------------------------------------------

        nf4_model, fp16_model, _ = load_model_pair(
            model_id
        )

        # ----------------------------------------------------
        # Extract quantization residuals
        # Time Complexity: O(L * W)
        # L = number of layers
        # W = weights per layer
        # ----------------------------------------------------

        residuals = extract_residuals(
            nf4_model,
            fp16_model,
            family
        )

        # ----------------------------------------------------
        # Configure embedding
        # ----------------------------------------------------

        config = EmbeddingConfig(
            total_payload_bits=50_000,
            model_family=family,
            num_hidden_layers=n_layers
        )

        # ----------------------------------------------------
        # Embed payload
        # Time Complexity: O(number of carrier weights)
        # ----------------------------------------------------

        embed_result = IntelligentEmbedder(
            config
        ).embed(
            "A" * 6000,
            residuals
        )

        # ----------------------------------------------------
        # Security validator
        # ----------------------------------------------------

        validator = SecurityValidator(
            max_kl_divergence=0.05,
            max_detector_accuracy=0.55
        )

        # ----------------------------------------------------
        # Validate security
        # Time Complexity:
        # O(C + D)
        # C = carrier count
        # D = detector evaluation cost
        # ----------------------------------------------------

        result = validator.validate(
            original_residuals=residuals,
            embedded_residuals=(
                embed_result.embedded_residuals
            ),
            carrier_indices=(
                embed_result.carrier_indices
            )
        )

        # ----------------------------------------------------
        # Results
        # ----------------------------------------------------

        print(result.report())

        print(
            f"Gate status: {result.status}"
        )

    except Exception as e:

        print("\nFAILED:", model_id)
        print(
            f"{type(e).__name__}: {e}"
        )
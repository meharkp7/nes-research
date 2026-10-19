import os

# Enable CPU fallback for unsupported MPS operations.
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"

from src.model.model_loader import (
    load_model_pair,
    extract_residuals
)

from src.embedding.intelligent_embedder import IntelligentEmbedder
from src.core.types import EmbeddingConfig
from src.evaluation.robustness_validator import RobustnessValidator


# ============================================================
# MODELS
# Time Complexity: O(number of models)
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
# GAUSSIAN NOISE LEVELS
# ============================================================

SIGMAS = [
    0.0,
    0.0005,
    0.001,
    0.002,
    0.005,
    0.010,
    0.020
]


# ============================================================
# RUN ROBUSTNESS EXPERIMENT
# ============================================================

for model_id, family, n_layers in MODELS:

    print("\n" + "=" * 70)
    print(f"Model: {model_id}")
    print(f"Family: {family}")
    print(f"Expected layers: {n_layers}")
    print("=" * 70)

    try:

        # ----------------------------------------------------
        # Load NF4 + FP16 models
        # Time Complexity: O(model size)
        # ----------------------------------------------------

        nf4, fp16, _ = load_model_pair(model_id)

        # ----------------------------------------------------
        # Extract NF4 quantization residuals
        # Time Complexity: O(L * W)
        # L = number of layers
        # W = down_proj weights per layer
        # ----------------------------------------------------

        residuals = extract_residuals(
            nf4,
            fp16,
            family
        )

        # ----------------------------------------------------
        # Embedding configuration
        # ----------------------------------------------------

        config = EmbeddingConfig(
            total_payload_bits=10_000,
            model_family=family,
            num_hidden_layers=n_layers
        )

        # ----------------------------------------------------
        # Embed test message
        # Time Complexity: O(number of carrier weights)
        # ----------------------------------------------------

        embed_result = IntelligentEmbedder(
            config
        ).embed(
            "A" * 1250,
            residuals
        )

        # ----------------------------------------------------
        # Generate ground-truth encrypted bits
        # ----------------------------------------------------


        # ----------------------------------------------------
        # Robustness validator
        # ----------------------------------------------------

        validator = RobustnessValidator(
            max_ber_at_001=0.02,
            max_ber_at_002=0.10,
            num_trials=5
        )

        # ----------------------------------------------------
        # Test all Gaussian noise levels
        # Time Complexity:
        # O(S * T * C)
        # S = number of sigma values
        # T = number of trials
        # C = number of carrier values
        # ----------------------------------------------------

        result = validator.validate(
            embed_result.embedded_residuals,
            embed_result.carrier_indices,
            embed_result.embedded_bits,
            SIGMAS
        )

        # ----------------------------------------------------
        # Print results
        # ----------------------------------------------------

        print(f"\n{model_id}")
        print(result.report())

    except Exception as e:

        print(f"\nFAILED: {model_id}")
        print(f"{type(e).__name__}: {e}")
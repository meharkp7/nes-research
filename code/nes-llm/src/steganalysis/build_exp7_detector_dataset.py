"""
Exp7 — Build a real neural-detector dataset from the CURRENT NES pipeline.

Each clean/stego pair uses:
    - the same model layer
    - the same tensor region
    - the same patch position

Therefore, the detector cannot exploit different layer/region distributions.
The only intended difference is the NES embedding.
"""

import os
import pickle
import random

os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"

import torch

from src.model.model_loader import load_model_pair, extract_residuals
from src.embedding.intelligent_embedder import IntelligentEmbedder
from src.core.types import EmbeddingConfig


# ============================================================
# Configuration
# ============================================================

MODEL_ID = "Qwen/Qwen2.5-3B"
MODEL_FAMILY = "qwen"
NUM_HIDDEN_LAYERS = 36

NUM_PAIRS = 500
PATCH_SIZE = 4096
PAYLOAD_BITS = 10_000

OUTPUT_PATH = "artifacts/exp7_detector_dataset.pkl"

SEED = 42


# ============================================================
# Helpers
# ============================================================

def get_layer_carrier_indices(carrier_indices, layer_id):
    """
    Handle the current carrier_indices structure.
    """

    if isinstance(carrier_indices, dict):
        return carrier_indices.get(layer_id, [])

    return []


def extract_patch(tensor, patch_start, patch_size):
    """
    Extract one fixed-size patch from a tensor.
    """

    flat = tensor.detach().cpu().float().flatten()

    patch_end = patch_start + patch_size

    if patch_end > flat.numel():
        return None

    return flat[patch_start:patch_end].clone()


# ============================================================
# Main
# ============================================================

def main():

    random.seed(SEED)
    torch.manual_seed(SEED)

    print("=" * 70)
    print("EXP7 — REAL PAIRED NEURAL DETECTOR DATASET")
    print("=" * 70)

    print(f"Model:        {MODEL_ID}")
    print(f"Pairs:        {NUM_PAIRS}")
    print(f"Patch size:   {PATCH_SIZE}")
    print(f"Payload bits: {PAYLOAD_BITS}")
    print()

    # --------------------------------------------------------
    # Load actual model pair
    # --------------------------------------------------------

    print("Loading model pair...")

    nf4_model, fp16_model, _ = load_model_pair(MODEL_ID)

    print("Extracting real residuals...")

    residuals = extract_residuals(
        nf4_model,
        fp16_model,
        MODEL_FAMILY,
    )

    print(f"Residual layers: {len(residuals)}")
    print()

    # --------------------------------------------------------
    # Dataset
    # --------------------------------------------------------

    dataset = []

    successful_pairs = 0

    # --------------------------------------------------------
    # Generate paired clean/stego examples
    # --------------------------------------------------------

    for sample_id in range(NUM_PAIRS):

        if sample_id % 25 == 0:
            print(
                f"Generating pair "
                f"{sample_id}/{NUM_PAIRS}"
            )

        # ----------------------------------------------------
        # Current production embedding pipeline
        # ----------------------------------------------------

        config = EmbeddingConfig(
            total_payload_bits=PAYLOAD_BITS,
            model_family=MODEL_FAMILY,
            num_hidden_layers=NUM_HIDDEN_LAYERS,
        )

        embedder = IntelligentEmbedder(config)

        message = (
            f"EXP7_SAMPLE_{sample_id}_"
            + "A" * 1200
        )

        embed_result = embedder.embed(
            message,
            residuals,
        )

        # ----------------------------------------------------
        # Find layers that actually received carriers
        # ----------------------------------------------------

        candidate_layers = []

        for layer_id in residuals:

            indices = get_layer_carrier_indices(
                embed_result.carrier_indices,
                layer_id,
            )

            if indices:
                candidate_layers.append(layer_id)

        if not candidate_layers:
            continue

        # ----------------------------------------------------
        # Choose a layer that actually contains embedding
        # ----------------------------------------------------

        layer_id = random.choice(candidate_layers)

        carrier_indices = get_layer_carrier_indices(
            embed_result.carrier_indices,
            layer_id,
        )

        if not carrier_indices:
            continue

        # ----------------------------------------------------
        # Choose a carrier position.
        #
        # The resulting patch is guaranteed to contain
        # at least one embedded position.
        # ----------------------------------------------------

        carrier_position = random.choice(carrier_indices)

        patch_start = (
            carrier_position // PATCH_SIZE
        ) * PATCH_SIZE

        clean_patch = extract_patch(
            residuals[layer_id],
            patch_start,
            PATCH_SIZE,
        )

        stego_patch = extract_patch(
            embed_result.embedded_residuals[layer_id],
            patch_start,
            PATCH_SIZE,
        )

        if clean_patch is None or stego_patch is None:
            continue

        # ----------------------------------------------------
        # Sanity check:
        # clean and stego must have identical shape
        # ----------------------------------------------------

        if clean_patch.shape != stego_patch.shape:
            continue

        # ----------------------------------------------------
        # Store paired samples.
        #
        # label 0 = clean
        # label 1 = stego
        # ----------------------------------------------------

        dataset.append(
            {
                "features": clean_patch,
                "label": 0,
                "layer_id": layer_id,
                "sample_id": sample_id,
                "patch_start": patch_start,
            }
        )

        dataset.append(
            {
                "features": stego_patch,
                "label": 1,
                "layer_id": layer_id,
                "sample_id": sample_id,
                "patch_start": patch_start,
            }
        )

        successful_pairs += 1

    # --------------------------------------------------------
    # Shuffle
    # --------------------------------------------------------

    random.shuffle(dataset)

    print()
    print("=" * 70)
    print("DATASET SUMMARY")
    print("=" * 70)

    print(f"Requested pairs:  {NUM_PAIRS}")
    print(f"Successful pairs: {successful_pairs}")
    print(f"Clean samples:    {successful_pairs}")
    print(f"Stego samples:    {successful_pairs}")
    print(f"Total samples:    {len(dataset)}")

    # --------------------------------------------------------
    # Verify pairing
    # --------------------------------------------------------

    labels = [sample["label"] for sample in dataset]

    clean_count = labels.count(0)
    stego_count = labels.count(1)

    print()
    print(f"Label 0 (clean): {clean_count}")
    print(f"Label 1 (stego): {stego_count}")

    if clean_count != stego_count:
        raise RuntimeError(
            "Dataset is not balanced."
        )

    if len(dataset) == 0:
        raise RuntimeError(
            "No detector samples were generated."
        )

    # --------------------------------------------------------
    # Save
    # --------------------------------------------------------

    os.makedirs(
        os.path.dirname(OUTPUT_PATH),
        exist_ok=True,
    )

    with open(OUTPUT_PATH, "wb") as f:
        pickle.dump(dataset, f)

    print()
    print(f"Saved dataset → {OUTPUT_PATH}")
    print("=" * 70)


if __name__ == "__main__":
    main()
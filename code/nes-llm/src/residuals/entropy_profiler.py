from transformers import AutoModelForCausalLM
from transformers import BitsAndBytesConfig
from bitsandbytes.functional import dequantize_4bit

from scipy.stats import entropy
from scipy.stats import skew
from scipy.stats import kurtosis

import torch
import json
import numpy as np


MODEL_ID = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"


def residual_metrics(residual):

    residual = residual.cpu().numpy().flatten()

    hist, _ = np.histogram(
        residual,
        bins=256,
        density=True
    )

    hist = hist + 1e-12

    return {
        "entropy": float(entropy(hist)),
        "skewness": float(skew(residual)),
        "kurtosis": float(kurtosis(residual)),
        "std": float(np.std(residual))
    }


def get_residual(fp16_module, nf4_module):

    fp16_weight = (
        fp16_module.weight
        .detach()
        .float()
        .cpu()
    )

    nf4_weight = (
        dequantize_4bit(
            nf4_module.weight.data,
            quant_state=nf4_module.weight.quant_state
        )
        .detach()
        .float()
        .cpu()
    )

    return fp16_weight - nf4_weight


def main():

    print("Loading FP16 model...")

    fp16_model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        dtype=torch.float16
    ).cpu()

    print("Loading NF4 model...")

    quant_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.float16,
    )

    nf4_model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        quantization_config=quant_config,
        device_map="auto"
    )

    results = {}

    num_layers = len(fp16_model.model.layers)

    for layer_idx in range(num_layers):

        print(f"Layer {layer_idx}")

        fp16_layer = fp16_model.model.layers[layer_idx]
        nf4_layer = nf4_model.model.layers[layer_idx]

        families = {
            "q_proj": (
                fp16_layer.self_attn.q_proj,
                nf4_layer.self_attn.q_proj
            ),
            "k_proj": (
                fp16_layer.self_attn.k_proj,
                nf4_layer.self_attn.k_proj
            ),
            "v_proj": (
                fp16_layer.self_attn.v_proj,
                nf4_layer.self_attn.v_proj
            ),
            "o_proj": (
                fp16_layer.self_attn.o_proj,
                nf4_layer.self_attn.o_proj
            ),
            "gate_proj": (
                fp16_layer.mlp.gate_proj,
                nf4_layer.mlp.gate_proj
            ),
            "up_proj": (
                fp16_layer.mlp.up_proj,
                nf4_layer.mlp.up_proj
            ),
            "down_proj": (
                fp16_layer.mlp.down_proj,
                nf4_layer.mlp.down_proj
            ),
        }

        layer_results = {}

        for name, (fp16_mod, nf4_mod) in families.items():

            residual = get_residual(
                fp16_mod,
                nf4_mod
            )

            layer_results[name] = residual_metrics(
                residual
            )

        results[f"layer_{layer_idx}"] = layer_results

    with open(
        "residual_entropy_profile.json",
        "w"
    ) as f:

        json.dump(
            results,
            f,
            indent=4
        )

    print("\nSaved residual_entropy_profile.json")


if __name__ == "__main__":
    main()
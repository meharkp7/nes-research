from transformers import AutoModelForCausalLM
from transformers import BitsAndBytesConfig
from bitsandbytes.functional import dequantize_4bit

import torch
import json


MODEL_ID = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"


def residual_stats(residual):

    return {
        "mean": residual.mean().item(),
        "std": residual.std().item(),
        "variance": residual.var().item(),
        "min": residual.min().item(),
        "max": residual.max().item(),
        "l2_norm": torch.norm(residual).item(),
    }


def build_residual(fp16_layer, nf4_layer):

    fp16_weight = (
        fp16_layer.weight
        .detach()
        .float()
        .cpu()
    )

    nf4_weight = (
        dequantize_4bit(
            nf4_layer.weight.data,
            quant_state=nf4_layer.weight.quant_state,
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
        dtype=torch.float16,
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
        device_map="auto",
    )

    results = {}

    num_layers = len(fp16_model.model.layers)

    print(f"\nTotal Layers: {num_layers}")

    for layer_idx in range(num_layers):

        print(f"\nProcessing Layer {layer_idx}")

        fp16_layer = fp16_model.model.layers[layer_idx]
        nf4_layer = nf4_model.model.layers[layer_idx]

        layer_results = {}

        families = {
            "q_proj": (
                fp16_layer.self_attn.q_proj,
                nf4_layer.self_attn.q_proj,
            ),
            "k_proj": (
                fp16_layer.self_attn.k_proj,
                nf4_layer.self_attn.k_proj,
            ),
            "v_proj": (
                fp16_layer.self_attn.v_proj,
                nf4_layer.self_attn.v_proj,
            ),
            "o_proj": (
                fp16_layer.self_attn.o_proj,
                nf4_layer.self_attn.o_proj,
            ),
            "gate_proj": (
                fp16_layer.mlp.gate_proj,
                nf4_layer.mlp.gate_proj,
            ),
            "up_proj": (
                fp16_layer.mlp.up_proj,
                nf4_layer.mlp.up_proj,
            ),
            "down_proj": (
                fp16_layer.mlp.down_proj,
                nf4_layer.mlp.down_proj,
            ),
        }

        for family, (fp16_mod, nf4_mod) in families.items():

            residual = build_residual(
                fp16_mod,
                nf4_mod
            )

            layer_results[family] = residual_stats(
                residual
            )

        results[f"layer_{layer_idx}"] = layer_results

    with open(
        "layer_residual_profile.json",
        "w"
    ) as f:

        json.dump(
            results,
            f,
            indent=4
        )

    print("\nSaved layer_residual_profile.json")


if __name__ == "__main__":
    main()
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

    layer = 0

    families = {
        "q_proj": (
            fp16_model.model.layers[layer].self_attn.q_proj,
            nf4_model.model.layers[layer].self_attn.q_proj,
        ),
        "k_proj": (
            fp16_model.model.layers[layer].self_attn.k_proj,
            nf4_model.model.layers[layer].self_attn.k_proj,
        ),
        "v_proj": (
            fp16_model.model.layers[layer].self_attn.v_proj,
            nf4_model.model.layers[layer].self_attn.v_proj,
        ),
        "o_proj": (
            fp16_model.model.layers[layer].self_attn.o_proj,
            nf4_model.model.layers[layer].self_attn.o_proj,
        ),
        "gate_proj": (
            fp16_model.model.layers[layer].mlp.gate_proj,
            nf4_model.model.layers[layer].mlp.gate_proj,
        ),
        "up_proj": (
            fp16_model.model.layers[layer].mlp.up_proj,
            nf4_model.model.layers[layer].mlp.up_proj,
        ),
        "down_proj": (
            fp16_model.model.layers[layer].mlp.down_proj,
            nf4_model.model.layers[layer].mlp.down_proj,
        ),
    }

    results = {}

    for family, (fp16_layer, nf4_layer) in families.items():

        print(f"\nProcessing {family}")

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

        residual = fp16_weight - nf4_weight

        stats = residual_stats(residual)

        results[family] = stats

        print(stats)

    with open(
        "family_residual_profile.json",
        "w"
    ) as f:

        json.dump(
            results,
            f,
            indent=4
        )

    print("\nSaved family_residual_profile.json")


if __name__ == "__main__":
    main()
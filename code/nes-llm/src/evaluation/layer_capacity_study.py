import json
import torch

from transformers import (
    AutoModelForCausalLM,
    BitsAndBytesConfig,
)

from bitsandbytes.functional import (
    dequantize_4bit
)

MODEL_ID = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"


def build_residual(fp16_layer, nf4_layer):

    fp16_weight = (
        fp16_layer.weight
        .data
        .float()
        .cpu()
    )

    nf4_weight = (
        dequantize_4bit(
            nf4_layer.weight.data,
            quant_state=nf4_layer.weight.quant_state,
        )
        .float()
        .cpu()
    )

    return fp16_weight - nf4_weight


def distortion(original, modified):

    diff = modified - original

    return {
        "mean_shift":
            abs(
                original.mean().item()
                -
                modified.mean().item()
            ),

        "std_shift":
            abs(
                original.std().item()
                -
                modified.std().item()
            ),

        "l2":
            torch.norm(diff).item(),
    }


def fake_embed(
    residual,
    payload_bits,
):

    residual = residual.clone()

    flat = residual.flatten()

    payload_bits = min(
        payload_bits,
        len(flat)
    )

    positions = torch.randperm(
        len(flat)
    )[:payload_bits]

    bits = torch.randint(
        0,
        2,
        (payload_bits,)
    )

    for pos, bit in zip(
        positions,
        bits
    ):

        value = max(
            abs(flat[pos].item()),
            0.001
        )

        if bit == 1:
            flat[pos] = value
        else:
            flat[pos] = -value

    return residual


def main():

    payload_sizes = [
        100,
        1000,
        10000,
        50000,
        100000,
        250000,
        500000,
    ]

    print("Loading FP16...")

    fp16 = (
        AutoModelForCausalLM
        .from_pretrained(
            MODEL_ID,
            torch_dtype=torch.float16,
            device_map="cpu",
        )
    )

    print("Loading NF4...")

    quant_config = (
        BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
        )
    )

    nf4 = (
        AutoModelForCausalLM
        .from_pretrained(
            MODEL_ID,
            quantization_config=quant_config,
            device_map="cpu",
        )
    )

    layer_fp16 = fp16.model.layers[0]
    layer_nf4 = nf4.model.layers[0]

    families = {

        "q_proj": (
            layer_fp16.self_attn.q_proj,
            layer_nf4.self_attn.q_proj
        ),

        "k_proj": (
            layer_fp16.self_attn.k_proj,
            layer_nf4.self_attn.k_proj
        ),

        "v_proj": (
            layer_fp16.self_attn.v_proj,
            layer_nf4.self_attn.v_proj
        ),

        "o_proj": (
            layer_fp16.self_attn.o_proj,
            layer_nf4.self_attn.o_proj
        ),

        "gate_proj": (
            layer_fp16.mlp.gate_proj,
            layer_nf4.mlp.gate_proj
        ),

        "up_proj": (
            layer_fp16.mlp.up_proj,
            layer_nf4.mlp.up_proj
        ),

        "down_proj": (
            layer_fp16.mlp.down_proj,
            layer_nf4.mlp.down_proj
        ),
    }

    results = {}

    for name, (
        fp16_mod,
        nf4_mod
    ) in families.items():

        print(
            f"\n{'=' * 70}"
        )

        print(
            f"FAMILY: {name}"
        )

        print(
            f"{'=' * 70}"
        )

        residual = build_residual(
            fp16_mod,
            nf4_mod
        )

        results[name] = {}

        for payload_size in payload_sizes:

            print(
                f"\nPayload: {payload_size}"
            )

            embedded = fake_embed(
                residual,
                payload_size
            )

            stats = distortion(
                residual,
                embedded
            )

            results[name][
                str(payload_size)
            ] = stats

            print(stats)

    with open(
        "layer_capacity_results.json",
        "w"
    ) as f:

        json.dump(
            results,
            f,
            indent=4
        )

    print(
        "\nSaved layer_capacity_results.json"
    )


if __name__ == "__main__":
    main()
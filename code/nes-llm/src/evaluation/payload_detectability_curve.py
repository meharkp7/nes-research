import json
import torch
import numpy as np

from scipy.stats import entropy

from transformers import (
    AutoModelForCausalLM,
    BitsAndBytesConfig,
)

from bitsandbytes.functional import (
    dequantize_4bit
)

from src.embedding.keyed_residual_embedder import (
    KeyedResidualEmbedder
)

MODEL_ID = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"


def build_residual(fp16_layer, nf4_layer):

    fp16_weight = (
        fp16_layer.weight.data
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


def compute_entropy(tensor):

    values = tensor.flatten().numpy()

    hist, _ = np.histogram(
        values,
        bins=256,
        density=True,
    )

    hist += 1e-10

    return entropy(hist)


def kl_divergence(a, b):

    p, _ = np.histogram(
        a.flatten().numpy(),
        bins=256,
        density=True,
    )

    q, _ = np.histogram(
        b.flatten().numpy(),
        bins=256,
        density=True,
    )

    p += 1e-10
    q += 1e-10

    return entropy(p, q)


def sign_ratio(tensor):

    flat = tensor.flatten()

    return (
        (flat > 0).sum().item()
        / len(flat)
    )


def analyze(
    original,
    embedded,
):

    return {

        "kl":
            kl_divergence(
                original,
                embedded
            ),

        "entropy_shift":
            abs(
                compute_entropy(
                    original
                )
                -
                compute_entropy(
                    embedded
                )
            ),

        "sign_shift":
            abs(
                sign_ratio(
                    original
                )
                -
                sign_ratio(
                    embedded
                )
            )
    }


def main():

    payloads = [

        1000,
        10000,
        50000,
        100000,
        250000,
        500000,
    ]

    print(
        "Loading FP16..."
    )

    fp16 = (
        AutoModelForCausalLM
        .from_pretrained(
            MODEL_ID,
            torch_dtype=torch.float16,
            device_map="cpu",
        )
    )

    print(
        "Loading NF4..."
    )

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

        "down_proj": (
            layer_fp16.mlp.down_proj,
            layer_nf4.mlp.down_proj
        ),

        "gate_proj": (
            layer_fp16.mlp.gate_proj,
            layer_nf4.mlp.gate_proj
        ),

        "up_proj": (
            layer_fp16.mlp.up_proj,
            layer_nf4.mlp.up_proj
        ),
    }

    results = {}

    for family, (
        fp16_mod,
        nf4_mod
    ) in families.items():

        print(
            f"\n===== {family} ====="
        )

        residual = build_residual(
            fp16_mod,
            nf4_mod
        )

        results[family] = {}

        for payload in payloads:

            print(
                f"Payload: {payload}"
            )

            bits = torch.randint(
                0,
                2,
                (payload,)
            ).tolist()

            embedded = (
                KeyedResidualEmbedder
                .embed_bits(
                    residual.clone(),
                    bits,
                    "nes_secret"
                )
            )

            stats = analyze(
                residual,
                embedded
            )

            print(stats)

            results[
                family
            ][
                str(payload)
            ] = stats

    with open(
        "payload_detectability_curve.json",
        "w"
    ) as f:

        json.dump(
            results,
            f,
            indent=4
        )

    print(
        "\nSaved payload_detectability_curve.json"
    )


if __name__ == "__main__":
    main()
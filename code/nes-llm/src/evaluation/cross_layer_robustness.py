import json
import torch

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


def bit_error_rate(a, b):

    errors = sum(
        x != y
        for x, y in zip(a, b)
    )

    return errors / len(a)


def add_noise(
    tensor,
    sigma,
):

    noise = (
        torch.randn_like(tensor)
        * sigma
    )

    return tensor + noise


def main():

    print("Loading FP16...")

    fp16 = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        torch_dtype=torch.float16,
        device_map="cpu",
    )

    print("Loading NF4...")

    quant_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
    )

    nf4 = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        quantization_config=quant_config,
        device_map="cpu",
    )

    layer_fp16 = fp16.model.layers[0]
    layer_nf4 = nf4.model.layers[0]

    families = {

        "down_proj": (
            layer_fp16.mlp.down_proj,
            layer_nf4.mlp.down_proj,
        ),

        "gate_proj": (
            layer_fp16.mlp.gate_proj,
            layer_nf4.mlp.gate_proj,
        ),

        "up_proj": (
            layer_fp16.mlp.up_proj,
            layer_nf4.mlp.up_proj,
        ),
    }

    payload = 100000

    sigma = 1e-2

    results = {}

    for name, (
        fp16_layer,
        nf4_layer,
    ) in families.items():

        print(
            f"\n===== {name} ====="
        )

        residual = build_residual(
            fp16_layer,
            nf4_layer,
        )

        bits = torch.randint(
            0,
            2,
            (payload,),
        ).tolist()

        stego = (
            KeyedResidualEmbedder.embed_bits(
                residual.clone(),
                bits,
                "nes_secret",
            )
        )

        noisy = add_noise(
            stego,
            sigma,
        )

        recovered = (
            KeyedResidualEmbedder.extract_bits(
                noisy,
                "nes_secret",
                payload,
            )
        )

        ber = bit_error_rate(
            bits,
            recovered,
        )

        accuracy = 1 - ber

        results[name] = {

            "sigma": sigma,

            "ber": ber,

            "accuracy": accuracy,
        }

        print(
            results[name]
        )

    with open(
        "cross_layer_robustness.json",
        "w",
    ) as f:

        json.dump(
            results,
            f,
            indent=4,
        )

    print(
        "\nSaved cross_layer_robustness.json"
    )


if __name__ == "__main__":
    main()
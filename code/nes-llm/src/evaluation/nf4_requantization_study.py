import json
import torch

from transformers import (
    AutoModelForCausalLM,
    BitsAndBytesConfig,
)

from bitsandbytes.functional import (
    quantize_4bit,
    dequantize_4bit,
)

from src.embedding.keyed_residual_embedder import (
    KeyedResidualEmbedder,
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

    layer_fp16 = (
        fp16.model.layers[0]
        .mlp.down_proj
    )

    layer_nf4 = (
        nf4.model.layers[0]
        .mlp.down_proj
    )

    residual = build_residual(
        layer_fp16,
        layer_nf4,
    )

    payloads = [
        1000,
        10000,
        50000,
        100000,
    ]

    results = {}

    for payload in payloads:

        print(f"\nPayload: {payload}")

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

        print("NF4 Quantizing...")

        qweight, qstate = quantize_4bit(
            stego,
            quant_type="nf4",
        )

        requantized = dequantize_4bit(
            qweight,
            quant_state=qstate,
        )

        recovered = (
            KeyedResidualEmbedder.extract_bits(
                requantized,
                "nes_secret",
                payload,
            )
        )

        ber = bit_error_rate(
            bits,
            recovered,
        )

        acc = 1 - ber

        results[payload] = {
            "ber": ber,
            "accuracy": acc,
        }

        print(results[payload])

    with open(
        "nf4_requantization_results.json",
        "w",
    ) as f:

        json.dump(
            results,
            f,
            indent=4,
        )

    print(
        "\nSaved nf4_requantization_results.json"
    )


if __name__ == "__main__":
    main()
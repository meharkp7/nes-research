import json
import torch

from transformers import (
    AutoModelForCausalLM,
    BitsAndBytesConfig,
)

from bitsandbytes.functional import (
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


def fake_quantize(tensor, levels):

    mn = tensor.min()
    mx = tensor.max()

    normalized = (
        tensor - mn
    ) / (
        mx - mn + 1e-12
    )

    quantized = torch.round(
        normalized * (levels - 1)
    )

    dequantized = (
        quantized / (levels - 1)
    )

    return (
        dequantized * (mx - mn)
        + mn
    )


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

    fp16_layer = (
        fp16.model.layers[0]
        .mlp.down_proj
    )

    nf4_layer = (
        nf4.model.layers[0]
        .mlp.down_proj
    )

    residual = build_residual(
        fp16_layer,
        nf4_layer,
    )

    payload = 100000

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

    levels_list = [
        256,
        128,
        64,
        32,
        16,
        8,
        4,
    ]

    results = {}

    for levels in levels_list:

        requantized = fake_quantize(
            stego,
            levels,
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

        results[str(levels)] = {
            "ber": ber,
            "accuracy": 1 - ber,
        }

        print(
            levels,
            results[str(levels)],
        )

    with open(
        "quantization_robustness.json",
        "w",
    ) as f:

        json.dump(
            results,
            f,
            indent=4,
        )

    print(
        "\nSaved quantization_robustness.json"
    )


if __name__ == "__main__":
    main()
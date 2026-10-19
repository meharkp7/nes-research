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


def nf4_quantize(x):

    qweight, qstate = quantize_4bit(
        x,
        quant_type="nf4",
    )

    return dequantize_4bit(
        qweight,
        quant_state=qstate,
    )


def int8_quantize(x):

    scale = (
        x.abs().max() / 127
    )

    q = torch.round(
        x / scale
    ).clamp(
        -127,
        127,
    )

    return q * scale


def int4_quantize(x):

    scale = (
        x.abs().max() / 7
    )

    q = torch.round(
        x / scale
    ).clamp(
        -7,
        7,
    )

    return q * scale


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

    print("Building residual...")

    residual = build_residual(
        fp16_layer,
        nf4_layer,
    )

    payload = 5000000

    print(
        f"Payload: {payload}"
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

    quantizers = {

        "nf4": nf4_quantize,

        "int8": int8_quantize,

        "int4": int4_quantize,
    }

    results = {}

    for name, quantizer in quantizers.items():

        print(
            f"\n===== {name.upper()} ====="
        )

        processed = quantizer(
            stego.clone()
        )

        recovered = (
            KeyedResidualEmbedder.extract_bits(
                processed,
                "nes_secret",
                payload,
            )
        )

        ber = bit_error_rate(
            bits,
            recovered,
        )

        acc = 1 - ber

        results[name] = {

            "ber": ber,

            "accuracy": acc,
        }

        print(
            results[name]
        )

    with open(
        "multi_quantizer_results.json",
        "w",
    ) as f:

        json.dump(
            results,
            f,
            indent=4,
        )

    print(
        "\nSaved multi_quantizer_results.json"
    )


if __name__ == "__main__":
    main()
import pickle
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


def main():

    print("Loading FP16 model...")

    fp16 = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        torch_dtype=torch.float16,
        device_map="cpu",
    )

    print("Loading NF4 model...")

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

    families = [

        (
            layer_fp16.mlp.down_proj,
            layer_nf4.mlp.down_proj,
        ),

        (
            layer_fp16.mlp.gate_proj,
            layer_nf4.mlp.gate_proj,
        ),

        (
            layer_fp16.mlp.up_proj,
            layer_nf4.mlp.up_proj,
        ),
    ]

    dataset = []

    for i in range(500):

        fp16_mod, nf4_mod = families[
            i % len(families)
        ]

        residual = build_residual(
            fp16_mod,
            nf4_mod,
        )

        PATCH_SIZE = 4096
        NUM_PATCHES = 10

        bits = torch.randint(
            0,
            2,
            (100000,),
        ).tolist()

        stego = KeyedResidualEmbedder.embed_bits(
            residual.clone(),
            bits,
            "nes_secret",
        )

        clean_flat = residual.flatten()
        stego_flat = stego.flatten()

        for _ in range(NUM_PATCHES):

            start = torch.randint(
                0,
                len(clean_flat) - PATCH_SIZE,
                (1,)
            ).item()

            clean_patch = clean_flat[
                start:start + PATCH_SIZE
            ]

            stego_patch = stego_flat[
                start:start + PATCH_SIZE
            ]

            dataset.append(
                (
                    clean_patch.clone(),
                    0,
                )
            )

            dataset.append(
                (
                    stego_patch.clone(),
                    1,
                )
            )

        if i % 50 == 0:
            print(
                f"Generated {i}"
            )

    with open(
        "real_detector_dataset.pkl",
        "wb",
    ) as f:

        pickle.dump(
            dataset,
            f,
        )

    print(
        "\nSaved real_detector_dataset.pkl"
    )


if __name__ == "__main__":
    main()
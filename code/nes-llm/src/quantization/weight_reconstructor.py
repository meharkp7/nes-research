from transformers import AutoModelForCausalLM
from transformers import BitsAndBytesConfig

from bitsandbytes.functional import dequantize_4bit

import torch


MODEL_ID = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"


def main():

    quant_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.float16,
    )

    print("Loading model...")

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        quantization_config=quant_config,
        device_map="auto",
    )

    target = model.model.layers[0].self_attn.q_proj

    print("\nStored Shape:")
    print(target.weight.shape)

    print("\nOriginal Shape From QuantState:")
    print(target.weight.quant_state.shape)

    reconstructed = dequantize_4bit(
        target.weight.data,
        quant_state=target.weight.quant_state,
    )

    print("\nReconstructed Shape:")
    print(reconstructed.shape)

    print("\nDtype:")
    print(reconstructed.dtype)

    print("\nStatistics")

    print(
        "Mean:",
        reconstructed.float().mean().item()
    )

    print(
        "Std:",
        reconstructed.float().std().item()
    )

    print(
        "Min:",
        reconstructed.float().min().item()
    )

    print(
        "Max:",
        reconstructed.float().max().item()
    )


if __name__ == "__main__":
    main()
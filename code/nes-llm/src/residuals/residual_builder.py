from transformers import AutoModelForCausalLM
from transformers import BitsAndBytesConfig
from bitsandbytes.functional import dequantize_4bit

import torch


MODEL_ID = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"


def main():

    print("Loading FP16 model...")

    fp16_model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        torch_dtype=torch.float16,
        device_map="cpu",
    )

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

    fp16_layer = fp16_model.model.layers[0].self_attn.q_proj

    nf4_layer = nf4_model.model.layers[0].self_attn.q_proj

    fp16_weight = (
        fp16_layer.weight.detach()
        .float()
        .to("cpu")
    )

    nf4_weight = (
        dequantize_4bit(
            nf4_layer.weight.data,
            quant_state=nf4_layer.weight.quant_state,
        )
        .detach()
        .float()
        .to("cpu")
    )

    print("\nFP16 DEVICE:")
    print(fp16_weight.device)

    print("\nNF4 DEVICE:")
    print(nf4_weight.device)

    residual = fp16_weight - nf4_weight

    print("\nResidual Statistics")

    print(
        "Mean:",
        residual.mean().item()
    )

    print(
        "Std:",
        residual.std().item()
    )

    print(
        "Min:",
        residual.min().item()
    )

    print(
        "Max:",
        residual.max().item()
    )

    print(
        "L2 Norm:",
        torch.norm(residual).item()
    )


if __name__ == "__main__":
    main()
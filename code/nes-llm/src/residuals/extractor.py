from typing import Dict

import torch
from transformers import AutoModelForCausalLM


TARGET_SUFFIXES = (
    "self_attn.q_proj",
    "self_attn.k_proj",
    "self_attn.v_proj",
    "self_attn.o_proj",
    "mlp.up_proj",
    "mlp.down_proj",
    "mlp.gate_proj",
)


class TensorExtractor:
    def __init__(self, model):
        self.model = model

    def extract_target_tensors(self) -> Dict[str, torch.Tensor]:
        tensors = {}

        for name, module in self.model.named_modules():
            if any(name.endswith(suffix) for suffix in TARGET_SUFFIXES):
                if hasattr(module, "weight"):
                    tensors[name] = module.weight.detach().cpu()

        return tensors

    def get_tensor_summary(self):
        summary = []

        tensors = self.extract_target_tensors()

        for name, tensor in tensors.items():
            summary.append(
                {
                    "name": name,
                    "shape": tuple(tensor.shape),
                    "parameters": tensor.numel(),
                    "dtype": str(tensor.dtype),
                }
            )

        return summary


def load_tinyllama():
    model = AutoModelForCausalLM.from_pretrained(
        "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
        torch_dtype=torch.float16,
        low_cpu_mem_usage=True,
        device_map="cpu",
    )

    return model


def print_tensor_report(summary):
    print("\n" + "=" * 120)
    print("TARGET TENSOR REPORT")
    print("=" * 120)

    total_params = 0

    for item in summary:
        total_params += item["parameters"]

        print(
            f"{item['name']:<60}"
            f" Shape={str(item['shape']):<20}"
            f" Params={item['parameters']:,}"
        )

    print("=" * 120)
    print(f"Total Target Parameters: {total_params:,}")
    print("=" * 120)


if __name__ == "__main__":
    print("Loading TinyLlama...")

    model = load_tinyllama()

    extractor = TensorExtractor(model)

    summary = extractor.get_tensor_summary()

    print_tensor_report(summary)
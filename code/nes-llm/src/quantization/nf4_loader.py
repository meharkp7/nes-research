from transformers import AutoModelForCausalLM
from transformers import BitsAndBytesConfig
import torch


MODEL_ID = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"


class NF4Loader:

    @staticmethod
    def load_nf4_model():

        quant_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.float16,
        )

        model = AutoModelForCausalLM.from_pretrained(
            MODEL_ID,
            quantization_config=quant_config,
            device_map="auto",
            trust_remote_code=True,
        )

        return model

    @staticmethod
    def inspect_layer(
        model,
        layer_idx=0,
        tensor_name="q_proj",
    ):

        layer = model.model.layers[layer_idx]

        target = getattr(
            layer.self_attn,
            tensor_name
        )

        print("\n" + "=" * 120)
        print("NF4 KERNEL INSPECTION")
        print("=" * 120)

        print(f"\nLayer: {layer_idx}")
        print(f"Tensor: {tensor_name}")

        print("\nModule Type:")
        print(type(target))

        print("\nWeight Type:")
        print(type(target.weight))

        print("\nStored Shape:")
        print(target.weight.shape)

        print("\nQuant Type:")
        print(target.weight.quant_type)

        print("\nBlock Size:")
        print(target.weight.blocksize)

        print("\nHas Quant State:")
        print(hasattr(target.weight, "quant_state"))

        if hasattr(target.weight, "quant_state"):

            print("\nQuant State Type:")
            print(type(target.weight.quant_state))

            print("\nQuant State Attributes:")

            for attr in dir(target.weight.quant_state):

                if attr.startswith("_"):
                    continue

                print(attr)

        print("\nAttempting Reconstruction...")

        try:

            reconstructed = target.weight.dequantize()

            print("\nSUCCESS")

            print("\nReconstructed Type:")
            print(type(reconstructed))

            print("\nReconstructed Shape:")
            print(reconstructed.shape)

            print("\nReconstructed Dtype:")
            print(reconstructed.dtype)

            print("\nStatistics")

            print(
                f"Mean      : {reconstructed.float().mean().item():.8f}"
            )

            print(
                f"Std       : {reconstructed.float().std().item():.8f}"
            )

            print(
                f"Min       : {reconstructed.float().min().item():.8f}"
            )

            print(
                f"Max       : {reconstructed.float().max().item():.8f}"
            )

        except Exception as e:

            print("\nDEQUANTIZATION FAILED")
            print(type(e))
            print(e)

        print("\n" + "=" * 120)

    @staticmethod
    def inspect_all_target_families(model):

        families = [
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
        ]

        for family in families:

            NF4Loader.inspect_layer(
                model,
                layer_idx=0,
                tensor_name=family,
            )


if __name__ == "__main__":

    print("Loading NF4 TinyLlama...")

    model = NF4Loader.load_nf4_model()

    NF4Loader.inspect_layer(
        model,
        layer_idx=0,
        tensor_name="q_proj",
    )
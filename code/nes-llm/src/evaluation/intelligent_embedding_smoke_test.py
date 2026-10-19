import torch

from transformers import (
    AutoModelForCausalLM,
    BitsAndBytesConfig,
)

from src.embedding.intelligent_embedder import (
    IntelligentEmbedder,
)

MODEL_ID = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"


def random_payload(
    n_bits=10000,
):

    return torch.randint(

        0,

        2,

        (n_bits,),

    ).tolist()


def load_models():

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

    return fp16, nf4


def main():

    fp16, nf4 = load_models()

    payload = random_payload()

    print()

    print(

        "Payload Bits:",

        len(payload),

    )

    embedder = IntelligentEmbedder()

    print()

    print(

        "Running Intelligent Embedder..."

    )

    result = embedder.embed(

        fp16,

        nf4,

        payload,

    )

    print()

    print("=" * 60)

    print("Embedding Summary")

    print("=" * 60)

    print()

    print(

        "Embedded Profiles:",

        len(result.stego_profiles),

    )

    print(

        "Allocation Entries:",

        len(result.allocation_plan),

    )

    print(

        "Metadata Entries:",

        len(result.layer_metadata),

    )

    print()

    total_capacity = sum(

        metadata.capacity

        for metadata in result.layer_metadata

    )

    total_payload = sum(

        metadata.payload_size

        for metadata in result.layer_metadata

    )

    print(

        "Total Capacity:",

        total_capacity,

    )

    print(

        "Embedded Bits:",

        total_payload,

    )

    print()

    utilization = (

        100

        *

        total_payload

        /

        total_capacity

    )

    print(

        f"Utilization: {utilization:.6f}%"

    )

    print()

    first = result.layer_metadata[0]

    print(

        "First Embedded Layer:",

        first.layer,

    )

    print(

        "Module:",

        first.module,

    )

    print(

        "Embedded Bits:",

        first.payload_size,

    )

    print(

        "Carrier Capacity:",

        first.capacity,

    )

    print()

    print("Smoke Test Passed ✅")


if __name__ == "__main__":

    main()
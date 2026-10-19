import torch
import json

from src.embedding.keyed_residual_embedder import (
    KeyedResidualEmbedder
)


def tensor_stats(tensor):

    flat = tensor.flatten().float()

    return {
        "mean": flat.mean().item(),
        "std": flat.std().item(),
        "min": flat.min().item(),
        "max": flat.max().item(),
    }


def distribution_shift(
    original,
    embedded,
):

    original = original.float()
    embedded = embedded.float()

    return {
        "mean_shift":
            abs(
                original.mean().item()
                -
                embedded.mean().item()
            ),

        "std_shift":
            abs(
                original.std().item()
                -
                embedded.std().item()
            ),
    }


def main():

    capacity = 1_000_000

    residual = torch.randn(
        capacity
    )

    payload_sizes = [

        100,
        1000,
        10000,
        50000,
        100000,
        250000,
        500000,
        #  now after this the rng.sample() will give error bcs capacity set to 500000 , will go in negtive 
        #  ValueError("Sample larger than population or is negative")
        864969846,  
        198263984698246,
        9864161823608294620,
        81725918269826928489284,
        265223,
        432
    ]

    results = {}

    for payload_size in payload_sizes:

        print(
            f"\nTesting {payload_size} bits"
        )

        bits = torch.randint(
            0,
            2,
            (payload_size,)
        ).tolist()

        embedded = (
            KeyedResidualEmbedder.embed_bits(
                residual,
                bits,
                "mehar123"
            )
        )

        shift = (
            distribution_shift(
                residual,
                embedded
            )
        )

        results[
            str(payload_size)
        ] = shift

        print(shift)

    with open(
        "capacity_results.json",
        "w"
    ) as f:

        json.dump(
            results,
            f,
            indent=4
        )

    print(
        "\nSaved capacity_results.json"
    )


if __name__ == "__main__":
    main()
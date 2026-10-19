import torch
import pickle

from src.embedding.keyed_residual_embedder import (
    KeyedResidualEmbedder
)


def main():

    samples = []

    residual_size = 100000

    for i in range(1000):

        residual = torch.randn(
            residual_size
        )

        # clean sample
        samples.append(
            (
                residual.clone(),
                0
            )
        )

        # stego sample
        bits = torch.randint(
            0,
            2,
            (10000,)
        ).tolist()

        stego = (
            KeyedResidualEmbedder
            .embed_bits(
                residual.clone(),
                bits,
                "nes_secret"
            )
        )

        samples.append(
            (
                stego,
                1
            )
        )

        if i % 100 == 0:

            print(
                f"Generated {i}"
            )

    with open(
        "detector_dataset.pkl",
        "wb"
    ) as f:

        pickle.dump(
            samples,
            f
        )

    print(
        "Saved detector_dataset.pkl"
    )


if __name__ == "__main__":
    main()
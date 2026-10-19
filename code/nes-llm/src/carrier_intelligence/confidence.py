import torch


class ConfidenceDecoder:
    """
    Estimates recovery confidence for each
    extracted bit.

    Confidence is based on the normalized
    distance from the embedding decision
    boundary.
    """

    def __init__(
        self,
        temperature: float = 5.0,
        eps: float = 1e-8,
    ):

        self.temperature = temperature
        self.eps = eps

    def confidence(
        self,
        embedded_tensor,
        margins,
        positions,
    ):

        flat = embedded_tensor.flatten()

        margin_flat = margins.flatten()

        probabilities = []

        for pos in positions:

            distance = abs(
                flat[pos]
            )

            normalized = (

                distance

                /

                (

                    margin_flat[pos]

                    + self.eps

                )

            )

            probability = torch.sigmoid(

                normalized

                * self.temperature

            )

            probabilities.append(

                probability.item()

            )

        return probabilities

    def statistics(
        self,
        confidence,
    ):

        tensor = torch.tensor(
            confidence
        )

        return {

            "mean":
            tensor.mean().item(),

            "std":
            tensor.std().item(),

            "min":
            tensor.min().item(),

            "max":
            tensor.max().item(),

        }


def main():

    residual = torch.randn(
        1000
    )

    margins = torch.rand(
        1000
    ) * 0.02

    positions = list(
        range(
            100
        )
    )

    decoder = (
        ConfidenceDecoder()
    )

    confidence = decoder.confidence(

        residual,

        margins,

        positions,

    )

    print()

    print(

        decoder.statistics(

            confidence

        )

    )


if __name__ == "__main__":
    main()
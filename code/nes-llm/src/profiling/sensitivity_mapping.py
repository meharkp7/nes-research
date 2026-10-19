import torch


class SensitivityMapper:
    """
    Computes a local sensitivity map for a residual tensor.

    A coefficient is considered sensitive if a small
    perturbation causes a relatively large local change.

    Output
    ------
    Tensor of identical shape.
    """

    def __init__(
        self,
        epsilon: float = 1e-4,
    ):
        self.epsilon = epsilon

    def compute(
        self,
        residual_tensor: torch.Tensor,
    ) -> torch.Tensor:

        residual = (
            residual_tensor
            .detach()
            .float()
        )

        sensitivity = torch.zeros_like(
            residual
        )

        flat = residual.flatten()
        sens = sensitivity.flatten()

        for i in range(len(flat)):

            original = flat[i]

            flat[i] += self.epsilon

            perturbed = flat[i]

            sens[i] = (
                abs(
                    perturbed - original
                )
                / self.epsilon
            )

            flat[i] = original

        return sensitivity.reshape(
            residual_tensor.shape
        )


def main():

    residual = torch.randn(
        32,
        32,
    )

    mapper = SensitivityMapper()

    sensitivity = mapper.compute(
        residual
    )

    print()

    print("Sensitivity Shape")

    print(
        sensitivity.shape
    )

    print()

    print("Mean Sensitivity")

    print(
        sensitivity.mean().item()
    )


if __name__ == "__main__":
    main()
import numpy as np
import torch
import torch.nn.functional as F

from scipy.stats import entropy as scipy_entropy


class LocalEntropyAnalyzer:
    """
    Computes a local entropy map for a tensor.

    Each coefficient receives an entropy value
    computed over its surrounding neighborhood.

    Output:
        Same shape as input tensor.
    """

    def __init__(
        self,
        window_size: int = 3,
        bins: int = 32,
    ):
        self.window_size = window_size
        self.bins = bins

    def _patch_entropy(
        self,
        patch: np.ndarray,
    ) -> float:

        hist, _ = np.histogram(
            patch,
            bins=self.bins,
            density=True,
        )

        hist = hist + 1e-12

        return float(
            scipy_entropy(hist)
        )

    def compute(
        self,
        tensor: torch.Tensor,
    ) -> torch.Tensor:
        """
        Computes a local entropy value
        for every coefficient.

        Returns
        -------
        torch.Tensor

            Shape identical to input.
        """

        original_shape = tensor.shape

        if tensor.ndim == 1:

            tensor = tensor.view(
                1,
                1,
                1,
                -1,
            )

        elif tensor.ndim == 2:

            tensor = tensor.unsqueeze(0).unsqueeze(0)

        else:

            raise ValueError(
                "Expected 1D or 2D tensor."
            )

        padding = self.window_size // 2

        padded = F.pad(
            tensor,
            (
                padding,
                padding,
                padding,
                padding,
            ),
            mode="reflect",
        )

        entropy_map = torch.zeros_like(
            tensor,
            dtype=torch.float32,
        )

        H = tensor.shape[-2]
        W = tensor.shape[-1]

        for i in range(H):

            for j in range(W):

                patch = padded[
                    0,
                    0,
                    i:i+self.window_size,
                    j:j+self.window_size,
                ]

                entropy_map[
                    0,
                    0,
                    i,
                    j,
                ] = self._patch_entropy(
                    patch.cpu().numpy()
                )

        return entropy_map.reshape(
            original_shape
        )


def main():

    residual = torch.randn(
        32,
        32,
    )

    analyzer = LocalEntropyAnalyzer()

    entropy = analyzer.compute(
        residual
    )

    print()

    print("Input Shape")

    print(
        residual.shape
    )

    print()

    print("Entropy Shape")

    print(
        entropy.shape
    )

    print()

    print("Mean Local Entropy")

    print(
        entropy.mean().item()
    )


if __name__ == "__main__":
    main()
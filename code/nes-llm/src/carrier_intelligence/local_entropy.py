import torch


class LocalEntropyEstimator:
    """
    Computes a local Shannon entropy map over
    quantization residuals.

    Pipeline
    --------
    Residual
        ↓
    Normalize
        ↓
    Quantize into bins
        ↓
    Sliding Windows
        ↓
    Shannon Entropy
        ↓
    Average Overlapping Windows
        ↓
    Entropy Map
    """

    def __init__(

        self,

        window_size=64,

        stride=32,

        num_bins=32,

        eps=1e-12,

    ):

        self.window_size = window_size
        self.stride = stride
        self.num_bins = num_bins
        self.eps = eps

    ##########################################################

    def compute(

        self,

        residual,

    ):

        original_shape = residual.shape

        ######################################################
        # Flatten
        ######################################################

        x = residual.flatten().float()

        ######################################################
        # Normalize to [0,1]
        ######################################################

        xmin = x.min()
        xmax = x.max()

        x = (x - xmin) / (

            xmax - xmin + self.eps

        )

        ######################################################
        # Quantize into bins
        ######################################################

        bins = torch.clamp(

            (x * (self.num_bins - 1)).long(),

            min=0,

            max=self.num_bins - 1,

        )

        ######################################################
        # Output buffers
        ######################################################

        entropy_sum = torch.zeros_like(x)

        entropy_count = torch.zeros_like(x)

        ######################################################
        # Sliding window entropy
        ######################################################

        for start in range(

            0,

            len(x) - self.window_size + 1,

            self.stride,

        ):

            end = start + self.window_size

            window = bins[start:end]

            hist = torch.bincount(

                window,

                minlength=self.num_bins,

            ).float()

            prob = hist / hist.sum()

            prob = prob[prob > 0]

            entropy = -(

                prob * torch.log2(prob)

            ).sum()

            entropy_sum[start:end] += entropy

            entropy_count[start:end] += 1

        ######################################################
        # Handle tail window
        ######################################################

        if len(x) % self.stride != 0:

            start = max(

                0,

                len(x) - self.window_size,

            )

            end = len(x)

            window = bins[start:end]

            hist = torch.bincount(

                window,

                minlength=self.num_bins,

            ).float()

            prob = hist / hist.sum()

            prob = prob[prob > 0]

            entropy = -(

                prob * torch.log2(prob)

            ).sum()

            entropy_sum[start:end] += entropy

            entropy_count[start:end] += 1

        ######################################################
        # Average overlapping windows
        ######################################################

        entropy_map = entropy_sum / (

            entropy_count + self.eps

        )

        ######################################################
        # Normalize entropy to [0,1]
        ######################################################

        entropy_map = (

            entropy_map - entropy_map.min()

        ) / (

            entropy_map.max()

            - entropy_map.min()

            + self.eps

        )

        ######################################################
        # Restore original shape
        ######################################################

        return entropy_map.reshape(original_shape)


##############################################################

def main():

    residual = torch.randn(

        4096,

        4096,

    )

    estimator = LocalEntropyEstimator()

    entropy = estimator.compute(

        residual

    )

    print()

    print("Shape :", entropy.shape)

    print("Mean  :", entropy.mean().item())

    print("Std   :", entropy.std().item())

    print("Min   :", entropy.min().item())

    print("Max   :", entropy.max().item())


if __name__ == "__main__":
    main()
import numpy as np
import torch
from scipy.stats import skew, kurtosis
from scipy.stats import entropy as scipy_entropy


class TensorStatistics:

    @staticmethod
    def compute(tensor: torch.Tensor):
        data = tensor.detach().cpu().float().numpy().flatten()

        mean = float(np.mean(data))
        std = float(np.std(data))
        variance = float(np.var(data))

        minimum = float(np.min(data))
        maximum = float(np.max(data))

        skewness = float(skew(data))
        kurt = float(kurtosis(data))

        hist, _ = np.histogram(
            data,
            bins=256,
            density=True
        )

        hist = hist + 1e-12

        ent = float(scipy_entropy(hist))

        return {
            "mean": mean,
            "std": std,
            "variance": variance,
            "min": minimum,
            "max": maximum,
            "skewness": skewness,
            "kurtosis": kurt,
            "entropy": ent,
            "num_parameters": int(data.size)
        }
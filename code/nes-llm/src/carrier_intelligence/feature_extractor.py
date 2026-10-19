import torch

from src.carrier_intelligence.robustness import (
    RobustnessAnalyzer,
)

from src.carrier_intelligence.information import (
    InformationAnalyzer,
)

from src.carrier_intelligence.distortion import (
    DistortionAnalyzer,
)


class CarrierFeatureExtractor:
    """
    Extracts a complete feature matrix for
    every carrier.

    Output

        Shape

            [num_carriers, 9]
    """

    def __init__(self):

        self.robustness = RobustnessAnalyzer()

        self.information = InformationAnalyzer()

        self.distortion = DistortionAnalyzer()

    def extract(
        self,
        residual,
        fp16_weight,
        quantized_weight,
    ):

        robust = self.robustness.robustness(
            fp16_weight,
            quantized_weight,
        )

        info = self.information.information(
            residual,
        )

        dist = self.distortion.distortion(
            residual,
        )

        feature_matrix = torch.stack(
            [
                info["magnitude"].flatten(),
                info["variance"].flatten(),
                info["std"].flatten(),

                robust["quantization_error"].flatten(),
                robust["stability"].flatten(),
                robust["robustness"].flatten(),

                dist["cost"].flatten(),
                dist["zscore"].flatten(),
                dist["distortion"].flatten(),
            ],
            dim=1,
        )

        return feature_matrix
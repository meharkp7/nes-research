import torch

from src.carrier_selection.selector_base import (
    CarrierSelector,
)

from src.carrier_selection.carrier_scorer import (
    CarrierScorer,
)


class MagnitudeSelector(CarrierSelector):
    """
    Select the highest-magnitude residuals.
    """

    def select(
        self,
        residual_tensor,
        num_bits,
        secret_key=None,
    ):

        flat = residual_tensor.flatten()

        scores = CarrierScorer.normalize(
            CarrierScorer.magnitude(flat)
        )

        ranked = torch.argsort(
            scores,
            descending=True,
        )

        return ranked[:num_bits].tolist()
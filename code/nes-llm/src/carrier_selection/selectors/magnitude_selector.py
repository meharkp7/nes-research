import hashlib
import random

import torch

from src.carrier_selection.selector_base import (
    CarrierSelector,
)


class MagnitudeSelector(CarrierSelector):
    """
    Magnitude-aware carrier selection.

    Carriers are ranked according to the absolute
    residual magnitude.

    A keyed shuffle is then applied over the highest
    ranked candidates to preserve security while
    favouring robust embedding locations.
    """

    def __init__(
        self,
        oversampling_factor=10,
    ):
        self.oversampling_factor = oversampling_factor

    def select(
        self,
        residual_tensor,
        num_bits,
        secret_key,
    ):

        flat = residual_tensor.flatten()

        scores = flat.abs()

        candidate_count = min(
            len(flat),
            num_bits * self.oversampling_factor,
        )

        ranked = torch.argsort(
            scores,
            descending=True,
        )

        candidates = ranked[:candidate_count].tolist()

        seed = int(
            hashlib.sha256(
                secret_key.encode()
            ).hexdigest(),
            16,
        )

        rng = random.Random(seed)

        rng.shuffle(candidates)

        return candidates[:num_bits]
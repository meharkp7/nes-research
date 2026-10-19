import hashlib
import random
import torch

from src.carrier_selection.selector_base import (
    CarrierSelector,
)

from src.carrier_selection.adaptive_scorer import (
    AdaptiveScorer,
)


class AdaptiveSelector(CarrierSelector):
    """
    Adaptive Carrier Selection (ACS).

    Carrier quality is determined by the
    AdaptiveScorer.

    The highest-scoring carriers are selected
    for embedding.

    A deterministic keyed shuffle is optionally
    applied to avoid embedding in a predictable
    order.
    """

    def __init__(
        self,
        scorer=None,
        randomize=True,
    ):

        if scorer is None:
            scorer = AdaptiveScorer()

        self.scorer = scorer
        self.randomize = randomize

    def select(
        self,
        residual_tensor,
        num_bits,
        secret_key,
    ):

        scores = self.scorer.score(
            residual_tensor
        ).flatten()

        ranked = torch.argsort(
            scores,
            descending=True,
        ).tolist()

        selected = ranked[:num_bits]

        if self.randomize:

            seed = int(
                hashlib.sha256(
                    secret_key.encode()
                ).hexdigest(),
                16,
            )

            rng = random.Random(seed)

            rng.shuffle(selected)

        return selected
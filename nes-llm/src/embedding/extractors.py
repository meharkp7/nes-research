"""
Scheme-specific bit extraction.

``DecryptPipeline`` hardcodes ``SignExtractor``, which reads a bit from
the carrier's sign. That is correct for sign-based schemes and wrong for
everything else: a parity/grid scheme stores the bit in ``floor(v/w) % 2``
and carries no sign information to read.

This module maps a strategy to the extractor that can actually decode it,
and exposes one entry point so extraction is selected the same way
embedding is.
"""

import inspect
from typing import Any, Callable, Dict, List, Optional

import torch


class SignBasedExtractor:
    """Reads a bit from the carrier's sign.

    Thin wrapper so every extractor has the same call shape as the
    non-sign ones.
    """

    def extract(
        self,
        residuals: Dict[int, torch.Tensor],
        carrier_indices: Dict[int, List[int]],
        residuals_ref: Optional[Dict[int, torch.Tensor]] = None,
    ) -> List[int]:
        from src.extraction.sign_extractor import SignExtractor

        return SignExtractor().extract(residuals, carrier_indices)


class LweParityExtractor:
    """Reads a bit from grid-interval membership.

    ``bit = floor(value / interval_width) % 2``, with the grid width
    derived from the stego tensor's own statistics. Does not need the
    cover: the embedding is sparse enough that the embedded tensor's
    standard deviation equals the cover's.
    """

    def __init__(self, strategy):
        self.strategy = strategy

    def extract(
        self,
        residuals: Dict[int, torch.Tensor],
        carrier_indices: Dict[int, List[int]],
        residuals_ref: Optional[Dict[int, torch.Tensor]] = None,
    ) -> List[int]:
        return self.strategy.extract(
            residuals, carrier_indices, residuals_ref
        )


class AdaptiveRoutedExtractor:
    """Decoder for whichever branch ``AdaptiveStrategy`` selected.

    Only the strategy knows which branch the noise estimate routed to,
    so the companion extractor is fetched from it at call time
    (``get_extractor``). ``residuals_ref`` — the cover, when a caller
    genuinely has one — is forwarded the way the design's own
    companion expects; with no cover (what an extractor really holds)
    the stego tensor stands in, which ``LWEStrategy.extract``
    documents as equivalent for sparse payloads.
    """

    def __init__(self, strategy):
        self.strategy = strategy

    def extract(
        self,
        residuals: Dict[int, torch.Tensor],
        carrier_indices: Dict[int, List[int]],
        residuals_ref: Optional[Dict[int, torch.Tensor]] = None,
    ) -> List[int]:
        ref = residuals_ref if residuals_ref is not None else residuals
        return self.strategy.get_extractor(ref).extract(
            residuals, carrier_indices
        )


# Strategies whose encoding is not a sign flip, and therefore need an
# extractor other than SignExtractor.
NON_SIGN_EXTRACTORS: Dict[str, Callable] = {
    "lwe": LweParityExtractor,
    "adaptive": AdaptiveRoutedExtractor,
}

# Everything else uses sign, which is what production already does.
DEFAULT_EXTRACTOR = SignBasedExtractor


def extractor_for(strategy_name: str, strategy_instance: Any = None):
    """Return the extractor that can decode this strategy's encoding.

    Non-sign schemes need their strategy instance because the grid width
    is key-derived and lives on the strategy. The sign extractor wraps
    production's SignExtractor and takes no instance.
    """
    factory = NON_SIGN_EXTRACTORS.get(
        strategy_name, DEFAULT_EXTRACTOR
    )

    if factory is DEFAULT_EXTRACTOR:
        return factory()

    return factory(strategy_instance)


def can_decode(strategy_name: str) -> bool:
    """Whether any extractor exists for this strategy."""
    return strategy_name in NON_SIGN_EXTRACTORS or True


def extract_bits(
    strategy_name: str,
    strategy_instance: Any,
    embedded: Dict[int, torch.Tensor],
    carrier_indices: Dict[int, List[int]],
    residuals_ref: Optional[Dict[int, torch.Tensor]] = None,
) -> List[int]:
    """Extract bits using the extractor that matches the scheme.

    ``residuals_ref`` is passed through for schemes that genuinely need
    it, and ignored for those that do not.
    """
    extractor = extractor_for(strategy_name, strategy_instance)

    signature = inspect.signature(extractor.extract)

    if "residuals_ref" in signature.parameters:
        return extractor.extract(
            embedded, carrier_indices, residuals_ref
        )

    return extractor.extract(embedded, carrier_indices)
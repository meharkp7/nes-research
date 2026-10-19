"""
Dataset loading helpers for the experiment suite.

Kept separate so a missing/offline dataset produces a clear NOT_RUN with
a reason, rather than an exception that looks like an experiment failure.
"""

from functools import lru_cache
from typing import List, Optional

DEFAULT_DATASET = "wikitext"
DEFAULT_CONFIG = "wikitext-2-raw-v1"
DEFAULT_SPLIT = "validation"


def load_wikitext2(
    split: str = DEFAULT_SPLIT,
    dataset: str = DEFAULT_DATASET,
    config: str = DEFAULT_CONFIG,
) -> Optional[List[str]]:
    """Return the raw text lines, or ``None`` if the dataset is unavailable."""
    try:
        from datasets import load_dataset

        data = load_dataset(dataset, config, split=split)
        return list(data["text"])
    except Exception as exc:  # offline, gated, renamed, missing cache
        print(f"  [dataset] unavailable: {type(exc).__name__}: {exc}")
        return None


def select_texts(
    texts: List[str],
    num_texts: int = 200,
    min_length: int = 50,
) -> List[str]:
    """Guide protocol: keep texts longer than ``min_length``, take the first N."""
    return [
        text
        for text in texts
        if len(text.strip()) > min_length
    ][:num_texts]


@lru_cache(maxsize=4)
def load_texts(
    num_texts: int = 200,
    min_length: int = 50,
    split: str = DEFAULT_SPLIT,
) -> tuple:
    """Load and filter WikiText-2. Returns ``(texts, error)``."""
    raw = load_wikitext2(split=split)

    if raw is None:
        return (), (
            f"wikitext-2-raw-v1 split '{split}' could not be loaded "
            "(dataset missing or offline)"
        )

    selected = select_texts(raw, num_texts, min_length)

    if not selected:
        return (), "no WikiText-2 samples passed the length filter"

    return tuple(selected), ""
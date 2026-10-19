"""
Delta-only distribution (W8).

What ships is a patch, not a checkpoint: the delta
``W_stego - W_clean`` (~10,256 of ~811M values) plus the integrity
metadata to catch a corrupted or rewritten file (W8.2), and a
recipient tool that reconstructs the payload from base model + delta +
key without ever meeting the sender (W8.1).

    from src.delta import build_delta, save_delta, load_delta
    from src.delta import recover_payload

See ``format`` for the file format and ``recipient`` for recovery.
"""

from src.delta.format import (  # noqa: F401
    DELTA_DEFINITION,
    DELTA_FORMAT,
    DELTA_VERSION,
    DeltaIntegrityError,
    build_delta,
    canonical_sha256,
    load_delta,
    save_delta,
    verify_delta,
)
from src.delta.recipient import (  # noqa: F401
    UnsupportedStrategyError,
    reconstruct_residuals,
    recover_payload,
)

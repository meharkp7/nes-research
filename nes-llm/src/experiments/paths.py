"""
Canonical filesystem locations for the NES experiment suite.

Everything is resolved from this file's own location so that running the
suite from ``nes-research/`` versus ``nes-research/nes-llm/`` always reads
and writes the same artifacts.

Layout::

    nes-research/
    ├── results/          structured experiment outputs
    ├── artifacts/        large binary datasets
    └── nes-llm/
        └── src/experiments/paths.py   (this file)
"""

from pathlib import Path

# nes-llm/src/experiments/paths.py -> nes-llm/src/experiments
#                                    -> nes-llm/src
#                                    -> nes-llm
#                                    -> nes-research
REPO_ROOT = Path(__file__).resolve().parents[3]
PROJECT_ROOT = Path(__file__).resolve().parents[2]

RESULTS_DIR = REPO_ROOT / "results"
ARTIFACTS_DIR = REPO_ROOT / "artifacts"

RESULTS_DIR.mkdir(parents=True, exist_ok=True)
ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)

ARCHIVE_DIR = RESULTS_DIR / "_archive"


def model_slug(model_id: str) -> str:
    """Filesystem-safe, collision-free slug for a HuggingFace model id.

    ``Qwen/Qwen2.5-3B`` -> ``qwen__qwen2.5-3b``

    The ``__`` separator is deliberate: it matches the existing
    ``cache/models/<slug>`` convention from ``ModelTensorCache``, so a
    single slug function serves both the cache and the result artifacts.
    """
    return model_id.replace("/", "__").lower()


def result_path(experiment: str, model_id: str) -> Path:
    """Standard artifact path for one experiment on one model.

    Follows the naming convention: ``exp3_qwen__qwen2.5-3b.json``.
    """
    return RESULTS_DIR / f"{experiment}_{model_slug(model_id)}.json"
from __future__ import annotations

import importlib.util
from pathlib import Path

import pandas as pd
import pytest

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "run_packed_nf4_detector_benchmark.py"
spec = importlib.util.spec_from_file_location("nf4_benchmark", SCRIPT)
benchmark = importlib.util.module_from_spec(spec)
assert spec.loader is not None
spec.loader.exec_module(benchmark)


def synthetic_dataset(pairs_per_model: int = 15) -> pd.DataFrame:
    rows = []
    for model_i, model in enumerate(("qwen", "tinyllama", "gemma")):
        for i in range(pairs_per_model):
            source, run, block = f"src{model_i}", f"run{i}", i
            common = {
                "source_id": source, "run_id": run, "block_index": block,
                "model_id": model, "tensor_key": "layers.0.down_proj",
                "artifact_id": f"{model}-{run}", "artifact_sha256": f"hash-{model}-{run}",
                "split": "old", "block_value_count": 64.0,
                "entropy_bits": 3.5 + (i % 2) * 0.01,
            }
            rows.append({**common, "role": "clean", "label": 0, "artifact_id": f"{model}-{run}-clean",
                         "artifact_sha256": f"{model}-{run}-clean-hash"})
            rows.append({**common, "role": "embedded", "label": 1, "artifact_id": f"{model}-{run}-embedded",
                         "artifact_sha256": f"{model}-{run}-embedded-hash",
                         "entropy_bits": 3.7 + (i % 2) * 0.01})
    return pd.DataFrame(rows)


def test_validation_and_mixed_split_keep_pairs_and_all_models():
    df = synthetic_dataset()
    features = benchmark.validate_dataset(df)
    split = benchmark.make_mixed_split(df, seed=7)
    assert features == ["block_value_count", "entropy_bits"]
    assert set(split["split"]) == {"train", "validation", "test"}
    assert all(set(g["label"].astype(int)) == {0, 1} for _, g in split.groupby(list(benchmark.PAIR_COLS)))
    for _, part in split.groupby("split"):
        assert set(part["model_id"]) == {"qwen", "tinyllama", "gemma"}
        assert set(part["label"].astype(int)) == {0, 1}


def test_split_is_reproducible():
    df = synthetic_dataset()
    a = benchmark.make_mixed_split(df, seed=123)
    b = benchmark.make_mixed_split(df, seed=123)
    assert a["split"].tolist() == b["split"].tolist()


def test_invalid_pair_is_rejected():
    df = synthetic_dataset()
    df = df.drop(df.index[0])
    with pytest.raises(ValueError, match="matched pair"):
        benchmark.validate_dataset(df)

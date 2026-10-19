import pandas as pd
import pytest

from scripts.packed_nf4_method_eval_utils import (
    infer_method, validate_chunk, validate_manifest,
)


@pytest.fixture
def manifest():
    return {"group_to_split": {
        "train-a": "train", "train-b": "train",
        "train-c": "train", "validation-a": "validation",
        "test-a": "test", "test-b": "test",
    }}


def rows():
    return pd.DataFrame([
        {"run_id": "train-a_qse_20261011", "model_id": "train-a",
         "role": "clean", "label": 0, "split": "train"},
        {"run_id": "train-a_dce_20261011", "model_id": "train-a",
         "role": "embedded", "label": 1, "split": "train"},
    ])


def test_manifest_requires_grouped_splits(manifest):
    assert validate_manifest(manifest) == {
        "train": 3, "validation": 1, "test": 2
    }


def test_manifest_rejects_too_few_test_models():
    with pytest.raises(ValueError, match="Insufficient"):
        validate_manifest({"group_to_split": {
            "a": "train", "b": "train", "c": "validation",
            "d": "test",
        }})


def test_infer_method_and_validate_chunk(manifest):
    frame = rows()
    assert infer_method(frame["run_id"]).tolist() == ["qse", "dce"]
    assert validate_chunk(frame, manifest).tolist() == ["qse", "dce"]


def test_rejects_split_mismatch(manifest):
    frame = rows()
    frame.loc[0, "split"] = "test"
    with pytest.raises(ValueError, match="split disagrees"):
        validate_chunk(frame, manifest)


def test_rejects_role_label_mismatch(manifest):
    frame = rows()
    frame.loc[0, "label"] = 1
    with pytest.raises(ValueError, match="Role/label mismatch"):
        validate_chunk(frame, manifest)


def test_rejects_unknown_run_suffix(manifest):
    frame = rows()
    frame.loc[0, "run_id"] = "train-a_unknown_20261011"
    with pytest.raises(ValueError, match="Unrecognized"):
        validate_chunk(frame, manifest)

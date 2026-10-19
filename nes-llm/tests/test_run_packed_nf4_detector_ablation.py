from __future__ import annotations

import importlib.util
from pathlib import Path

import pandas as pd

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "run_packed_nf4_detector_ablation.py"
spec = importlib.util.spec_from_file_location("nf4_ablation", SCRIPT)
ablation = importlib.util.module_from_spec(spec)
assert spec.loader is not None
spec.loader.exec_module(ablation)


def synthetic_qwen_dataset(groups: int = 12, blocks_per_group: int = 3) -> pd.DataFrame:
    rows = []
    for run_i in range(groups):
        for block in range(blocks_per_group):
            common = {
                "source_id": "qwen-source",
                "run_id": f"run-{run_i}",
                "block_index": block,
                "model_id": ablation.QWEN,
                "tensor_key": "layers.0.down_proj",
                "split": "old",
                "block_value_count": 64.0,
                "entropy_bits": 3.0 + (block % 2) * 0.1,
                "code_frequency_06": 0.1 + (block % 3) * 0.01,
            }
            rows.append({
                **common, "role": "clean", "label": 0,
                "artifact_id": f"clean-{run_i}-{block}",
                "artifact_sha256": f"clean-hash-{run_i}-{block}",
            })
            rows.append({
                **common, "role": "embedded", "label": 1,
                "artifact_id": f"embedded-{run_i}-{block}",
                "artifact_sha256": f"embedded-hash-{run_i}-{block}",
                "entropy_bits": common["entropy_bits"] - 0.2,
                "code_frequency_06": common["code_frequency_06"] + 0.03,
            })
    return pd.DataFrame(rows)


def test_ablation_conditions_are_distinct_and_have_held_out_metrics():
    df = synthetic_qwen_dataset()
    features = ablation.validate_dataset(df)
    split = ablation.make_mixed_split(df, seed=42)
    parts = {name: split[split["split"] == name] for name in ablation.SPLITS}
    full = ablation.evaluate_condition(parts["train"], parts["validation"], parts["test"], features, 42)
    entropy = ablation.evaluate_condition(parts["train"], parts["validation"], parts["test"], ["entropy_bits"], 42)
    no_entropy = ablation.evaluate_condition(
        parts["train"], parts["validation"], parts["test"],
        [feature for feature in features if feature != "entropy_bits"], 42,
    )
    assert full["n_features"] == len(features)
    assert entropy["features"] == ["entropy_bits"]
    assert "entropy_bits" not in no_entropy["features"]
    for result in (full, entropy, no_entropy):
        assert 0.0 <= result["test_pooled"]["roc_auc"] <= 1.0
        assert result["test_pooled"]["n_rows"] == len(parts["test"])
        assert result["test_by_source_run"]


def test_paired_delta_distribution_reports_nonzero_fraction_and_quantiles():
    df = synthetic_qwen_dataset(groups=4, blocks_per_group=2)
    result = ablation.paired_delta_distribution(df, ["entropy_bits", "code_frequency_06"])
    qwen = result["by_model"][ablation.QWEN]
    assert qwen["entropy_bits"]["matched_pairs"] == 8
    assert qwen["entropy_bits"]["nonzero_fraction"] == 1.0
    assert abs(qwen["entropy_bits"]["quantiles"]["0.5"] - (-0.2)) < 1e-12
    assert qwen["code_frequency_06"]["mean"] > 0


def test_qwen_grouped_sensitivity_holds_out_groups_or_reports_insufficient_groups():
    df = synthetic_qwen_dataset(groups=8, blocks_per_group=2)
    result = ablation.qwen_grouped_sensitivity(df, ["entropy_bits", "code_frequency_06"], seed=3)
    assert result["status"] == "COMPLETED"
    held_out = [group for fold in result["folds"] for group in fold["held_out_groups"]]
    assert len(held_out) == len(set(held_out)) == result["group_count"]
    small = ablation.qwen_grouped_sensitivity(df[df["run_id"].isin(["run-0", "run-1"])],
                                               ["entropy_bits"], seed=3)
    assert small["status"] == "NOT_EVALUABLE"

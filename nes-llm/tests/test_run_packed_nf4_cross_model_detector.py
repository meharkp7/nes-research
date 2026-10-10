import numpy as np
import pandas as pd

from scripts.run_packed_nf4_cross_model_detector import cross_model_results


def _dataset():
    rows = []
    rng = np.random.RandomState(11)
    for model in ("model_a", "model_b", "model_c"):
        for group in ("run_1", "run_2"):
            for block in range(40):
                signal = int(block % 2)
                # Shared label signal plus a model-specific offset; model ID is
                # metadata and must not be passed as a feature.
                value = signal + rng.normal(0, 0.35)
                for label, role in ((0, "clean"), (1, "embedded")):
                    rows.append({
                        "artifact_id": f"{model}-{group}-{block}-{label}",
                        "artifact_sha256": f"sha-{model}-{group}-{block}-{label}",
                        "run_id": group,
                        "source_id": model,
                        "model_id": model,
                        "tensor_key": "layer.test",
                        "role": role,
                        "label": label,
                        "block_index": block,
                        "feature": value + (0.03 if label else 0.0) + rng.normal(0, 0.1),
                        "feature_2": rng.normal(),
                    })
    return pd.DataFrame(rows)


def test_cross_model_evaluation_holds_out_each_model_and_reports_transfer():
    df = _dataset()
    result = cross_model_results(df, ["feature", "feature_2"], seed=7)
    assert result["status"] == "COMPLETED"
    assert result["model_count"] == 3
    assert set(result["results_by_held_out_model"]) == {"model_a", "model_b", "model_c"}
    for model, item in result["results_by_held_out_model"].items():
        assert item["status"] == "COMPLETED"
        assert model not in item["train_model_families"]
        assert item["held_out_model_family"] == model
        assert set(item["estimators"]) == {"logistic_regression", "random_forest", "hist_gradient_boosting"}
        for metrics in item["estimators"].values():
            assert 0.0 <= metrics["roc_auc"] <= 1.0
            assert metrics["metrics_at_fixed_threshold_0_5"]["threshold"] == 0.5


def test_cross_model_requires_three_model_families():
    df = _dataset()
    result = cross_model_results(df[df["model_id"] != "model_c"], ["feature", "feature_2"], seed=7)
    assert result["status"] == "NOT_EVALUABLE"

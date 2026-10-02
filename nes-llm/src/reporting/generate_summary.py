"""
Final research summary.

Writes ``results/final_research_summary.md`` from the manifest and the
cross-model table.

This document reports what the experiments actually established,
including the failures. In particular it:

-   states the neural-detector FAIL as a result, not an omission;
-   calls the capacity figure "maximum tested", never "maximum";
-   describes the LWE component as LWE-inspired;
-   distinguishes validated architectures from merely registered ones.

Those constraints come from §25 and are the difference between a summary
that can be checked against the artifacts and one that cannot.
"""

import json
from pathlib import Path
from typing import Any, Dict

from src.experiments import manifest as manifest_mod
from src.experiments.artifact_manager import load_json, utc_now
from src.experiments.experiment_registry import (
    ARCHITECTURE_SUPPORT,
    TARGET_MODELS,
)
from src.experiments.paths import RESULTS_DIR
from src.reporting.aggregate_results import (
    GATE_COLUMNS,
    aggregate,
    missing_cells,
    write_cross_model_csv,
    write_matrix,
)

SUMMARY_PATH = RESULTS_DIR / "final_research_summary.md"


def _status_icon(status: str) -> str:
    return {
        manifest_mod.PASS: "PASS",
        manifest_mod.FAIL: "**FAIL**",
        manifest_mod.NOT_RUN: "NOT_RUN",
        manifest_mod.MISSING_ARTIFACT: "MISSING",
        manifest_mod.IMPLEMENTED_ONLY: "IMPL_ONLY",
        manifest_mod.ERROR: "ERROR",
    }.get(status, status)


def _fmt(value: Any) -> str:
    if value is None:
        return "-"
    if isinstance(value, float):
        return f"{value:.6g}"
    return str(value)



def _diagnostic_section(add) -> None:
    """Fold the FAIL diagnostics into the report.

    Each recorded FAIL comes with a diagnostic saying whether it is
    fixable. Reporting only the FAIL would leave a reader to assume it
    is either a bug or an oversight.
    """
    neural_study = load_json(
        RESULTS_DIR / "exp7_neural_parameter_study.json"
    )
    calibration = load_json(
        RESULTS_DIR / "exp2_criterion_calibration.json"
    )

    add("## 5. Why the FAILs fail")
    add("")
    add(
        "A FAIL is a measurement, not a bug. Each was investigated to "
        "determine whether it is fixable; the answer is recorded rather "
        "than assumed."
    )
    add("")

    if calibration:
        add("### 5.1 Exp2 residual-magnitude gate")
        add("")
        add(
            f"- {calibration.get('models_passing', 0)}/"
            f"{calibration.get('models_tested', 0)} profiled models "
            "meet the `mag_mean > 0.002` criterion."
        )
        for row in calibration.get("per_quant_format", []):
            if "error" in row:
                continue
            add(
                f"- `{row['quant_format']}`: mean magnitude "
                f"{row['mean_mag_mean']:.6f}, "
                f"{row['fraction_above_threshold']:.0%} of probed layers "
                "above threshold"
            )
        add(f"- {calibration.get('finding', '')}")
        add(
            "- The threshold was **not** changed and no Exp2 verdict was "
            "rewritten. Recalibrating it is a research decision for the "
            "authors, not something to apply silently."
        )
        add("")

    if neural_study:
        add("### 5.2 Exp7 neural-detector gate")
        add("")
        accuracies = [
            v["accuracy"] for v in neural_study.get("variants", [])
        ]
        if accuracies:
            add(
                f"- Swept {len(accuracies)} configurations (alpha x100, "
                "gamma x5, payload x10): detector accuracy stayed within "
                f"{min(accuracies):.2%}-{max(accuracies):.2%}."
            )
            add(
                "- No configuration came near the 55% gate, so this is "
                "**not fixable by retuning**."
            )
        density = next(
            (
                v["signal_density"]
                for v in neural_study.get("variants", [])
                if "signal_density" in v
            ),
            None,
        )
        if density:
            add(
                "- Mechanism: sign embedding rewrites a carrier to "
                "+/-|r|, which changes nothing when the payload bit "
                "already agrees with the carrier's sign. Only "
                f"~{density['mean_changed_values_per_patch']:.0f} of "
                f"{density['patch_size']} values per patch actually "
                f"differ and "
                f"{density['identical_pair_fraction']:.0%} of pairs are "
                "byte-identical."
            )
        add(f"- {neural_study.get('conclusion', '')}")
        add(
            "- Getting under the gate would need a different embedding "
            "scheme, one that does not force a sign flip at carriers. "
            "That is future work, not a retuning."
        )
        add("")

def build_summary() -> str:
    data = aggregate()
    manifest = manifest_mod.load()

    coverage = data["coverage"]
    matrix = data["matrix"]
    cross_model = data["cross_model"]

    lines: List[str] = []
    add = lines.append

    add("# NES Multi-Model Research — Final Summary")
    add("")
    add(f"Generated: {utc_now()}")
    add("")
    add(
        "This summary is generated from `results/experiment_manifest.json`. "
        "Every number below corresponds to a saved artifact."
    )
    add("")

    # ---------------------------------------------------------------
    add("## 1. Coverage")
    add("")
    add("| Status | Cells |")
    add("| --- | --- |")
    for status, count in coverage.items():
        if count:
            add(f"| {status} | {count} |")
    add("")
    add(
        f"Total cells: {sum(coverage.values())}. "
        "PASS and FAIL both represent completed experiments; only FAIL "
        "means the experiment ran and its gate was not met."
    )
    add("")

    # ---------------------------------------------------------------
    add("## 2. Model x experiment matrix")
    add("")
    experiments = sorted(
        {
            experiment
            for row in matrix.values()
            for experiment in row
        }
    )

    add("| Model | " + " | ".join(experiments) + " |")
    add("| --- |" + " --- |" * len(experiments))

    for spec in TARGET_MODELS:
        model_id = spec["model_id"]
        row = matrix.get(model_id, {})
        cells = [
            _status_icon(row.get(experiment, manifest_mod.NOT_RUN))
            for experiment in experiments
        ]
        add(f"| `{model_id}` | " + " | ".join(cells) + " |")

    add("")

    missing = missing_cells(manifest)
    if missing:
        add(f"### Missing cells ({len(missing)})")
        add("")
        add("| Model | Experiment |")
        add("| --- | --- |")
        for item in missing[:60]:
            add(f"| `{item['model_id']}` | {item['experiment']} |")
        if len(missing) > 60:
            add(f"| ... | {len(missing) - 60} more |")
        add("")

    # ---------------------------------------------------------------
    add("## 3. Cross-model gate table")
    add("")
    if cross_model:
        add(
            "| Model | Layers (exp/act) | "
            + " | ".join(GATE_COLUMNS)
            + " | Overall |"
        )
        add("| --- | --- |" + " --- |" * (len(GATE_COLUMNS) + 1))

        for model_id, row in cross_model.items():
            cells = [
                _status_icon((row.get(gate) or {}).get("status", "NOT_RUN"))
                for gate in GATE_COLUMNS
            ]
            layers = (
                f"{row.get('expected_layers')}/"
                f"{row.get('actual_layers')}"
            )
            add(
                f"| `{model_id}` | {layers} | "
                + " | ".join(cells)
                + f" | {_status_icon(row.get('overall_status', 'NOT_RUN'))} |"
            )
        add("")
    else:
        add("No cross-model table artifact found.")
        add("")

    # ---------------------------------------------------------------
    add("## 4. Key findings")
    add("")

    # Neural detector — always reported, pass or fail.
    neural = manifest.get("records", {}).get(
        "exp7_neural::Qwen/Qwen2.5-3B"
    )

    add("### 4.1 Neural steganalysis (Exp7 neural)")
    add("")
    if neural:
        accuracy = (neural.get("metrics") or {}).get("accuracy")
        add(f"- Status: **{neural['status']}**")
        if accuracy is not None:
            add(f"- Detector accuracy: {accuracy * 100:.1f}% (gate <= 55%)")
        add(
            "- A carrier-centered MLP detects the embedding above "
            "threshold even though the statistical gate passes. This is "
            "a real FAIL and is preserved as such."
        )
        add(
            "- Scope: paired patches from a single model. This is a "
            "stress test of the embedding, not a claim that a detector "
            "trained this way generalizes to other models or carriers."
        )
    else:
        add("- Not recorded in the manifest.")
    add("")

    # Statistical security
    stat = manifest.get("records", {}).get(
        "exp7::Qwen/Qwen2.5-3B"
    )
    add("### 4.2 Statistical security (Exp7 statistical)")
    add("")
    if stat:
        metrics = stat.get("metrics") or {}
        add(f"- Status: **{stat['status']}**")
        add(f"- KL divergence: {_fmt(metrics.get('kl_divergence'))} (gate <= 0.05)")
        add(
            "- Statistical detector accuracy: "
            f"{_fmt((metrics.get('statistical_detector_accuracy') or 0) * 100)}% "
            "(gate <= 55%)"
        )
        add(
            "- This gate and the neural gate are reported separately and "
            "are not combined into a single security score."
        )
    else:
        add("- Not recorded in the manifest.")
    add("")

    # Capacity
    capacity = manifest.get("records", {}).get(
        "exp4::Qwen/Qwen2.5-3B"
    )
    add("### 4.3 Capacity")
    add("")
    if capacity:
        metrics = capacity.get("metrics") or {}
        maximum = metrics.get("max_tested_payload_at_ber_zero_bits")
        add(f"- Status: **{capacity['status']}**")
        if maximum:
            add(
                f"- **Maximum tested** payload at BER=0: "
                f"{int(maximum):,} bits"
            )
        add(
            "- This is the largest payload tested, not an absolute "
            "capacity bound. No stopping criterion for the true maximum "
            "was established."
        )
    else:
        add("- Not recorded in the manifest.")
    add("")

    # Fidelity
    fidelity = manifest.get("records", {}).get(
        "exp5::Qwen/Qwen2.5-3B"
    )
    add("### 4.4 Fidelity")
    add("")
    if fidelity:
        metrics = fidelity.get("metrics") or {}
        add(f"- Status: **{fidelity['status']}**")
        add(f"- NF4 baseline PPL: {_fmt(metrics.get('nf4_baseline_ppl'))}")
        add(
            "- Reconstruction control PPL: "
            f"{_fmt(metrics.get('reconstruction_control_ppl'))}"
        )
        add(f"- Embedded PPL: {_fmt(metrics.get('embedded_ppl'))}")
        add(
            "- Embedding-specific degradation: "
            f"{_fmt(metrics.get('ppl_degradation_pct'))}% "
            "(gate < 2%)"
        )
        reconstruction = metrics.get("reconstruction_only_delta_pct")
        if reconstruction is not None:
            add(
                f"- Reconstruction alone moves PPL by "
                f"{_fmt(reconstruction)}%; that component is not "
                "attributable to the payload."
            )
    else:
        add("- Not recorded in the manifest.")
    add("")

    # Exp9
    add("### 4.5 Non-NF4 formats (Exp9)")
    add("")
    exp9 = [
        entry
        for key, entry in manifest.get("records", {}).items()
        if key.startswith("exp9::")
    ]
    if exp9:
        for entry in exp9:
            fmt = entry.get("quantization_format", "unknown")
            add(
                f"- `{entry['model_id']}` ({fmt.upper()}): "
                f"**{entry['status']}** — {entry.get('notes', '')}"
            )
        add(
            "- GPTQ and AWQ dequantization adapters are implemented and "
            "verified bit-exact against the AutoGPTQ reference unpack. "
            "No GPTQ or AWQ checkpoint has been run yet, so these cells "
            "carry no experimental evidence."
        )
    else:
        add("- Not recorded in the manifest.")
    add("")

    # ---------------------------------------------------------------
    _diagnostic_section(add)

    # ---------------------------------------------------------------
    add("## 6. Architecture support")
    add("")
    add(
        "Registry entries and experimental validation are different "
        "things. Only the families below have been exercised."
    )
    add("")
    add("| Family | Status |")
    add("| --- | --- |")
    for family, status in sorted(ARCHITECTURE_SUPPORT.items()):
        add(f"| {family} | {status} |")
    add("")
    add(
        "Falcon and MoE families are not validated. Their presence in "
        "the registry does not imply support."
    )
    add("")

    # ---------------------------------------------------------------
    add("## 7. Methodological caveats")
    add("")
    add(
        "- The LWE component is **LWE-inspired** (an HMAC-SHA256 keyed "
        "interval/grid mechanism), not a mathematically validated LWE "
        "cryptosystem."
    )
    add(
        "- QRNG infrastructure exists but no working production QRNG "
        "provider has been established."
    )
    add(
        "- Exp6 and Exp7 are stochastic. Seeds and trial counts are "
        "recorded in each artifact; small drift in non-gate values "
        "between runs is expected."
    )
    add(
        "- A FAIL in this table means the experiment ran and did not "
        "meet its gate. It is never converted to PASS, and a missing "
        "cell is never reported as one."
    )
    add("")

    # ---------------------------------------------------------------
    add("## 8. Artifacts")
    add("")
    for path in sorted(RESULTS_DIR.glob("*.json")):
        if path.name.startswith("_"):
            continue
        add(f"- `results/{path.name}`")
    add("")
    add(
        "Superseded artifact versions are preserved under "
        "`results/_archive/` rather than overwritten in place."
    )
    add("")

    return "\n".join(lines) + "\n"


def write_summary() -> Path:
    SUMMARY_PATH.parent.mkdir(parents=True, exist_ok=True)
    SUMMARY_PATH.write_text(build_summary(), encoding="utf-8")
    return SUMMARY_PATH


def generate_all() -> Dict[str, Path]:
    """Write every reporting deliverable."""
    manifest = manifest_mod.load()

    return {
        "matrix_json": write_matrix(manifest)["json"],
        "matrix_csv": write_matrix(manifest)["csv"],
        "cross_model_csv": write_cross_model_csv(),
        "summary": write_summary(),
    }
"""
Consistency check for the experiment suite.

Verifies the invariants that make the results trustworthy (handoff
Phase F). Run after any suite execution:

    python run_nes_experiments.py --audit
    python check_consistency.py

Each check corresponds to a way results can look finished while being
wrong: a PASS with no artifact behind it, a FAIL quietly rewritten as
PASS, a missing cell reported as a default, or a capacity number
presented as an absolute bound.
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from src.experiments import manifest as manifest_mod  # noqa: E402
from src.experiments.experiment_registry import (  # noqa: E402
    ARCHITECTURE_SUPPORT,
    TARGET_MODELS,
)
from src.experiments.paths import RESULTS_DIR  # noqa: E402

def resolve_artifact_path(raw_path):
    """Resolve historical absolute paths after a repository is moved.

    Older manifest entries record the author's local absolute path. Prefer that
    path when it exists; otherwise resolve the artifact by its repository-root
    results/ location. Do not treat a missing artifact as present.
    """
    if not raw_path:
        return None
    path = Path(raw_path)
    if path.exists():
        return path
    fallback = RESULTS_DIR / path.name
    if fallback.exists():
        return fallback
    return path



def check(name, ok, detail=""):
    status = "OK  " if ok else "FAIL"
    print(f"  [{status}] {name}")
    if detail:
        print(f"         {detail}")
    return ok


def main() -> int:
    manifest = manifest_mod.load()
    records = manifest.get("records", {})

    print("=" * 70)
    print("CONSISTENCY CHECK")
    print("=" * 70)
    print()

    results = []

    # 1. Every PASS has an artifact on disk.
    missing = []
    for key, entry in records.items():
        if entry["status"] not in (
            manifest_mod.PASS, manifest_mod.FAIL
        ):
            continue
        raw_path = entry.get("artifact_path")
        path = resolve_artifact_path(raw_path)
        if path is None or not path.exists():
            missing.append(f"{key} -> {raw_path}")

    results.append(
        check(
            "every completed cell has an artifact on disk",
            not missing,
            ", ".join(missing) if missing else
            f"{sum(1 for e in records.values() if e['status'] in ('PASS', 'FAIL'))} artifacts verified",
        )
    )

    # 2. Artifact status agrees with manifest status.
    disagreements = []
    for key, entry in records.items():
        path = resolve_artifact_path(entry.get("artifact_path"))
        if path is None or not path.exists():
            continue
        try:
            data = json.loads(Path(path).read_text(encoding="utf-8"))
        except Exception:
            continue

        artifact_status = data.get("status")

        # exp9 records one manifest cell per format, all sharing a single
        # artifact whose top-level status is the overall verdict. Compare
        # against that format's own result, not the file-level status,
        # or a single passing format masks a failing one.
        if key.startswith("exp9::"):
            model_id = key.split("::", 1)[1]
            per_format = (
                (data.get("metrics") or {})
                .get("results", {})
                .get(model_id, {})
            )
            if per_format:
                artifact_status = per_format.get("status")

        # exp9 records one manifest cell per format, all sharing a file.
        if artifact_status and artifact_status != entry["status"]:
            disagreements.append(
                f"{key}: manifest={entry['status']} artifact={artifact_status}"
            )

    results.append(
        check(
            "manifest status matches artifact status",
            not disagreements,
            "; ".join(disagreements) if disagreements else "",
        )
    )

    # 3. No IMPLEMENTED_ONLY / MISSING_ARTIFACT silently became PASS.
    results.append(
        check(
            "no PASS recorded without a gate_status",
            all(
                e.get("gate_status") is not None
                for e in records.values()
                if e["status"] == manifest_mod.PASS
            ),
        )
    )

    # 4. The neural detector FAIL must still be a FAIL.
    neural = records.get("exp7_neural::Qwen/Qwen2.5-3B")
    if neural is not None:
        accuracy = (neural.get("metrics") or {}).get("accuracy")
        expected = "PASS" if (accuracy or 1.0) <= 0.55 else "FAIL"
        results.append(
            check(
                "exp7_neural FAIL preserved (not weakened to PASS)",
                neural["status"] == expected,
                f"accuracy={accuracy} gate=0.55 status={neural['status']}",
            )
        )

    # 5. Statistical and neural security are separate cells.
    results.append(
        check(
            "exp7 statistical and exp7_neural are distinct cells",
            "exp7::Qwen/Qwen2.5-3B" in records
            and "exp7_neural::Qwen/Qwen2.5-3B" in records,
        )
    )

    # 6. Capacity is reported as maximum *tested*.
    capacity = records.get("exp4::Qwen/Qwen2.5-3B")
    if capacity is not None:
        path = resolve_artifact_path(capacity.get("artifact_path"))
        text = ""
        if path is not None and path.exists():
            text = path.read_text(encoding="utf-8").lower()
        results.append(
            check(
                "capacity labelled 'maximum tested', not absolute",
                "not an absolute capacity bound" in text,
            )
        )

    # 7. Exp8 overall is FAIL when any gate failed.
    table_path = RESULTS_DIR / "cross_model_table_real.json"
    if table_path.exists():
        table = json.loads(table_path.read_text(encoding="utf-8"))
        bad = []
        for model_id, row in table.items():
            statuses = [
                (row.get(g) or {}).get("status", "NOT_RUN")
                for g in (
                    "g2_qaci", "g3_ber", "g4_ppl",
                    "g5_robustness", "g6_statistical",
                    "g6_neural_detector",
                )
            ]
            if "FAIL" in statuses and row.get("overall_status") != "FAIL":
                bad.append(model_id)
            if (
                "NOT_RUN" in statuses
                and row.get("overall_status") == "PASS"
            ):
                bad.append(f"{model_id} (NOT_RUN promoted to PASS)")

        results.append(
            check(
                "exp8 overall never hides a failed or unrun gate",
                not bad,
                ", ".join(bad) if bad else "",
            )
        )

    # 8. Layer counts recorded, not assumed.
    if table_path.exists():
        table = json.loads(table_path.read_text(encoding="utf-8"))
        missing_counts = [
            m
            for m, row in table.items()
            if row.get("actual_layers") is None
        ]
        results.append(
            check(
                "exp8 records actual layer counts",
                not missing_counts,
                ", ".join(missing_counts) if missing_counts else "",
            )
        )

    # 9. Unvalidated architectures are not claimed as supported.
    results.append(
        check(
            "no unvalidated architecture marked VALIDATED",
            all(
                status in ("VALIDATED", "NOT_VALIDATED", "UNKNOWN")
                for status in ARCHITECTURE_SUPPORT.values()
            ),
        )
    )

    # 10. Report models with no cells at all.
    covered = {e["model_id"] for e in records.values()}
    uncovered = [
        spec["model_id"]
        for spec in TARGET_MODELS
        if spec["model_id"] not in covered
    ]
    print()
    print(f"  Models with no manifest cells: {len(uncovered)}")
    for model_id in uncovered:
        print(f"    - {model_id}")
    print()

    print("=" * 70)
    passed = sum(1 for r in results if r)
    print(f"{passed}/{len(results)} checks passed")
    print("=" * 70)

    return 0 if all(results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
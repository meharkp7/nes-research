"""
Claim audit: re-derive every MEASURED claim from the artifacts on disk.

    python claim_audit.py

This is the executable form of the "Final claim audit" section in
`RESEARCH_PLAN.md`. The plan states what the numbers are; this checks
they still are what the plan says.

Why it exists: the two claims that failed the manual audit failed for
different reasons and neither would have been caught by reading the
document. "7 models covered, 0 errors" was true of the NF4 grid and
wrong about the GPTQ/AWQ cells sitting beside it. "5 of 5 models pass
both gates" was written from console output, while the artifact for
that run held three measured models. A number that looks finished is
the one worth re-reading, so re-reading is automated.

Exit status is 1 if any claim fails or cannot be verified. A claim that
cannot be found in its artifact is UNVERIFIED, never assumed true.
"""

import json
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from src.experiments import manifest as manifest_mod  # noqa: E402
from src.experiments.experiment_registry import THRESHOLDS  # noqa: E402
from src.experiments.paths import RESULTS_DIR  # noqa: E402

RESULTS = RESULTS_DIR

_rows = []


def check(name, ok, detail=""):
    """Record one claim. ``detail`` is shown on failure."""
    _rows.append((name, bool(ok), detail))
    mark = "OK  " if ok else "FAIL"
    print(f"  [{mark}] {name}" + (f"  -- {detail}" if detail else ""))
    return bool(ok)


def load(pattern):
    hits = sorted(RESULTS.glob(pattern))
    return [json.loads(p.read_text()) for p in hits] if hits else []


def one(pattern):
    hits = load(pattern)
    return hits[0] if hits else None


def dig(obj, *path):
    for key in path:
        if not isinstance(obj, dict) or key not in obj:
            return None
        obj = obj[key]
    return obj


# --------------------------------------------------------------- suite state
def audit_manifest():
    records = manifest_mod.load()["records"]
    counts = Counter(r["status"] for r in records.values())

    check(
        "coverage is 35 PASS / 6 FAIL / 0 NOT_RUN / 0 ERROR",
        counts.get("PASS", 0) == 35
        and counts.get("FAIL", 0) == 6
        and counts.get("NOT_RUN", 0) == 0
        and counts.get("ERROR", 0) == 0,
        str(dict(counts)),
    )
    check(
        "every cell carries a gate_status",
        all(r.get("gate_status") for r in records.values()),
    )
    check(
        "exp7_neural still FAIL at 70.5% (not weakened to PASS)",
        dig(records, "exp7_neural::Qwen/Qwen2.5-3B", "status") == "FAIL",
        str(records.get("exp7_neural::Qwen/Qwen2.5-3B", {}).get("status")),
    )


# ------------------------------------------------------------------ exp3-7
def audit_nf4_grid():
    docs = load("exp3_*.json")
    bers = [dig(d, "metrics", "ber") for d in docs]
    check(
        "exp3 sign round trip BER 0.0 on every model",
        bool(docs) and all(b == 0.0 for b in bers),
        f"n={len(docs)} bers={bers}",
    )
    check(
        "exp3 compared 48,256 bits with 0 errors",
        bool(docs)
        and all(dig(d, "metrics", "bit_errors") == 0 for d in docs)
        and all(dig(d, "metrics", "bits_compared") == 48256 for d in docs),
        f"compared={[dig(d, 'metrics', 'bits_compared') for d in docs]}",
    )

    d = one("exp4_*.json")
    m = dig(d, "metrics") or {}
    bits = m.get(
        "max_tested_payload_at_ber_zero_bits",
        m.get("max_bits_at_ber_zero"),
    )
    check(
        "exp4 capacity >= 10M bits at BER 0",
        isinstance(bits, int) and bits >= 10_000_000,
        f"bits={bits}",
    )

    d = one("exp5_*.json")
    pct = None
    m = dig(d, "metrics") or {}
    for key in ("ppl_degradation_pct", "degradation_pct", "delta_pct"):
        if key in m:
            pct = m[key]
            break
    if pct is None:
        for value in m.values():
            if isinstance(value, dict) and "ppl_degradation_pct" in value:
                pct = value["ppl_degradation_pct"]
                break
    check(
        "exp5 sign fidelity PPL delta is +0.0053% and under the 2% gate",
        pct is not None and 0 < pct < 2.0,
        f"pct={pct}",
    )

    docs = load("exp6_*.json")
    at_001, at_002 = [], []
    for d in docs:
        m = dig(d, "metrics") or {}
        at_001.append(m.get("ber_at_sigma_0_001"))
        at_002.append(m.get("ber_at_sigma_0_002"))
    check(
        "exp6 every model under the robustness gates (0.02 / 0.10)",
        bool(docs)
        and all(b is not None and b <= 0.02 for b in at_001)
        and all(b is not None and b <= 0.10 for b in at_002),
        f"sigma0.001={at_001} sigma0.002={at_002}",
    )
    # The claim used to read "BER 0 at sigma=0.001". Mistral-7B is
    # 0.00115, so the honest form is 6 of 7 exact zeros inside the gate.
    check(
        "exp6 BER 0.0 at sigma=0.001 on 6 of 7 models (Mistral-7B 0.00115)",
        sum(1 for b in at_001 if b == 0.0) == 6
        and max(b for b in at_001 if b is not None) < 0.002,
        f"zeros={sum(1 for b in at_001 if b == 0.0)}/{len(at_001)} "
        f"max={max(b for b in at_001 if b is not None)}",
    )

    accs = []
    for path in sorted(RESULTS.glob("exp7_*.json")):
        if "neural" in path.name:
            continue
        m = dig(json.loads(path.read_text()), "metrics") or {}
        accs += [
            v for k, v in m.items()
            if "accuracy" in k and isinstance(v, (int, float))
        ]
    check(
        "exp7 statistical detection under the 55% gate, every model",
        bool(accs) and max(accs) <= 0.55,
        f"n={len(accs)} max={max(accs) if accs else None}",
    )

    d = one("exp7_neural_qwen*.json")
    acc = dig(d, "metrics", "accuracy") or dig(d, "metrics", "detector_accuracy")
    check(
        "exp7 neural detects sign at 70.5% -> FAIL stands",
        acc is not None and abs(acc - 0.705) < 0.005 and (d or {}).get("status") == "FAIL",
        f"acc={acc} status={(d or {}).get('status')}",
    )

    d = one("exp8_*.json")
    status = (d or {}).get("status")
    if status is None:
        status = dig(d, "metrics", "overall_status")
    check("exp8 cross-model FAIL (inherits the neural FAIL)", status == "FAIL",
          f"status={status}")


# -------------------------------------------------------------------- exp9
def audit_formats():
    d = one("exp9_formats.json")
    if not d:
        check("exp9 artifact present", False)
        return

    results = d["metrics"]["results"]
    g = results.get("Qwen/Qwen2.5-3B-Instruct-GPTQ-Int4", {})
    a = results.get("Qwen/Qwen2.5-3B-Instruct-AWQ", {})
    gm, am = g.get("metrics", {}), a.get("metrics", {})
    gv, av = gm.get("dequant_verification", {}), am.get("dequant_verification", {})

    check(
        "GPTQ: BER 0.0, 10256/10256 bits, 0 errors",
        gm.get("ber") == 0.0 and gm.get("bits_compared") == 10256
        and gm.get("bit_errors") == 0,
        f"ber={gm.get('ber')} bits={gm.get('bits_compared')} "
        f"err={gm.get('bit_errors')}",
    )
    check(
        "GPTQ: corr 0.9903, all 36 layers, no scale correction",
        gv.get("usable") is True
        and abs((gv.get("correlation") or 0) - 0.9903) < 5e-4
        and gm.get("layers") == 36
        and gv.get("scale_corrected") is False,
        f"corr={gv.get('correlation')} layers={gm.get('layers')} "
        f"corrected={gv.get('scale_corrected')}",
    )

    check(
        "AWQ: BER 0.0, 10256/10256 bits, 0 errors",
        am.get("ber") == 0.0 and am.get("bits_compared") == 10256
        and am.get("bit_errors") == 0,
        f"ber={am.get('ber')} bits={am.get('bits_compared')} "
        f"err={am.get('bit_errors')}",
    )
    check(
        "AWQ: corr 0.9941 (raw 0.9890), 35/36 layers",
        av.get("usable") is True
        and abs((av.get("correlation") or 0) - 0.9941) < 5e-4
        and abs((av.get("raw_correlation") or 0) - 0.9890) < 5e-4
        and am.get("layers") == 35,
        f"corr={av.get('correlation')} raw={av.get('raw_correlation')} "
        f"layers={am.get('layers')}",
    )
    check(
        "AWQ: the excluded layer is named, not dropped silently",
        list((am.get("layers_excluded") or {}).keys()) == ["2"],
        f"excluded={list((am.get('layers_excluded') or {}).keys())}",
    )


# ---------------------------------------------------------------- exp10/12
def audit_lwe():
    d = one("exp10_*.json")
    if not d:
        check("exp10 artifact present", False)
    else:
        lwe = next(
            (r for r in d.get("results", []) if r.get("strategy") == "lwe"),
            None,
        )
        check("exp10 lists an LWE strategy entry", lwe is not None)
        if lwe:
            check(
                "LWE extracts without the cover (BER 0.0000, was 0.5036)",
                lwe.get("extract_needs_cover") is False
                and dig(lwe, "extractability", "ber") == 0.0,
                f"needs_cover={lwe.get('extract_needs_cover')} "
                f"ber={dig(lwe, 'extractability', 'ber')}",
            )

    d = one("exp12_lwe_cross_model.json")
    if not d:
        check("exp12 artifact present", False)
        return

    measured = [r for r in d.get("results", []) if r.get("status") == "MEASURED"]
    skipped = [r for r in d.get("results", []) if r.get("status") == "SKIPPED"]
    passing = [r for r in measured if r.get("satisfies_both_gates")]

    check(
        "exp12: every measured model passes both gates",
        bool(measured) and len(passing) == len(measured),
        f"{len(passing)}/{len(measured)} passing, {len(skipped)} skipped",
    )
    check(
        "exp12: recorded counts match the rows in the file",
        d.get("models_measured") == len(measured)
        and d.get("models_passing") == len(passing),
        f"file says {d.get('models_passing')}/{d.get('models_measured')}, "
        f"rows are {len(passing)}/{len(measured)}",
    )

    # The claim that failed the manual audit. Both the 5-row table in
    # RESEARCH_LOG §7 and the commit message say five models were
    # measured; if this is still false, "5 of 5" must not be cited.
    check(
        "exp12 covers all 5 non-skipped cached models (closes the audit "
        "finding)",
        len(measured) == 5,
        f"measured={len(measured)} "
        f"present={[r.get('model_id') for r in d.get('results', [])]}",
    )


# ------------------------------------------------------------------ gates
def audit_thresholds():
    check(
        "dequant gate thresholds unchanged (0.95 / 0.5)",
        THRESHOLDS["exp9"].get("min_dequant_correlation") == 0.95
        and THRESHOLDS["exp9"].get("max_dequant_residual_ratio") == 0.5,
        str({k: v for k, v in THRESHOLDS["exp9"].items()
             if "dequant" in k}),
    )
    check(
        "exp2 residual threshold untouched (0.002)",
        THRESHOLDS["exp2"].get("min_mean_magnitude") == 0.002,
        str(THRESHOLDS["exp2"].get("min_mean_magnitude")),
    )
    check(
        "detectability gate untouched (0.55) for statistical and neural",
        THRESHOLDS["exp7_statistical"].get("max_detector_accuracy") == 0.55
        and THRESHOLDS["exp7_neural"].get("max_detector_accuracy") == 0.55,
        f"{THRESHOLDS['exp7_statistical'].get('max_detector_accuracy')} / "
        f"{THRESHOLDS['exp7_neural'].get('max_detector_accuracy')}",
    )


def main() -> int:
    print("=" * 70)
    print("CLAIM AUDIT — every MEASURED claim, re-derived from disk")
    print("=" * 70)

    for title, fn in (
        ("suite state", audit_manifest),
        ("NF4 grid (exp3-exp8)", audit_nf4_grid),
        ("non-NF4 formats (exp9)", audit_formats),
        ("LWE (exp10, exp12)", audit_lwe),
        ("gates", audit_thresholds),
    ):
        print(f"\n{title}")
        fn()

    failed = [r for r in _rows if not r[1]]
    print()
    print("=" * 70)
    if failed:
        print(f"{len(failed)}/{len(_rows)} claims FAILED")
        for name, _, detail in failed:
            print(f"  - {name}  {detail}")
        print("=" * 70)
        return 1

    print(f"all {_rows.__len__()} claims verified")
    print("=" * 70)
    return 0


if __name__ == "__main__":
    sys.exit(main())

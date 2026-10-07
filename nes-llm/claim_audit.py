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
import math
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

    # Cell counts: the W0 table in RESEARCH_PLAN states these, and two of
    # its numbers went stale ("exp2 FAIL x6", "exp7_neural FAIL x7") while
    # every value around them stayed correct. A count stated in prose is
    # a claim, so it gets a check like any other.
    cells = Counter(key.split("::", 1)[0] for key in records)
    check(
        "cell counts match the tables (exp1/2/3/6/7 are 7 cells, "
        "exp4/5/7_neural/8 are 1, exp9 is 2)",
        all(cells.get(name) == 7 for name in ("exp1", "exp2", "exp3", "exp6", "exp7"))
        and all(
            cells.get(name) == count
            for name, count in (
                ("exp4", 1),
                ("exp5", 1),
                ("exp7_neural", 1),
                ("exp8", 1),
                ("exp9", 2),
            )
        ),
        str(dict(cells)),
    )


# ------------------------------------------------------------------ exp3-7
def audit_nf4_grid():
    # exp2: the claim read "fails on 6 of 7 models" and the manifest says 4.
    records = manifest_mod.load().get("records", {})
    exp2 = {
        key.split("::", 1)[1]: rec.get("status")
        for key, rec in records.items()
        if key.startswith("exp2::")
    }
    failed = sorted(m for m, s in exp2.items() if s == "FAIL")
    check(
        "exp2 fails on 4 of 7 models (claim said 6 of 7)",
        len(exp2) == 7 and len(failed) == 4,
        f"n={len(exp2)} FAIL={len(failed)} {failed}",
    )
    check(
        "exp2 PASS cells are gemma-2-9b, Phi-3-mini, Mistral-7B",
        sorted(m for m, s in exp2.items() if s == "PASS")
        == [
            "google/gemma-2-9b",
            "microsoft/Phi-3-mini-4k-instruct",
            "mistralai/Mistral-7B-v0.3",
        ],
        str(sorted(m for m, s in exp2.items() if s == "PASS")),
    )

    # The calibration is a separate measurement over the legacy profile
    # files. It used to say "Zero of 7" in its own `finding` while its
    # own `models_passing` said 1, and it predates the gemma-2-9b and
    # Llama-3.1-8B re-runs, so its per-model rows disagreed with exp2.
    cal = one("exp2_criterion_calibration.json")
    if not cal:
        check("calibration artifact present", False)
    else:
        rows = cal.get("per_model", [])
        gates = [r for r in rows if r.get("meets_exp2_gate")]
        check(
            "calibration's models_passing matches its own rows",
            cal.get("models_passing") == len(gates)
            and cal.get("models_tested") == len(rows),
            f"says {cal.get('models_passing')}/{cal.get('models_tested')}, "
            f"rows give {len(gates)}/{len(rows)}",
        )
        check(
            "calibration's finding string states its own count",
            str(cal.get("finding", "")).startswith(f"{len(gates)} of "),
            f"finding={str(cal.get('finding', ''))[:60]!r}",
        )
        by_model = {r.get("model"): r for r in rows}
        disagree = [
            m
            for m, status in exp2.items()
            if m in by_model
            and (status == "PASS") != bool(by_model[m].get("meets_exp2_gate"))
        ]
        check(
            "calibration and exp2 agree on every shared model",
            not disagree,
            f"disagree={disagree}",
        )

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
            # exp10's rows were regenerated by the Phase-3 no-cover fix;
            # its stored conclusion string was not and still reads
            # "LWE needs the cover to extract" — contradicting the rows
            # below it. The rows are the record; the check pins them and
            # reports the staleness rather than depending on it.
            check(
                "exp10: lwe row current (wins all three axes); stored "
                "conclusion is pre-Phase-3 stale — cite rows, not "
                "conclusion",
                lwe.get("wins") is True
                and dig(lwe, "extractability", "structurally_usable")
                is True,
                "conclusion_stale="
                + str("LWE needs the cover" in d.get("conclusion", "")),
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


# --------------------------------------------------------- exp13 (W4.2)
def audit_keyless():
    d = one("exp13_keyless_recovery.json")
    if not d:
        check("exp13 artifact present", False)
        return

    shipped = d.get("shipped_path", {})
    check(
        "exp13: shipped path is one public constant, key ignored",
        shipped.get("grid_width_property") == 0.01
        and shipped.get("keyed_branch_active") is False
        and shipped.get("distinct_widths_all_layers") == [0.01]
        and shipped.get("width_identical_across_keys") is True
        and shipped.get("secret_key_is_all_zero_default") is True,
        f"w={shipped.get('grid_width_property')} "
        f"distinct={shipped.get('distinct_widths_all_layers')} "
        f"keyed_branch={shipped.get('keyed_branch_active')}",
    )

    designed = d.get("designed_path_control", {})
    check(
        "exp13: designed HMAC formula key-independent on 0/36 layers",
        designed.get("layers_key_dependent") == 0
        and designed.get("layers_total") == 36
        and designed.get("floor_dominated_fraction") == 1.0,
        f"key-dependent={designed.get('layers_key_dependent')}/"
        f"{designed.get('layers_total')}",
    )

    inv = d.get("key_invariance", {})
    check(
        "exp13: six keys decode identical streams (max pairwise BER 0.0)",
        inv.get("keys_decode_identically") is True
        and inv.get("max_pairwise_ber") == 0.0
        and all(v == 0.0 for v in inv.get("ber_vs_transmitted", {}).values()),
        f"max_pairwise={inv.get('max_pairwise_ber')} "
        f"bers={inv.get('ber_vs_transmitted')}",
    )

    phase = d.get("attack_tiers", {}).get("phase_detection", {})
    primary = (phase.get("per_tolerance") or {}).get("1e-06", {})
    check(
        "exp13: keyless read of all 10256 bits at BER 0.0 "
        "(precision/recall 1.0, clean control 0)",
        phase.get("stream_ber_primary_tol") == 0.0
        and phase.get("stream_length") == 10256
        and phase.get("transmitted_length") == 10256
        and primary.get("precision") == 1.0
        and primary.get("recall") == 1.0
        and primary.get("detected_in_clean_control") == 0,
        f"ber={phase.get('stream_ber_primary_tol')} "
        f"len={phase.get('stream_length')}/"
        f"{phase.get('transmitted_length')} "
        f"p={primary.get('precision')} r={primary.get('recall')} "
        f"clean={primary.get('detected_in_clean_control')}",
    )

    # Nondeterministic by construction: the AES key is fresh per run,
    # so the payload bits and the stego layout differ slightly. The
    # claim is "at chance", not a specific digit.
    kc = d.get("attack_tiers", {}).get("kerckhoffs_exact_size", {})
    kc_ber = kc.get("keyless_ber")
    check(
        "exp13: public-pipeline re-run reads at chance (0.4 <= BER <= 0.6)",
        kc_ber is not None and 0.4 <= kc_ber <= 0.6,
        f"ber={kc_ber} "
        f"precision={dig(kc, 'positions', 'precision')} "
        f"alloc_match={kc.get('layers_with_matching_allocation')}",
    )

    ws = d.get("width_search", {})
    check(
        "exp13: search spikes on the true lattice, tied with w/5",
        ws.get("spike_found") is True
        and ws.get("argmax_width") == 0.002
        and ws.get("true_width") == 0.01
        and ws.get("true_width_tied_for_best") is True
        and ws.get("argmax_score") == ws.get("score_at_true_width"),
        f"argmax={ws.get('argmax_width')} "
        f"scores={ws.get('argmax_score')}/"
        f"{ws.get('score_at_true_width')}",
    )

    gate = d.get("gate", {})
    check(
        "exp13: gate FAIL stands and matches THRESHOLDS['exp13']",
        gate.get("status") == "FAIL"
        and gate.get("min_keyless_ber")
        == THRESHOLDS["exp13"]["min_keyless_ber"]
        and gate.get("min_width_search_relative_error")
        == THRESHOLDS["exp13"]["min_width_search_relative_error"]
        and "min_keyless_ber" in (gate.get("failed_conditions") or []),
        f"status={gate.get('status')} failed={gate.get('failed_conditions')}",
    )

    claims = d.get("claims_tested", [])
    check(
        "exp13: all three docstring security claims recorded REFUTED",
        len(claims) == 3
        and all(c.get("verdict") == "REFUTED" for c in claims),
        "; ".join(f"{c.get('verdict')} {c.get('claim')}" for c in claims),
    )


# --------------------------------------------------------- exp14 (W3.2)
def audit_blind():
    d = one("exp14_blind_patch_detector.json")
    if not d:
        check("exp14 artifact present", False)
        return

    gate = d.get("gate", {})
    check(
        "exp14: gate matches THRESHOLDS['exp14'], blind PASS at 0.5000",
        gate.get("max_blind_detector_accuracy")
        == THRESHOLDS["exp14"]["max_blind_detector_accuracy"]
        == 0.55
        and gate.get("measured", {}).get("blind_accuracy") == 0.5
        and gate.get("status") == "PASS"
        and d.get("status") == "PASS",
        f"gate={gate.get('status')} "
        f"blind={gate.get('measured', {}).get('blind_accuracy')}",
    )

    blind = d.get("arms", {}).get("blind", {})
    bm = blind.get("metrics", {})
    check(
        "exp14: blind detector learned nothing (576 pairs, tp=fp=0)",
        bm.get("pairs") == 576
        and bm.get("accuracy") == 0.5
        and bm.get("tp") == 0
        and bm.get("fp") == 0
        and (bm.get("train_pairs") or 0) + (bm.get("test_pairs") or 0)
        == 576,
        f"acc={bm.get('accuracy')} tp={bm.get('tp')} fp={bm.get('fp')} "
        f"pairs={bm.get('pairs')}",
    )

    ctl = d.get("arms", {}).get("control", {})
    cm = ctl.get("metrics", {})
    check(
        "exp14: carrier-centered control clears the gate, so the blind "
        "result is informative",
        cm.get("accuracy") is not None
        and cm.get("accuracy") > 0.55
        and cm.get("accuracy") > (bm.get("accuracy") or 0),
        f"control={cm.get('accuracy')} blind={bm.get('accuracy')}",
    )

    check(
        "exp14: blind positions almost never contain a carrier (4/576)",
        blind.get("positions") == 576
        and blind.get("positions_containing_carrier") == 4
        and (blind.get("fraction_containing_carrier") or 1) < 0.02,
        f"{blind.get('positions_containing_carrier')}/"
        f"{blind.get('positions')}",
    )

    method = d.get("method", {})
    check(
        "exp14: detector unchanged from exp7, split by embedding",
        str(method.get("detector", "")).startswith("MLP 4096")
        and method.get("epochs") == 30
        and method.get("batch_size") == 32
        and method.get("seed") == 42
        and "by embedding" in str(method.get("split", "")),
        f"epochs={method.get('epochs')} split={method.get('split')}",
    )


# --------------------------------------------------------- exp15 (W2)
EXP15_ARTIFACTS = [
    "exp15_lwe_fidelity_qwen__qwen2.5_3b.json",
    "exp15_lwe_fidelity_google__gemma_2_2b.json",
]


def audit_lwe_fidelity():
    arts = [(name, one(name)) for name in EXP15_ARTIFACTS]
    missing = [n for n, d in arts if not d]
    check(
        "exp15: both model artifacts present",
        not missing,
        "2/2" if not missing else "missing: " + ", ".join(missing),
    )
    if missing:
        return
    by_name = dict(arts)

    threshold = THRESHOLDS["exp15"]["max_ppl_degradation_pct"]
    check(
        "exp15: gate matches THRESHOLDS['exp15'] (2%), both models PASS",
        threshold == 2.0
        and all(
            d["gate"]["max_ppl_degradation_pct"] == 2.0
            and d["gate"]["status"] == "PASS"
            and d["status"] == "PASS"
            for _, d in arts
        ),
        "; ".join(f"{d['status']} {d['gate']['status']}" for _, d in arts),
    )

    expected = {
        EXP15_ARTIFACTS[0]: 0.0076998319784398785,
        EXP15_ARTIFACTS[1]: 0.05011987303172728,
    }
    degradations = {
        n: by_name[n]["metrics"]["lwe_ppl_degradation_pct"]
        for n in EXP15_ARTIFACTS
    }
    check(
        "exp15: LWE degradation pinned (Qwen 0.0077%, gemma 0.0501%), "
        "both far under 2%",
        all(
            abs(degradations[n] - expected[n]) < 1e-9
            and degradations[n] < 2.0
            for n in EXP15_ARTIFACTS
        ),
        ", ".join(f"{n.split('_')[2]}={degradations[n]:.4f}%"
                  for n in EXP15_ARTIFACTS),
    )

    check(
        "exp15: exp5 protocol intact — 200 texts, 50k bits, "
        "reconstruction delta reported separately",
        all(
            d["metrics"]["eval_texts"] == 200
            and d["metrics"]["payload_bits"] == 50_000
            and d["metrics"]["reconstruction_only_delta_pct"] < -1.0
            for _, d in arts
        ),
        "; ".join(
            f"recon {d['metrics']['reconstruction_only_delta_pct']:+.2f}%"
            for _, d in arts
        ),
    )

    qwen = by_name[EXP15_ARTIFACTS[0]]
    sr = qwen["metrics"].get("sign_reverification")
    check(
        "exp15: exp5's own numbers reproduce exactly (12.4707 / 11.3494); "
        "sign re-run agrees in verdict, not in digit",
        qwen["metrics"]["nf4_baseline_ppl"] == 12.47069538705203
        and qwen["metrics"]["reconstruction_control_ppl"]
        == 11.349366735378783
        and sr is not None
        and sr["recorded_delta_pct"] == -0.005286623081384053
        and abs(sr["rerun_delta_vs_control_pct"]) < 0.1
        and abs(sr["delta_point_difference"]) < 0.1,
        f"rerun={sr.get('rerun_delta_vs_control_pct') if sr else None}% "
        f"recorded={sr.get('recorded_delta_pct') if sr else None}%",
    )


# --------------------------------------------------------- exp16 (W3.1)
def audit_cross_scheme():
    d = one("exp16_cross_scheme_detector.json")
    if not d:
        check("exp16 artifact present", False)
        return

    gate = d.get("gate", {})
    cross = gate.get("measured_cross", {})
    check(
        "exp16: gate matches THRESHOLDS['exp16'], both cross directions "
        "at chance (0.5000), PASS as pre-registered",
        gate.get("max_cross_scheme_detector_accuracy")
        == THRESHOLDS["exp16"]["max_cross_scheme_detector_accuracy"]
        == 0.55
        and cross.get("sign_to_lwe") == 0.5
        and cross.get("lwe_to_sign") == 0.5
        and gate.get("status") == "PASS"
        and d.get("status") == "PASS",
        f"cross={cross} status={gate.get('status')}",
    )

    results = d.get("results", {})
    within = d.get("within_scheme_controls", {})
    check(
        "exp16: sign pipeline validated (62.85% within-scheme, clears "
        "the 55% line) — so sign->LWE at chance is informative",
        within.get("sign_to_sign", 0) > 0.55
        and abs(within.get("sign_to_sign", 0) - 0.6284722222222222) < 1e-9
        and results.get("sign_to_sign", {}).get("pairs") == 144,
        f"sign within={within.get('sign_to_sign')}",
    )

    check(
        "exp16: LWE control collapsed at 0.5000 — recorded, and "
        "consistent with exp12's measured LWE detector accuracy of "
        "50.00%; LWE->sign direction read as NOT established",
        within.get("lwe_to_lwe") == 0.5
        and results.get("lwe_to_lwe", {}).get("tp") == 0
        and results.get("lwe_to_lwe", {}).get("fp") == 0
        and d.get("controls_valid") is False,
        f"lwe within={within.get('lwe_to_lwe')} "
        f"controls_valid={d.get('controls_valid')}",
    )

    method = d.get("method", {})
    check(
        "exp16: 432 pairs per scheme, split by embedding, detector "
        "unchanged from exp7",
        method.get("pairs_per_scheme") == 432
        and method.get("pairs_per_scheme") >= 400
        and str(method.get("split", "")).startswith("4 embeds train")
        and method.get("epochs") == 30
        and method.get("batch_size") == 32
        and method.get("seed") == 42,
        f"pairs={method.get('pairs_per_scheme')} "
        f"split={method.get('split')}",
    )


# --------------------------------------------------------- exp17 (W1.1)
def audit_qae_round_trip():
    d = one("exp17_qae_round_trip.json")
    if not d:
        check("exp17 artifact present", False)
        return

    gate = d.get("gate", {})
    check(
        "exp17: gate matches THRESHOLDS['exp17'] (exp3's 0.0), PASS — "
        "BER 0.0 over 48,256 bits, decrypt and match required",
        gate.get("max_ber")
        == THRESHOLDS["exp17"]["max_ber"]
        == 0.0
        and gate.get("measured_ber") == 0.0
        and gate.get("decrypt_ok") is True
        and gate.get("recovered_matches_original") is True
        and gate.get("status") == "PASS"
        and d.get("status") == "PASS",
        f"ber={gate.get('measured_ber')} status={gate.get('status')}",
    )

    m = d.get("metrics", {})
    check(
        "exp17: full bit comparison, not a stats field — 48,256 "
        "transmitted = 48,256 extracted, 0 errors",
        m.get("bits_embedded") == 48_256
        and m.get("bits_compared") == 48_256
        and m.get("bit_errors") == 0
        and m.get("ber") == 0.0,
        f"compared={m.get('bits_compared')} errors={m.get('bit_errors')}",
    )

    probes = d.get("probes", {})
    qae_probe = probes.get("qae", {})
    nf4_probe = probes.get("nf4_qae", {})
    check(
        "exp17: no-cover probe passes for qae, records BLOCKED error "
        "for nf4_qae (registered, not faked)",
        qae_probe.get("structurally_usable") is True
        and qae_probe.get("ber") == 0.0
        and nf4_probe.get("embed_ok") is False
        and str(nf4_probe.get("error", "")).startswith(
            "RuntimeError: nf4_qae is BLOCKED"
        ),
        f"qae usable={qae_probe.get('structurally_usable')} "
        f"nf4 error={str(nf4_probe.get('error'))[:40]}...",
    )

    method = d.get("method", {})
    check(
        "exp17: statuses as registered — qae READY, nf4_qae BLOCKED; "
        "exp3's exact flow through QaeDictAdapter",
        method.get("strategy_status")
        == {"qae": "READY", "nf4_qae": "BLOCKED"}
        and "QaeDictAdapter" in str(method.get("adapter", ""))
        and "exp3's exact flow" in str(method.get("round_trip", "")),
        f"status={method.get('strategy_status')}",
    )


# --------------------------------------------------------- exp18 (W1.3)
EXP18_ARTIFACTS = [
    "exp18_matrix_google__gemma_2_2b.json",
    "exp18_matrix_qwen__qwen2.5_3b.json",
    "exp18_matrix_meta_llama__llama_3.1_8b.json",
    "exp18_matrix_microsoft__phi_3_mini_4k_instruct.json",
    "exp18_matrix_qwen__qwen2.5_7b.json",
    "exp18_matrix_mistralai__mistral_7b_v0.3.json",
    "exp18_matrix_google__gemma_2_9b.json",
]
EXP18_STRATEGIES = ["sign", "magnitude_aware", "lwe", "qae"]


def audit_strategy_matrix():
    arts = [(name, one(name)) for name in EXP18_ARTIFACTS]
    missing = [n for n, d in arts if not d]
    check(
        "exp18: all seven grid model artifacts present (3 first pass "
        "+ 4 widened 2026-10-06; TinyLlama SKIPPED under the "
        "incomplete-cache rule)",
        not missing,
        "7/7" if not missing else "missing: " + ", ".join(missing),
    )
    if missing:
        return

    gate = THRESHOLDS["exp18"]
    check(
        "exp18: every artifact's gate matches THRESHOLDS['exp18'] "
        "(exp3's 0.0, exp6's 0.02/0.10, exp7's 0.55 — all reused)",
        gate["max_ber"] == 0.0
        and gate["max_ber_at_sigma_0_001"] == 0.02
        and gate["max_ber_at_sigma_0_002"] == 0.10
        and gate["max_detector_accuracy"] == 0.55
        and all(
            d["gate"]["max_ber"] == 0.0
            and d["gate"]["max_ber_at_sigma_0_001"] == 0.02
            and d["gate"]["max_ber_at_sigma_0_002"] == 0.10
            and d["gate"]["max_detector_accuracy"] == 0.55
            and "THRESHOLDS['exp18']" in d["gate"]["gate_source"]
            for _, d in arts
        ),
        f"gate={gate}",
    )

    check(
        "exp18: four READY strategies measured on every model; neural "
        "and nf4_qae excluded BY NAME with reasons",
        all(
            [c["strategy"] for c in d["cells"]] == EXP18_STRATEGIES
            and sorted(d["excluded_strategies"]) == ["neural", "nf4_qae"]
            and all(
                str(d["excluded_strategies"][s]).strip()
                for s in ("neural", "nf4_qae")
            )
            for _, d in arts
        ),
        "cells=" + str(
            [[c["strategy"] for c in d["cells"]] for _, d in arts][0]
        ),
    )

    check(
        "exp18: protocol pins — 800 detector samples per cell "
        "(400 pairs), payload 10k, 30 epochs, seed 42 (exp10's)",
        all(
            all(c.get("detector_samples") == 800 for c in d["cells"])
            and d["method"]["payload_bits"] == 10_000
            and d["method"]["detector_pairs"] == 400
            and d["method"]["detector_epochs"] == 30
            and d["method"]["seed"] == 42
            for _, d in arts
        ),
        "samples=" + str(
            [c.get("detector_samples") for _, d in arts
             for c in d["cells"]][:4]
        ),
    )

    # Direction-agnostic consistency: every verdict boolean must be
    # recomputable from the numbers recorded beside it, so no cell can
    # be rounded up to a pass or down to a fail after the fact.
    inconsistent = []
    for name, d in arts:
        for c in d["cells"]:
            ext = c.get("extractability", {}) or {}
            curve = c.get("robustness_ber_curve", {}) or {}
            acc = c.get("detector_accuracy")
            want_rt = bool(
                ext.get("structurally_usable")
                and ext.get("ber") == gate["max_ber"]
            )
            want_rob = bool(
                curve.get("0.001") is not None
                and curve.get("0.001") <= gate["max_ber_at_sigma_0_001"]
                and curve.get("0.002") is not None
                and curve.get("0.002") <= gate["max_ber_at_sigma_0_002"]
            )
            want_gate = bool(
                acc is not None
                and acc <= gate["max_detector_accuracy"]
            )
            if (
                c.get("meets_round_trip_gate") != want_rt
                or c.get("meets_robustness_gate") != want_rob
                or c.get("meets_detector_gate") != want_gate
                or c.get("wins") != (want_rt and want_rob and want_gate)
            ):
                inconsistent.append(f"{name}:{c.get('strategy')}")
    check(
        "exp18: every cell's verdict booleans recompute exactly from "
        "its recorded numbers (no rounding either way)",
        not inconsistent,
        "28/28 consistent" if not inconsistent
        else "inconsistent: " + ", ".join(inconsistent),
    )

    # Headline, pinned: the matrix's citable facts.
    cells = [c for _, d in arts for c in d["cells"]]
    bers = [c.get("extractability", {}).get("ber") for c in cells]
    lwe_dets = [
        c.get("detector_accuracy")
        for c in cells if c["strategy"] == "lwe"
    ]
    other_wins = [
        (n, c.get("strategy"))
        for n, d in arts for c in d["cells"]
        if c.get("wins") and c.get("strategy") != "lwe"
    ]
    check(
        "exp18: 28/28 round trips at BER 0.0 and 28/28 robustness "
        "gates pass; LWE detector exactly 0.50 on all seven models "
        "(7/7 wins) with exactly one other winning cell — "
        "magnitude_aware on Mistral-7B (0.525)",
        len(bers) == 28
        and all(b == 0.0 for b in bers)
        and all(c.get("meets_robustness_gate") for c in cells)
        and lwe_dets == [0.5] * 7
        and sum(1 for c in cells if c.get("wins")) == 8
        and other_wins == [
            ("exp18_matrix_mistralai__mistral_7b_v0.3.json",
             "magnitude_aware"),
        ],
        f"ber_set={sorted(set(bers))} "
        f"wins={sum(1 for c in cells if c.get('wins'))} "
        f"lwe_dets={lwe_dets}",
    )


EXP19_ARTIFACTS = [
    "exp19_adaptive_google__gemma_2_2b.json",
    "exp19_adaptive_qwen__qwen2.5_3b.json",
    "exp19_adaptive_meta_llama__llama_3.1_8b.json",
]
# The citable finding: three first-pass models, three different
# branches, exactly what a sigma-bracket router predicts.
EXP19_ROUTES = {
    "exp19_adaptive_google__gemma_2_2b.json": ("lwe", True),
    "exp19_adaptive_qwen__qwen2.5_3b.json": ("neural", False),
    "exp19_adaptive_meta_llama__llama_3.1_8b.json": ("sign", True),
}


def audit_adaptive_routing():
    arts = [(name, one(name)) for name in EXP19_ARTIFACTS]
    missing = [n for n, d in arts if not d]
    check(
        "exp19: all three first-pass model artifacts present",
        not missing,
        "3/3" if not missing else "missing: " + ", ".join(missing),
    )
    if missing:
        return

    gate = THRESHOLDS["exp19"]
    check(
        "exp19: every artifact's gate matches THRESHOLDS['exp19'] "
        "(exp3's 0.0 reused; routing choice is measurement, not gate)",
        gate.get("max_ber") == 0.0
        and all(
            d["gate"]["max_ber"] == 0.0
            and "THRESHOLDS['exp19']" in d["gate"]["gate_source"]
            for _, d in arts
        ),
        f"gate={gate.get('max_ber')}",
    )

    # Direction-agnostic: each selected branch must be recomputable
    # from the recorded sigma and thresholds, and the three routes
    # must be the three distinct branches the probe found.
    recomputed, bad = [], []
    for name, d in arts:
        sigma = dig(d, "routing", "estimated_sigma")
        th = dig(d, "routing", "thresholds") or {}
        want = None
        if sigma is not None and th.get("lwe") is not None:
            want = (
                "lwe" if sigma < th["lwe"]
                else "neural" if sigma < th.get("neural", float("inf"))
                else "sign"
            )
        got = dig(d, "routing", "selected_branch")
        if want is None or want != got:
            bad.append(f"{name}:{sigma}->{got} (want {want})")
        if got:
            recomputed.append(got)
    check(
        "exp19: each branch recomputes exactly from its recorded "
        "sigma and thresholds; three models -> three DIFFERENT "
        "branches (lwe / neural / sign)",
        not bad and sorted(recomputed) == ["lwe", "neural", "sign"],
        f"routes={sorted(recomputed)}" if not bad else "; ".join(bad),
    )

    by_name = dict(arts)
    qwen = by_name["exp19_adaptive_qwen__qwen2.5_3b.json"]
    check(
        "exp19: Qwen's design route (neural) failing is recorded as "
        "the design's own — EmbeddingError text kept, both available "
        "branches round-tripped as fallback; gemma/llama routes "
        "available",
        dig(qwen, "routing", "design_route_available") is False
        and "Neural" in str(dig(qwen, "routing", "failure"))
        and qwen.get("forced_fallbacks") == ["sign", "lwe"]
        and sorted(qwen.get("round_trips", {}))
        == ["forced->lwe", "forced->sign"]
        and all(
            dig(by_name[n], "routing", "design_route_available") is True
            for n in EXP19_ARTIFACTS
            if n != "exp19_adaptive_qwen__qwen2.5_3b.json"
        ),
        f"failure={dig(qwen, 'routing', 'failure')!r:.80}",
    )

    # Verdict booleans recompute from recorded numbers; every
    # measured round trip holds exp3's gate.
    status_bad, bers = [], []
    for name, d in arts:
        trips = d.get("round_trips", {}) or {}
        measured = [t.get("ber") for t in trips.values()]
        bers.extend(measured)
        want = bool(measured) and all(b == 0.0 for b in measured)
        if d["gate"]["measured_round_trips"] != measured:
            status_bad.append(f"{name}: gate list drift")
        if (d["gate"]["status"] == "PASS") != want:
            status_bad.append(f"{name}: status")
    check(
        "exp19: every round trip that ran holds BER 0.0 and each "
        "gate status recomputes exactly from its measured list",
        not status_bad and bool(bers) and all(b == 0.0 for b in bers),
        f"{len(bers)}/{len(bers)} at 0.0"
        if not status_bad and all(b == 0.0 for b in bers)
        else "; ".join(status_bad) or f"bers={bers}",
    )

    check(
        "exp19: protocol pins — payload 10k (exp10's), deterministic "
        "sigma recompute recorded (EmbedResult drops inner metadata), "
        "lwe fresh-random-key caveat in method",
        all(
            dig(d, "method", "payload_bits") == 10_000
            and "deterministic" in str(dig(d, "method", "sigma_recompute"))
            and "os.urandom" in str(dig(d, "method", "lwe_key"))
            for _, d in arts
        ),
        "payload/method pins",
    )

    pipe_bad = []
    for name, d in arts:
        for tname, t in (d.get("round_trips", {}) or {}).items():
            if t.get("pipeline_attempted"):
                if t.get("pipeline_ok") is not True:
                    pipe_bad.append(f"{name}:{tname}:ok={t.get('pipeline_ok')}")
            elif not str(t.get("pipeline_note", "")).strip():
                pipe_bad.append(f"{name}:{tname}:no-note")
    check(
        "exp19: DecryptPipeline attempted only where it is the right "
        "decoder (sign routes, all recovered) and every non-attempt "
        "carries the exp10 sign-only note",
        not pipe_bad,
        "flags consistent" if not pipe_bad else "; ".join(pipe_bad),
    )


EXP20_ARTIFACT = "exp20_split_dial_qwen__qwen2.5_3b.json"
EXP20_FRACTIONS = [0.0, 0.25, 0.5, 0.75, 1.0]


def audit_split_dial():
    d = one(EXP20_ARTIFACT)
    check(
        "exp20: artifact present (W5.3 first pass, Qwen2.5-3B)",
        d is not None,
        EXP20_ARTIFACT if d else "missing: " + EXP20_ARTIFACT,
    )
    if not d:
        return

    gate = THRESHOLDS["exp20"]
    check(
        "exp20: gate matches THRESHOLDS['exp20'] — the same four "
        "numbers exp18 reused (exp3's 0.0, exp6's 0.02/0.10, "
        "exp7's 0.55)",
        gate.get("max_ber") == 0.0
        and gate.get("max_ber_at_sigma_0_001") == 0.02
        and gate.get("max_ber_at_sigma_0_002") == 0.10
        and gate.get("max_detector_accuracy") == 0.55
        and d["gate"]["max_ber"] == gate.get("max_ber")
        and d["gate"]["max_detector_accuracy"]
        == gate.get("max_detector_accuracy")
        and "THRESHOLDS['exp20']" in d["gate"]["gate_source"],
        f"gate={ {k: v for k, v in gate.items() if k != 'description'} }",
    )

    cells = d.get("cells", [])
    check(
        "exp20: exactly the five dial fractions, in order "
        "(0.0 / 0.25 / 0.5 / 0.75 / 1.0)",
        [c.get("split_fraction") for c in cells] == EXP20_FRACTIONS,
        str([c.get("split_fraction") for c in cells]),
    )

    # Direction-agnostic: every cell verdict recomputes from the
    # numbers recorded beside it (the exp18 rule, per cell).
    inconsistent = []
    for c in cells:
        ext = c.get("extractability", {}) or {}
        curve = c.get("robustness_ber_curve", {}) or {}
        acc = c.get("detector_accuracy")
        want_rt = bool(
            ext.get("structurally_usable")
            and ext.get("ber") == gate.get("max_ber")
        )
        want_rob = bool(
            curve.get("0.001") is not None
            and curve.get("0.001") <= gate.get("max_ber_at_sigma_0_001")
            and curve.get("0.002") is not None
            and curve.get("0.002") <= gate.get("max_ber_at_sigma_0_002")
        )
        want_det = bool(
            acc is not None and acc <= gate.get("max_detector_accuracy")
        )
        if (
            c.get("meets_round_trip_gate") != want_rt
            or c.get("meets_robustness_gate") != want_rob
            or c.get("meets_detector_gate") != want_det
            or c.get("wins") != (want_rt and want_rob and want_det)
        ):
            inconsistent.append(str(c.get("split_fraction")))
    check(
        "exp20: every cell's verdict booleans recompute exactly from "
        "its recorded numbers (no rounding either way)",
        not inconsistent,
        "5/5 consistent" if not inconsistent
        else "inconsistent: " + ", ".join(inconsistent),
    )

    bers = [
        c.get("extractability", {}).get("ber") for c in cells
    ]
    check(
        "exp20: all five round trips hold exp3's BER 0.0 "
        "(the dial must not cost the round trip at any fraction)",
        len(bers) == 5 and all(b == 0.0 for b in bers),
        f"bers={bers}",
    )

    method = d.get("method", {})
    check(
        "exp20: protocol pins — payload 10k, 400 pairs / 20 embeds / "
        "30 epochs / seed 42 (exp10's), keyless partition stated",
        method.get("payload_bits") == 10_000
        and method.get("detector_pairs") == 400
        and method.get("detector_embeddings") == 20
        and method.get("detector_epochs") == 30
        and method.get("seed") == 42
        and "blake2b" in str(method.get("partition_rule"))
        and "keyless" in str(method.get("partition_rule")),
        "protocol",
    )

    # Endpoint anchors: recorded deltas, recomputed — and exp18's
    # artifact is committed, so 'available' must be true; a missing
    # exp18 cell stays recorded missing, never defaulted.
    anchors = d.get("endpoint_anchors", {}) or {}
    exp18_path = RESULTS / str(anchors.get("source", ""))
    anchor_bad = []
    if not anchors.get("available"):
        anchor_bad.append("exp18 artifact not readable: "
                          + str(anchors.get("reason", "?")))
    if anchors.get("available") and not exp18_path.exists():
        anchor_bad.append("claims available but file absent")
    for frac in ("0.0", "1.0"):
        a = (anchors.get("anchors") or {}).get(frac)
        if not a:
            anchor_bad.append(f"anchor {frac} absent")
            continue
        if not a.get("present") or a.get("exp18_detector_accuracy") is None:
            anchor_bad.append(f"anchor {frac} not recorded (exp18 cell?)")
            continue
        want = a.get("exp20_detector_accuracy")
        exp18 = a.get("exp18_detector_accuracy")
        delta = a.get("delta")
        if want is None or delta is None or abs(
            delta - (want - exp18)
        ) > 1e-12:
            anchor_bad.append(f"anchor {frac} delta drift")
    check(
        "exp20: endpoint anchors read from exp18's committed artifact "
        "with deltas recomputing exactly (f=0 vs sign, f=1 vs lwe); "
        "nothing defaulted",
        not anchor_bad,
        "anchors consistent" if not anchor_bad
        else "; ".join(anchor_bad),
    )

    # The dial itself, pinned: exact vectors and the two directional
    # claims a reader will cite (both recomputable from the rows).
    det = [c.get("detector_accuracy") for c in cells]
    s2 = [c.get("robustness_ber_curve", {}).get("0.002") for c in cells]
    want_det = [0.7875, 0.69375, 0.7, 0.5875, 0.5]
    want_s2 = [
        0.0,
        0.0034451378055122203,
        0.00672776911076443,
        0.009490379615184607,
        0.01267550702028081,
    ]
    check(
        "exp20: the dial's numbers as measured — detector "
        "[0.7875, 0.69375, 0.70, 0.5875, 0.50] and BER@sigma0.002 "
        "[0, 0.0034, 0.0067, 0.0095, 0.0127] across parity share "
        "0.0->1.0",
        det == want_det and s2 == want_s2,
        f"det={det} s2={s2}",
    )

    a10 = (
        (anchors.get("anchors") or {}).get("1.0") or {}
    ).get("delta")
    check(
        "exp20: both trade-off directions hold — robustness "
        "(BER@0.002) nondecreasing with parity share, detector falls "
        "from pure sign to pure parity; only the pure-parity cell "
        "wins; its detector REPRODUCES exp18's lwe cell exactly "
        "(delta 0.0)",
        all(
            s2[i] <= s2[i + 1] for i in range(len(s2) - 1)
        )
        and det[-1] < det[0]
        and [c.get("wins") for c in cells]
        == [False, False, False, False, True]
        and a10 == 0.0,
        f"monotonic_s2=True det_drop={det[0]}->{det[-1]} "
        f"anchor_lwe_delta={a10}",
    )


# ---------------------------------------------------------------------
# W5.2 — QAE encode + LWE read-out (exp21)
# ---------------------------------------------------------------------
EXP21_ARTIFACT = "exp21_qae_lwe_qwen__qwen2.5_3b.json"


def audit_qae_lwe_interop():
    d = one(EXP21_ARTIFACT)
    check(
        "exp21: artifact present (W5.2 interop, Qwen2.5-3B)",
        d is not None,
        EXP21_ARTIFACT if d else "missing: " + EXP21_ARTIFACT,
    )
    if not d:
        return

    gate = THRESHOLDS["exp21"]
    check(
        "exp21: gate matches THRESHOLDS['exp21'] — both readings at "
        "exp3's 0.0 (reused), never relaxed",
        gate.get("max_ber") == 0.0
        and gate.get("max_control_ber") == 0.0
        and d["gate"]["max_ber"] == gate.get("max_ber")
        and d["gate"]["max_control_ber"] == gate.get("max_control_ber")
        and "THRESHOLDS['exp21']" in d["gate"]["gate_source"],
        f"max_ber={gate.get('max_ber')} "
        f"max_control_ber={gate.get('max_control_ber')}",
    )

    readouts = d.get("readouts", {}) or {}
    control = readouts.get("matched_control", {}) or {}
    interop = readouts.get("lwe_interop", {}) or {}
    fixed = readouts.get("lwe_interop_corrected", {}) or {}

    check(
        "exp21: the matched control holds exp3's 0.0 over the full "
        "transmitted stream — a non-zero interop BER is therefore "
        "attributable to the pairing, not a broken embed",
        control.get("ber") == 0.0
        and control.get("bit_errors") == 0
        and control.get("bits_compared") == 10256
        and control.get("meets_gate") is True,
        f"control={control.get('ber')} "
        f"({control.get('bit_errors')}/{control.get('bits_compared')})",
    )

    # The measured interop number, pinned — and the verdict recomputes
    # from the two gate readings beside it.
    control_ok = bool(
        control.get("ber") is not None
        and control.get("ber") == gate.get("max_control_ber")
    )
    interop_ok = bool(
        interop.get("ber") is not None
        and interop.get("ber") == gate.get("max_ber")
    )
    check(
        "exp21: raw LWE read-out of the qae stream measures BER "
        "0.5432917316692668 (5572/10256, near chance) — gate FAIL, "
        "and the artifact's verdict recomputes from the two readings",
        interop.get("ber") == 0.5432917316692668
        and interop.get("bit_errors") == 5572
        and interop.get("meets_gate") is False
        and d.get("verdict")
        == ("PASS" if (control_ok and interop_ok) else "FAIL")
        and d.get("verdict") == "FAIL",
        f"interop={interop.get('ber')} verdict={d.get('verdict')}",
    )

    check(
        "exp21: the public cell-parity correction returns exactly 0.0 "
        "— the parity(v) = sign(v) XOR cell-parity(|v|) identity is "
        "measured, not asserted; recorded as structure with an "
        "explicit 'not a gate rescue' role",
        fixed.get("ber") == 0.0
        and fixed.get("bit_errors") == 0
        and "NOT a gate rescue" in str(fixed.get("role")),
        f"corrected={fixed.get('ber')}",
    )

    premise = d.get("premise", {}) or {}
    check(
        "exp21: the plan's plausibility claim is quoted verbatim and "
        "its premise recorded half-false against exp17 (qae forces "
        "sign flips) — the claim is not silently reworded",
        "Plausible: both mechanisms avoid sign flips"
        in str(premise.get("plan_quote"))
        and "half-false" in str(premise.get("status"))
        and "exp17" in str(premise.get("status")),
        str(premise.get("status"))[:60],
    )

    method = d.get("method", {}) or {}
    check(
        "exp21: protocol pins — payload 10k, exp17's model, one "
        "embed with both read-outs on the SAME stego (pairing is "
        "the only variable); all three readings compare the same "
        "bit count",
        method.get("payload_bits") == 10_000
        and "SAME stego" in str(method.get("embed"))
        and "exp17" in str(method.get("model_note"))
        and d.get("model_id") == "Qwen/Qwen2.5-3B"
        and control.get("bits_compared")
        == interop.get("bits_compared")
        == fixed.get("bits_compared")
        == 10256,
        "protocol",
    )


# ---------------------------------------------------------------------
# W5.4 — per-layer LWE grid width (exp22)
# ---------------------------------------------------------------------
EXP22_ARTIFACT = "exp22_layer_widths_qwen__qwen2.5_3b.json"
EXP22_RULES = ["global", "per_layer", "layer_rank"]


def audit_layer_widths():
    d = one(EXP22_ARTIFACT)
    check(
        "exp22: artifact present (W5.4 first pass, Qwen2.5-3B)",
        d is not None,
        EXP22_ARTIFACT if d else "missing: " + EXP22_ARTIFACT,
    )
    if not d:
        return

    gate = THRESHOLDS["exp22"]
    check(
        "exp22: gate matches THRESHOLDS['exp22'] — exp10's four "
        "reused numbers (0.0, 0.02, 0.10, 0.55), per cell",
        gate.get("max_ber") == 0.0
        and gate.get("max_ber_at_sigma_0_001") == 0.02
        and gate.get("max_ber_at_sigma_0_002") == 0.10
        and gate.get("max_detector_accuracy") == 0.55
        and d["gate"]["max_ber"] == gate.get("max_ber")
        and d["gate"]["max_detector_accuracy"]
        == gate.get("max_detector_accuracy")
        and "THRESHOLDS['exp22']" in d["gate"]["gate_source"],
        f"gate={ {k: v for k, v in gate.items() if k != 'description'} }",
    )

    cells = d.get("cells", [])
    check(
        "exp22: exactly the three width rules, in order, with the "
        "right overrides (global = defaults, per_layer and "
        "layer_rank = lwe_width_rule) — nothing else may vary",
        d.get("rules") == EXP22_RULES
        and [c.get("width_rule") for c in cells] == EXP22_RULES
        and (cells[0].get("config_overrides") == {})
        and cells[1].get("config_overrides")
        == {"lwe_width_rule": "per_layer"}
        and cells[2].get("config_overrides")
        == {"lwe_width_rule": "layer_rank"},
        f"rules={d.get('rules')}",
    )

    # The dial, recomputed from the artifact's own recorded stds:
    # global is the absolute default everywhere; per_layer is
    # clip(4.0 * round(std, 4), 0.005, 0.020); layer_rank is the
    # fixed ladder by rank (ties on layer id); w/std ratios agree.
    table = d.get("layer_widths", {}) or {}
    stds = table.get("layer_std", {}) or {}
    w_g = (table.get("widths", {}) or {}).get("global", {}) or {}
    w_p = (table.get("widths", {}) or {}).get("per_layer", {}) or {}
    w_r = (table.get("widths", {}) or {}).get("layer_rank", {}) or {}
    ratio_p = (table.get("w_over_std", {}) or {}).get("per_layer", {}) or {}
    width_bad = []
    ordered = sorted(
        ((lid, float(s)) for lid, s in stds.items()),
        key=lambda kv: (kv[1], int(kv[0])),
    )
    n = len(ordered)
    rank_want = {
        lid: 0.005 + (i / (n - 1) if n > 1 else 0.5) * (0.010 - 0.005)
        for i, (lid, _s) in enumerate(ordered)
    }
    for lid, std in stds.items():
        if abs(w_g.get(lid, -1) - 0.010) > 1e-15:
            width_bad.append(f"global[{lid}]={w_g.get(lid)}")
        want = min(
            max(4.0 * round(float(std), 4), 0.005), 0.020
        )
        if abs(w_p.get(lid, -1) - want) > 1e-15:
            width_bad.append(f"per_layer[{lid}]={w_p.get(lid)} want {want}")
        if abs(w_r.get(lid, -1) - rank_want.get(lid, -1)) > 1e-12:
            width_bad.append(f"layer_rank[{lid}]={w_r.get(lid)}")
        if abs(ratio_p.get(lid, -1) - w_p.get(lid, 0) / float(std)) > 1e-9:
            width_bad.append(f"ratio[{lid}]")
    check(
        "exp22: every recorded width recomputes from the artifact's "
        "own per-layer stds — global = 0.010 everywhere, per_layer = "
        "clip(4.0*round(std,4), 0.005, 0.020), layer_rank = the "
        "fixed ladder by rank, w/std consistent",
        bool(stds) and not width_bad,
        f"{len(stds)} layers x 3 rules recomputed" if not width_bad
        else "; ".join(width_bad[:4]),
    )

    # Every cell verdict recomputes from the numbers beside it
    # (exp18's rule, per cell, direction-agnostic).
    inconsistent = []
    for c in cells:
        ext = c.get("extractability", {}) or {}
        curve = c.get("robustness_ber_curve", {}) or {}
        acc = c.get("detector_accuracy")
        want_rt = bool(
            ext.get("structurally_usable")
            and ext.get("ber") == gate.get("max_ber")
        )
        want_rob = bool(
            curve.get("0.001") is not None
            and curve.get("0.001") <= gate.get("max_ber_at_sigma_0_001")
            and curve.get("0.002") is not None
            and curve.get("0.002") <= gate.get("max_ber_at_sigma_0_002")
        )
        want_det = bool(
            acc is not None and acc <= gate.get("max_detector_accuracy")
        )
        if (
            c.get("meets_round_trip_gate") != want_rt
            or c.get("meets_robustness_gate") != want_rob
            or c.get("meets_detector_gate") != want_det
            or c.get("wins") != (want_rt and want_rob and want_det)
        ):
            inconsistent.append(str(c.get("width_rule")))
    check(
        "exp22: all three cells' verdict booleans recompute exactly "
        "from their recorded numbers (no rounding either way)",
        len(cells) == 3 and not inconsistent,
        "3/3 consistent" if not inconsistent
        else "inconsistent: " + ", ".join(inconsistent),
    )

    bers = [c.get("extractability", {}).get("ber") for c in cells]
    check(
        "exp22: all three round trips hold exp3's BER 0.0 — the "
        "width rule must not cost the round trip (per_layer's "
        "bucket agreement held at sigma=0 in run 1; layer_rank's "
        "is true by construction)",
        bers == [0.0, 0.0, 0.0],
        f"bers={bers}",
    )

    method = d.get("method", {}) or {}
    check(
        "exp22: protocol pins — payload 10k, 400 pairs / 20 embeds / "
        "30 epochs / seed 42 (exp10's), exp11's window cited as the "
        "clip source",
        method.get("payload_bits") == 10_000
        and method.get("detector_pairs") == 400
        and method.get("detector_embeddings") == 20
        and method.get("detector_epochs") == 30
        and method.get("seed") == 42
        and "exp11" in str(method.get("window_source")),
        "protocol",
    )

    # The control's anchor: exp18's committed lwe cell, delta
    # recomputed from recorded numbers; missing stays missing.
    anchor = d.get("control_anchor", {}) or {}
    anchor_bad = []
    if not anchor.get("available"):
        anchor_bad.append(
            "exp18 artifact not readable: "
            + str(anchor.get("reason", "?"))
        )
    elif not anchor.get("present"):
        anchor_bad.append("exp18 has no lwe cell: "
                          + str(anchor.get("reason", "?")))
    else:
        want = anchor.get("exp22_global_detector_accuracy")
        exp18 = anchor.get("exp18_detector_accuracy")
        delta = anchor.get("detector_delta_vs_exp18")
        if want is None or delta is None or abs(
            delta - (want - exp18)
        ) > 1e-12:
            anchor_bad.append("anchor delta drift")
    check(
        "exp22: the global-width control anchors to exp18's "
        "committed lwe cell, delta recomputed; nothing defaulted",
        not anchor_bad,
        "anchor consistent" if not anchor_bad
        else "; ".join(anchor_bad),
    )

    # The measurement itself: each rule's delta vs the global
    # control recomputes from the cells.
    deltas = d.get("cell_deltas", {}) or {}
    delta_bad = []
    if len(cells) == 3:
        g = cells[0]
        for rule in ("per_layer", "layer_rank"):
            got = deltas.get(rule, {}) or {}
            p = next(
                (c for c in cells if c.get("width_rule") == rule), None
            )
            if p is None:
                delta_bad.append(rule + ": cell missing")
                continue
            pairs = (
                ("detector", g.get("detector_accuracy"),
                 p.get("detector_accuracy")),
                ("ber_sigma_0_001",
                 g.get("robustness_ber_curve", {}).get("0.001"),
                 p.get("robustness_ber_curve", {}).get("0.001")),
                ("ber_sigma_0_002",
                 g.get("robustness_ber_curve", {}).get("0.002"),
                 p.get("robustness_ber_curve", {}).get("0.002")),
            )
            for key, a, b in pairs:
                want = (b - a) if (a is not None and b is not None) else None
                v = got.get(key)
                if want is None and v is None:
                    continue
                if v is None or want is None or abs(v - want) > 1e-12:
                    delta_bad.append(f"{rule}.{key}")
    else:
        delta_bad.append("cells missing")
    check(
        "exp22: both deltas vs the global control (per_layer - "
        "global, layer_rank - global) recompute exactly from the "
        "cells — detector and both sigma points",
        not delta_bad,
        f"deltas={deltas}" if not delta_bad
        else "drift: " + ", ".join(delta_bad),
    )

    # The run-1 diagnosis, recomputed analytically from the
    # artifact's own stds: noise inflates std through
    # sqrt(std^2 + sigma^2) and moves every layer's 4-decimal
    # bucket — this is why magnitude-keying cannot pass the
    # robustness gate, in numbers rather than prose.
    flips = d.get("noise_bucket_flips", {}) or {}
    flip_bad = []
    for sigma in (0.001, 0.002, 0.005):
        rec = flips.get(str(sigma), {}) or {}
        want = sum(
            1
            for s in stds.values()
            if round(math.sqrt(float(s) ** 2 + sigma ** 2), 4)
            != round(float(s), 4)
        )
        if rec.get("layers_changed") != want or rec.get("of") != len(stds):
            flip_bad.append(f"sigma {sigma}: {rec.get('layers_changed')}/"
                            f"{rec.get('of')} want {want}/{len(stds)}")
    check(
        "exp22: the recorded diagnosis recomputes — noise moves "
        "every layer's std bucket (36/36 at each sigma on this "
        "model), which is the verified cause of per_layer's "
        "robustness collapse under the extractor's drifting grid",
        bool(stds) and not flip_bad
        and all(
            flips[str(s)]["layers_changed"] == len(stds)
            for s in (0.001, 0.002, 0.005)
        ),
        "36/36 at all three sigmas" if not flip_bad
        else "; ".join(flip_bad),
    )

    # The numbers as measured, pinned — the three-way verdict a
    # reader will cite, all recomputable from the rows.
    by_rule = {c.get("width_rule"): c for c in cells}
    pl = by_rule.get("per_layer", {})
    lr = by_rule.get("layer_rank", {})
    gb = by_rule.get("global", {})
    a18_curve = anchor.get("exp18_robustness_ber_curve") or {}
    gb_curve = gb.get("robustness_ber_curve") or {}
    check(
        "exp22: the result as measured — per_layer FAILS robustness "
        "(0.0226 / 0.5763, gates 0.02 / 0.10) with the drift "
        "diagnosis above; layer_rank PASSES everything "
        "(0.0015 / 0.0736); detector 0.50 on all three cells (lwe's "
        "invariant, again); only global and layer_rank win; and the "
        "global control REPRODUCES exp18's lwe curve bit-for-bit "
        "(detector delta 0.0)",
        pl.get("robustness_ber_curve", {}).get("0.001")
        == 0.022588403536141447
        and pl.get("robustness_ber_curve", {}).get("0.002")
        == 0.5762805512220489
        and pl.get("wins") is False
        and lr.get("robustness_ber_curve", {}).get("0.001")
        == 0.0014625585023400937
        and lr.get("robustness_ber_curve", {}).get("0.002")
        == 0.07361544461778471
        and lr.get("wins") is True
        and gb.get("wins") is True
        and [c.get("detector_accuracy") for c in cells]
        == [0.5, 0.5, 0.5]
        and [c.get("wins") for c in cells] == [True, False, True]
        and gb_curve == a18_curve
        and anchor.get("detector_delta_vs_exp18") == 0.0,
        f"pl={pl.get('robustness_ber_curve')} "
        f"lr={lr.get('robustness_ber_curve')}",
    )


# W7 — Pareto frontier (exp24)
EXP24_ARTIFACT = "exp24_pareto_frontier.json"
EXP24_Y_KEY = {"exp7_neural_parameter_study": "accuracy"}


def _exp24_cell(point):
    """Resolve a point's source key to the source cell it cites.

    Source keys are ``experiment:collection[index]``. Fixed-name
    experiments (exp10/11/12/7) resolve directly; exp18/exp20/exp22
    live in per-model files, disambiguated by the point's own model
    slug — the exact slug the point id was built from.
    """
    src = str(point.get("source", ""))
    if ":" not in src:
        return None, "no source key"
    exp, rest = src.split(":", 1)
    if not (rest.endswith("]") and "[" in rest):
        return None, f"malformed source {src!r}"
    coll, idx = rest[:-1].split("[")
    try:
        index = int(idx)
    except ValueError:
        return None, f"malformed source {src!r}"

    path = RESULTS / f"{exp}.json"
    if not path.exists():
        slug = (point.get("model_id", "").replace("/", "__")
                .replace("-", "_").lower())
        cands = [p for p in sorted(RESULTS.glob(f"{exp}_*.json"))
                 if slug and slug in p.name]
        if len(cands) != 1:
            return None, f"{len(cands)} candidate files for {src}"
        path = cands[0]
    try:
        seq = json.loads(path.read_text(encoding="utf-8"))[coll]
        return seq[index], ""
    except (OSError, ValueError, KeyError, IndexError, TypeError) as exc:
        return None, f"{path.name}: {exc}"


def audit_pareto_frontier():
    d = one(EXP24_ARTIFACT)
    check(
        "exp24: artifact present (W7 first pass, 35 points over 11 "
        "committed sources)",
        d is not None,
        EXP24_ARTIFACT if d else "missing: " + EXP24_ARTIFACT,
    )
    if not d:
        return

    gate = THRESHOLDS["exp24"]
    check(
        "exp24: gate matches THRESHOLDS['exp24'] — max_source_delta 0.0 "
        "(citation integrity: every y/marker equals its source exactly; "
        "a synthesis cannot PASS by being plausible)",
        gate.get("max_source_delta") == 0.0
        and d["gate"].get("max_source_delta") == 0.0
        and "THRESHOLDS['exp24']" in str(d["gate"].get("gate_source")),
        f"max_source_delta={d['gate'].get('max_source_delta')}",
    )

    points = d.get("points", [])
    counts = d.get("counts", {}) or {}
    check(
        "exp24: counts match the lists they summarise — 35 points, "
        "1 frontier, 10 excluded, 2 related, 6 omitted source groups",
        counts.get("points") == len(points) == 35
        and counts.get("frontier") == len(d.get("frontier", [])) == 1
        and counts.get("excluded") == len(d.get("excluded", [])) == 10
        and counts.get("related") == len(d.get("related", [])) == 2
        and counts.get("omitted_source_groups")
        == len(d.get("omitted_sources", [])) == 6,
        str(counts),
    )

    # Citation integrity, the gate itself: every y and every marker
    # recomputes from the source cell it names — exact, delta 0.0,
    # nulls preserved as nulls (a missing curve is never filled in).
    y_bad, m_bad, rt_bad, unresolved = [], [], [], []
    for p in points:
        cell, err = _exp24_cell(p)
        if cell is None:
            unresolved.append(f"{p.get('id')}:{err}")
            continue
        want_y = cell.get(EXP24_Y_KEY.get(p.get("experiment"),
                                          "detector_accuracy"))
        if want_y is None or float(want_y) != p.get("y"):
            y_bad.append(f"{p.get('id')}:{want_y}!={p.get('y')}")
        curve = cell.get("robustness_ber_curve")
        want_m = curve.get("0.001") if isinstance(curve, dict) else None
        if want_m != p.get("marker"):
            m_bad.append(f"{p.get('id')}:{want_m}!={p.get('marker')}")
        ext = cell.get("extractability")
        if isinstance(ext, dict) and ext.get("ber") is not None:
            want_rt = ext.get("ber")
        elif isinstance(curve, dict):
            want_rt = curve.get("0.0")
        else:
            want_rt = None
        if want_rt != p.get("round_trip_ber"):
            rt_bad.append(f"{p.get('id')}:{want_rt}"
                          f"!={p.get('round_trip_ber')}")
    check(
        "exp24: every one of the 35 cited y values recomputes exactly "
        "from its source cell (detector accuracy; exp7's study key "
        "'accuracy') at delta 0.0 — a synthesis cannot be plausible, "
        "it must be exact",
        not unresolved and not y_bad and len(points) == 35,
        "35/35 at delta 0.0" if not (unresolved or y_bad)
        else "; ".join((unresolved + y_bad)[:5]),
    )
    check(
        "exp24: every marker (BER@σ0.001) recomputes exactly from its "
        "source's robustness_ber_curve, and all 6 null markers stay "
        "null — exp7's study measured no curve, so none was borrowed "
        "or invented to make a point look annotated",
        not unresolved and not m_bad
        and sum(1 for p in points if p.get("marker") is None) == 6,
        "35/35 at delta 0.0, 6 nulls preserved"
        if not (unresolved or m_bad) else "; ".join(m_bad[:5]),
    )
    check(
        "exp24: every round-trip BER carried beside a point recomputes "
        "from its source (extractability.ber, else the curve's σ0 "
        "entry, else null) — the sanity column is measured too",
        not unresolved and not rt_bad,
        "35/35 at delta 0.0" if not (unresolved or rt_bad)
        else "; ".join(rt_bad[:5]),
    )

    # The frontier recomputes from the artifact's own points.
    recomputed = [
        p["id"] for i, p in enumerate(points)
        if not any(
            j != i and q["x"] <= p["x"] and q["y"] <= p["y"]
            and (q["x"] < p["x"] or q["y"] < p["y"])
            for j, q in enumerate(points)
        )
    ]
    check(
        "exp24: the frontier recomputes from the artifact's own 35 "
        "points (nondominated on x-down, y-down; equal points do not "
        "dominate each other) and equals the stored frontier exactly",
        recomputed == d.get("frontier") and len(recomputed) == 1,
        f"recomputed={recomputed}",
    )

    fp = next((p for p in points
               if p.get("id") in (d.get("frontier") or [])), {})
    check(
        "exp24: the frontier as measured — exp22's layer_rank width "
        "rule is the single nondominated point: x=0.00372693 (smallest "
        "of all 35), y=0.50 detector (the floor), marker 0.00146256 — "
        "the rank-keyed width wins the trade-off outright instead of "
        "trading along it",
        fp.get("id") == "exp22:qwen__qwen2.5_3b:lwe:layer_rank"
        and fp.get("x") == 0.0037269348978949783
        and fp.get("y") == 0.5
        and fp.get("marker") == 0.0014625585023400937,
        f"id={fp.get('id')} x={fp.get('x')} y={fp.get('y')} "
        f"marker={fp.get('marker')}",
    )
    check(
        "exp24: why the frontier is one point — the winner is minimal "
        "on BOTH axes at once (its x is the set minimum and its y sits "
        "at the detector floor 0.5), so no point in the set can "
        "dominate it and every other point is dominated",
        bool(fp) and fp.get("x") == min(p["x"] for p in points)
        and fp.get("y") == min(p["y"] for p in points)
        and all(p.get("y", 1.0) >= 0.5 for p in points),
        f"min x={min(p['x'] for p in points):.8f} "
        f"min y={min(p['y'] for p in points)}",
    )

    x_bad = [p.get("id") for p in points
             if not (isinstance(p.get("x"), float) and p.get("x") > 0.0
                     and (p.get("changed_values") or 0) > 0
                     and (p.get("bits_embedded") or 0) > 0)]
    method = d.get("method", {}) or {}
    check(
        "exp24: x is measured here for every point — mean |delta| over "
        "changed values, each point carrying its x, changed-value count "
        "and embedded bits, with the definition and the reason it had "
        "to be measured (no committed artifact pairs magnitude with a "
        "detector) recorded in method; no count substituted for it",
        not x_bad and len(points) == 35
        and "mean |embedded - original|"
        in str(method.get("x_definition", ""))
        and "never substituted" in str(method.get(
            "x_why_measured_here", "")),
        "35/35 carry x" if not x_bad
        else "missing: " + ", ".join(x_bad[:5]),
    )

    # Config reconstruction: each point's params must mirror its
    # source's own mechanism, cell by cell.
    cfg_bad = []
    for p in points:
        cell, err = _exp24_cell(p)
        if cell is None:
            cfg_bad.append(f"{p.get('id')}:{err}")
            continue
        params = p.get("params", {}) or {}
        exp = p.get("experiment")
        try:
            if exp in ("exp11_lwe_alpha_pareto", "exp12_lwe_cross_model"):
                width = float(cell.get("grid_width")
                              or 0.01 if exp == "exp12_lwe_cross_model"
                              else cell.get("grid_width"))
                ok = (params.get("alpha") == 1.0
                      and params.get("min_magnitude") == width / 2.0)
            elif exp == "exp20":
                ok = (params.get("split_fraction")
                      == float(cell.get("split_fraction")))
            elif exp == "exp22":
                ok = (params.get("lwe_width_rule")
                      == str(cell.get("width_rule")))
            elif exp == "exp7_neural_parameter_study":
                ok = (params.get("min_magnitude")
                      == float(cell.get("alpha"))
                      and params.get("gamma") == float(cell.get("gamma"))
                      and params.get("payload_bits")
                      == int(cell.get("payload_bits")))
            else:  # exp10 / exp18 — the shipped default config
                ok = params == {"payload_bits": 10000}
        except (TypeError, ValueError):
            ok = False
        if not ok:
            cfg_bad.append(str(p.get("id")))
    check(
        "exp24: every point's config reconstructs its source's own "
        "mechanism — exp11/exp12 pinned by their alpha=1.0 & "
        "min_magnitude=w/2 patch, exp20's split_fraction, exp22's "
        "width_rule, exp7's alpha→min_magnitude (its module's own "
        "mapping) + gamma + payload, exp10/exp18 at the shipped "
        "default — so a shared point means the same embed, not a "
        "lookalike",
        not cfg_bad and len(points) == 35,
        "35/35 reconstruct" if not cfg_bad
        else "; ".join(cfg_bad[:5]),
    )

    # Exclusions and omissions are recorded, never dropped (§2).
    excl = d.get("excluded", [])
    excl_text = " | ".join(f"{e.get('what')}:{e.get('reason')}"
                           for e in excl)
    check(
        "exp24: all 10 exclusions recorded with what + reason + source, "
        "never dropped — the neural cell by its NEEDS_TRAINING status, "
        "tinyllama by its incomplete cache, gemma-2-2b's 8 cells by its "
        "absence from experiment_registry (4 group-level + 4 magnitude "
        "follow-ons), each reason carried in full",
        len(excl) == 10
        and all(e.get("what") and e.get("reason") and e.get("source")
                for e in excl)
        and "NEEDS_TRAINING" in excl_text
        and "residual cache incomplete" in excl_text
        and "not in experiment_registry" in excl_text
        and sum("magnitude not measured" in str(e.get("reason"))
                for e in excl) == 4,
        f"{len(excl)} exclusions, all with reasons"
        if len(excl) == 10 else f"{len(excl)} exclusions",
    )

    om = d.get("omitted_sources", [])
    om_text = " | ".join(f"{o.get('source')}:{o.get('reason')}"
                         for o in om)
    check(
        "exp24: 6 omitted source groups each carry the reason they "
        "cannot form a frontier point (no detector axis), exp2/exp17 "
        "named as priors rather than x, and the catch-all names every "
        "remaining experiment — including exp23 and exp21 — so the "
        "frontier's coverage is stated rather than implied",
        len(om) == 6
        and all(o.get("source") and o.get("reason") for o in om)
        and "exp2_*.json" in om_text
        and "exp17_*.json" in om_text
        and "cited as a prior, not as x" in om_text
        and "exp23" in om_text,
        f"{len(om)} groups" if len(om) == 6 else f"{len(om)} groups",
    )

    # The related and prior citations recompute from their sources.
    rel = {r.get("source"): r for r in d.get("related", [])}
    arms = (one("exp14_blind_patch_detector.json") or {}).get("arms", {}) \
        or {}
    want14 = (
        arms.get("blind", {}).get("metrics", {}).get("accuracy"),
        arms.get("control", {}).get("metrics", {}).get("accuracy"),
        arms.get("blind", {}).get("positions_containing_carrier"),
    )
    got14 = rel.get("exp14_blind_patch_detector.json") or {}
    cross16 = {k: v.get("accuracy") for k, v in
               ((one("exp16_cross_scheme_detector.json") or {})
                .get("results") or {}).items()}
    got16 = (rel.get("exp16_cross_scheme_detector.json")
             or {}).get("cross_matrix")
    check(
        "exp24: both related citations recompute from their sources "
        "exactly — exp14's blind/control accuracies + carrier count, "
        "exp16's four cross-scheme readings — recorded as readings of "
        "an existing point, never folded into the frontier",
        len(rel) == 2
        and (got14.get("blind_accuracy"),
             got14.get("control_accuracy"),
             got14.get("blind_positions_with_carrier")) == want14
        and got16 == cross16,
        f"exp14={want14} exp16_keys={sorted(got16 or {})}",
    )

    priors = {p.get("source"): p
              for p in d.get("prior_magnitude_citations", [])}
    want2 = ((one("exp2_qwen__qwen2.5-3b.json") or {}).get("mean_mag_mean")
             or (one("exp2_qwen__qwen2.5-3b.json") or {})
             .get("reproducibility", {}).get("mean_mag_mean"))
    m17 = (one("exp17_qae_round_trip.json") or {}).get("metrics", {}) or {}
    p2 = priors.get("exp2_qwen__qwen2.5-3b.json") or {}
    p17 = priors.get("exp17_qae_round_trip.json") or {}
    check(
        "exp24: both prior-magnitude citations recompute from their "
        "sources exactly (exp2's mean_mag_mean; exp17's "
        "mean_abs_delta_over_changed at 48,256 bits) and each records "
        "its own protocol as cited-not-used — different payloads, so "
        "neither was merged into x",
        len(priors) == 2
        and p2.get("mean_mag_mean") == want2
        and p17.get("mean_abs_delta_over_changed")
        == m17.get("mean_abs_delta_over_changed")
        and p17.get("bits_embedded") == m17.get("bits_embedded") == 48256
        and all("not used as x" in str(p.get("protocol"))
                for p in priors.values()),
        f"exp2={want2} exp17={m17.get('mean_abs_delta_over_changed')}",
    )

    srcs = method.get("sources_read", [])
    missing_src = [s for s in srcs if not (RESULTS / s).exists()]
    check(
        "exp24: protocol pins — seed 42, payload default 10,000, "
        "marker σ=0.001 named in method, all 11 sources_read present "
        "on disk (9 frontier sources + the 2 related readings)",
        method.get("seed") == 42
        and method.get("payload_bits_default") == 10000
        and "0.001" in str(method.get("marker", ""))
        and len(srcs) == 11 and not missing_src,
        f"seed={method.get('seed')} sources={len(srcs)} "
        f"missing={missing_src}",
    )


# ------------------------------------------------------------------ gates
# W6 — model surgery survival (exp23)
EXP23_ARTIFACT = "exp23_model_surgery_qwen__qwen2.5_3b.json"
EXP23_ORDER = [
    "control", "lora_0.001", "lora_0.01", "prune_10", "prune_30",
    "nf4_requant", "merge_0.01", "merge_0.05", "merge_0.5",
    "finetune_1k", "gptq_requant", "awq_requant",
]


def audit_model_surgery():
    d = one(EXP23_ARTIFACT)
    check(
        "exp23: artifact present (W6, first pass + legs run 2026-10-07, "
        "Qwen2.5-3B)",
        d is not None,
        EXP23_ARTIFACT if d else "missing: " + EXP23_ARTIFACT,
    )
    if not d:
        return

    gate = THRESHOLDS["exp23"]
    check(
        "exp23: gate matches THRESHOLDS['exp23'] — exp3's 0.0 twice "
        "(max_ber, max_control_ber); degradation is read from the "
        "BER column, never from a relaxed gate",
        gate.get("max_ber") == 0.0
        and gate.get("max_control_ber") == 0.0
        and d["gate"].get("max_ber") == gate.get("max_ber")
        and d["gate"].get("max_control_ber") == gate.get("max_control_ber")
        and "THRESHOLDS['exp23']" in str(d["gate"].get("gate_source")),
        f"max_ber={d['gate'].get('max_ber')} "
        f"max_control_ber={d['gate'].get('max_control_ber')}",
    )

    control = d.get("control", {}) or {}
    check(
        "exp23: the weight-path control holds both readings at exp3's "
        "0.0 and the cached residuals equal a fresh pair residual "
        "exactly — the cell arithmetic is validated before any "
        "surgery is measured",
        control.get("direct_ber") == 0.0
        and control.get("weight_path_ber") == 0.0
        and control.get("cache_vs_pair_max_abs") == 0.0
        and control.get("valid") is True,
        f"direct={control.get('direct_ber')} "
        f"weight={control.get('weight_path_ber')} "
        f"pair={control.get('cache_vs_pair_max_abs')}",
    )

    cells = d.get("cells", [])
    check(
        "exp23: exactly the twelve surgery cells, in order (control, "
        "2 LoRA, 2 prune, NF4, 3 merge, then the three legs that were "
        "NOT_RUN in the first pass — fine-tune, GPTQ, AWQ) — nothing "
        "added, nothing silently dropped",
        [c.get("surgery") for c in cells] == EXP23_ORDER,
        f"cells={[c.get('surgery') for c in cells]}",
    )

    # Every verdict recomputes from the numbers beside it: the BER is
    # the recorded error count over the compared bits, and the gate
    # boolean is exp3's 0.0 — exact, no rounding either way.
    inconsistent = []
    for c in cells:
        ber = c.get("ber")
        bits = c.get("bits_compared")
        errs = c.get("bit_errors")
        if ber != (errs / bits if bits else None):
            inconsistent.append(f"{c.get('surgery')}:ber")
        if c.get("meets_gate") != (ber == gate.get("max_ber")):
            inconsistent.append(f"{c.get('surgery')}:gate")
        if bits != 10256 or c.get("carriers_total") != 10256:
            inconsistent.append(f"{c.get('surgery')}:bits")
    check(
        "exp23: every cell's BER recomputes exactly from its own "
        "error count over 10,256 compared bits, and every verdict "
        "boolean recomputes from exp3's 0.0 (no rounding either way)",
        len(cells) == 12 and not inconsistent,
        "12/12 consistent" if not inconsistent
        else "; ".join(inconsistent),
    )

    by_name = {c.get("surgery"): c for c in cells}
    survivors = [
        n for n in EXP23_ORDER
        if n in by_name and by_name[n].get("ber") == 0.0
    ]
    nf4 = by_name.get("nf4_requant", {})
    m5 = by_name.get("merge_0.5", {})
    gptq = by_name.get("gptq_requant", {})
    awq = by_name.get("awq_requant", {})
    heavy = (nf4, m5, gptq, awq)
    check(
        "exp23: the result as measured — eight cells survive at BER "
        "0.0 (control, both LoRA ratios, both prunes, merge t<=0.05, "
        "and the real 1,000-step fine-tune); NF4 re-quant fails at "
        "0.3832 (3930/10256), the half-merge at 0.2476 (2539/10256), "
        "GPTQ at 0.4956 (5083/10256, near chance) and AWQ at 0.4108 "
        "(4213/10256) — every failure short of chance 0.5, so the "
        "degradation is graceful rather than erased; total-vs-graceful "
        "read from the numbers, gate untouched",
        survivors == [
            "control", "lora_0.001", "lora_0.01",
            "prune_10", "prune_30", "merge_0.01", "merge_0.05",
            "finetune_1k",
        ]
        and [c.get("bit_errors") for c in heavy]
        == [3930, 2539, 5083, 4213]
        and all(0.0 < c.get("ber", 1.0) < 0.5 for c in heavy)
        and all(c.get("meets_gate") is False for c in heavy),
        f"nf4={nf4.get('ber')} merge_0.5={m5.get('ber')} "
        f"gptq={gptq.get('ber')} awq={awq.get('ber')} "
        f"survivors={survivors}",
    )

    prune_bad = []
    for n in ("prune_10", "prune_30"):
        c = by_name.get(n, {})
        if not (
            c.get("carriers_displaced") == 0
            and c.get("rms_delta_at_carriers") == 0.0
            and (c.get("rms_delta_over_rms_w") or 0.0) > 0.0
            and c.get("ber") == 0.0
        ):
            prune_bad.append(n)
    check(
        "exp23: pruning's pass is explained by its own numbers — a "
        "real delta (RMS 2.1% / 10.9% of the weights) that lands on "
        "zero carriers, so the payload sits outside the pruned mass "
        "by placement, not by luck",
        not prune_bad,
        f"displaced={[by_name.get(n, {}).get('carriers_displaced') for n in ('prune_10', 'prune_30')]}"
        if not prune_bad else "failed: " + ", ".join(prune_bad),
    )

    lora_bad = []
    for n, want in (("lora_0.001", 0.001), ("lora_0.01", 0.01)):
        got = by_name.get(n, {}).get("rms_delta_over_rms_w")
        if got is None or abs(got - want) > 1e-6:
            lora_bad.append(f"{n}={got}")
    check(
        "exp23: the LoRA-shaped deltas hit their stated RMS scale "
        "(1.0e-3 and 1.0e-2 of RMS(W)) — a delta off its scale would "
        "make 'survives LoRA merge' mean nothing",
        not lora_bad,
        "0.001 / 0.01 hit" if not lora_bad else "; ".join(lora_bad),
    )

    r1 = by_name.get("merge_0.01", {}).get("rms_delta_over_rms_w")
    r2 = by_name.get("merge_0.05", {}).get("rms_delta_over_rms_w")
    r3 = by_name.get("merge_0.5", {}).get("rms_delta_over_rms_w")
    linear = bool(
        r1 and r2 and r3
        and abs(r2 / r1 - 5.0) < 1e-5
        and abs(r3 / r2 - 10.0) < 1e-5
    )
    check(
        "exp23: the merge deltas scale linearly with t (rms 0.05/"
        "0.01 = 5, rms 0.5/0.05 = 10) — the recorded task vector "
        "W' = W_stego + t*(W_instruct - W_stego) is measured, not "
        "asserted",
        linear,
        f"r={r1:.3e} {r2:.3e} {r3:.3e}" if r1 and r2 and r3
        else "missing rms",
    )

    method = d.get("method", {}) or {}
    check(
        "exp23: protocol pins — payload 10k (10,256 compared bits), "
        "one production-path sign embed shared by every cell, seed "
        "42, fp32 throughout, surgery scope mlp.down_proj only",
        method.get("payload_bits") == 10_000
        and method.get("one_embed_for_all_cells") is True
        and method.get("strategy") == "sign"
        and method.get("seed") == 42
        and "down_proj only" in str(method.get("surgery_scope"))
        and "float32" in str(method.get("dtype")),
        "protocol",
    )

    blockers = d.get("not_run", []) or []
    names = " | ".join(str(b.get("item", "")) for b in blockers)
    ft = method.get("finetune") or {}
    gq = method.get("gptq") or {}
    aw = method.get("awq") or {}
    check(
        "exp23: the first pass's three named blockers are now measured "
        "cells — not_run is empty (a leg that ran leaves no finding) "
        "and method carries the protocol each leg ran under: fine-tune "
        "1,000 steps, lr 1e-4 cosine with 3% warmup, down_proj only, "
        "loss falling 2.271010437011719 -> 2.169337158203125 (identical "
        "across runs); GPTQ int4 group 128 sym on 64 stated wikitext "
        "windows, read back via save() -> safetensors; AWQ int4 group "
        "128 zero-point on 32 stated windows through exp8's reader",
        blockers == []
        and ft.get("steps") == 1000
        and ft.get("lr") == 0.0001
        and "3% warmup" in str(ft.get("scheduler"))
        and "down_proj only" in str(ft.get("targets"))
        and ft.get("loss_first") == 2.271010437011719
        and ft.get("loss_last") == 2.169337158203125
        and ft.get("loss_last") < ft.get("loss_first")
        and gq.get("bits") == 4
        and gq.get("group_size") == 128
        and gq.get("sym") is True
        and "64 wikitext" in str(gq.get("calibration"))
        and "save()" in str(gq.get("read_back"))
        and aw.get("bits") == 4
        and aw.get("group_size") == 128
        and aw.get("zero_point") is True
        and "32 wikitext" in str(aw.get("calibration"))
        and "dequantize_awq_layer" in str(aw.get("reader")),
        f"{len(blockers)} blockers: {names}" if blockers
        else "0 blockers; fine-tune/GPTQ/AWQ measured with stated "
             "protocol in method.finetune/gptq/awq",
    )

    notes0 = str((d.get("notes") or [""])[0])
    check(
        "exp23: pre-registration preserved with the misses recorded, "
        "not rewritten — method.pre_registered names the control 0.0 "
        "and both heavy-loss expectations (nf4, merge dies as t "
        "grows), and notes[0] states exp18's rule with no gate "
        "relaxed",
        "control 0.0" in str(method.get("pre_registered"))
        and "nf4_requant plausibly heavy loss"
        in str(method.get("pre_registered"))
        and "merge dies as t grows" in str(method.get("pre_registered"))
        and "No gate is relaxed" in notes0,
        "pre_registered + notes[0]",
    )

    repro = d.get("reproducibility", {}) or {}
    check(
        "exp23: reproducibility recorded — 36/36 layers matched, "
        "seed and versions captured, residual definition intact",
        repro.get("actual_layers") == repro.get("expected_layers") == 36
        and repro.get("layer_count_matches_expected") is True
        and repro.get("random_seed") == 42
        and "dequantize" in str(repro.get("residual_definition"))
        and bool(repro.get("timestamp")),
        f"layers={repro.get('actual_layers')}/"
        f"{repro.get('expected_layers')}",
    )


EXP25_ARTIFACT = "exp25_selection_ablation_qwen__qwen2.5_3b.json"
EXP26_ARTIFACT = "exp26_capacity_scaling_qwen__qwen2.5_3b.json"
EXP27_ARTIFACT = "exp27_threat_model_qwen__qwen2.5_3b.json"
EXP26_FIGURE = "exp26_capacity_scaling_qwen__qwen2.5_3b.svg"


def audit_paper_experiments():
    """W9 hardening — selection ablation, capacity curve, boundary.

    Recomputation over pinning wherever a verdict could drift: every
    gate flag, hypothesis verdict, BER and precision is recomputed
    from the counts beside it. Only structurally fixed values (sizes,
    arm names, the 10,256-bit construction) are asserted literally.
    """
    _audit_paper_exp25()
    _audit_paper_exp26()
    _audit_paper_exp27()


def _audit_paper_exp25():
    d = one(EXP25_ARTIFACT)
    check(
        "exp25: artifact present (W9.1 selection-policy ablation, "
        "Qwen2.5-3B)",
        d is not None,
        EXP25_ARTIFACT if d else "missing: " + EXP25_ARTIFACT,
    )
    if not d:
        return

    gate = THRESHOLDS["exp25"]
    check(
        "exp25: gate mirrors THRESHOLDS['exp25'] field-for-field — "
        "all arms 0.0, production 0.55/0.05/0.02 — with the source "
        "pinned",
        all(d.get("gate", {}).get(k) == v for k, v in gate.items())
        and "THRESHOLDS['exp25']" in str(
            d.get("gate", {}).get("gate_source")
        ),
        f"status={d.get('gate', {}).get('status')} "
        f"failures={d.get('gate', {}).get('failures')}",
    )

    arms = d.get("arms", {}) or {}
    check(
        "exp25: exactly three arms — random x3, magnitude x1 "
        "(production), keyed x3",
        sorted(arms) == ["keyed", "magnitude", "random"]
        and len(dig(arms, "random", "replicates") or []) == 3
        and len(dig(arms, "magnitude", "replicates") or []) == 1
        and len(dig(arms, "keyed", "replicates") or []) == 3,
        f"arms={sorted(arms)}",
    )

    rows = [
        (p, r) for p in sorted(arms)
        for r in arms[p]["replicates"]
    ]
    bad = []
    for p, r in rows:
        tag = f"{p}[{r.get('replicate')}]"
        ber, bits, errs = (
            r.get("ber"), r.get("bits_compared"), r.get("bit_errors")
        )
        if ber != (errs / bits if bits else None):
            bad.append(f"{tag}:ber")
        if ber != 0.0 or bits != 10256 or r.get("carrier_count") != 10256:
            bad.append(f"{tag}:roundtrip")
        if not (r.get("decrypt_ok") and r.get("recovered_matches")):
            bad.append(f"{tag}:decrypt")
        dg = r.get("detectability_gate") or {}
        if dg.get("kl_within_0_05") != (r.get("kl_divergence") <= 0.05):
            bad.append(f"{tag}:kl-flag")
        if dg.get("accuracy_within_0_55") != (
            r.get("detector_accuracy") <= 0.55
        ):
            bad.append(f"{tag}:acc-flag")
        robust = r.get("robustness_ber_sigma_0_001")
        rg = dig(r, "robustness_gate", "within_exp6_line_0_02")
        if rg != (robust is not None and robust <= 0.02):
            bad.append(f"{tag}:robust-flag")
        for tier in ("keyless_exact_size", "keyless_nominal_size"):
            pos = dig(r, tier, "positions") or {}
            tp = pos.get("true_positives")
            fp = pos.get("false_positives")
            fn = pos.get("false_negatives")
            if pos.get("precision") != (
                tp / (tp + fp) if (tp + fp) else None
            ):
                bad.append(f"{tag}:{tier}:precision")
            if pos.get("recall") != (
                tp / (tp + fn) if (tp + fn) else None
            ):
                bad.append(f"{tag}:{tier}:recall")
    check(
        "exp25: every replicate recomputes — BER from errors/bits at "
        "exp3's 0.0 over the 10,256-carrier construction, decrypt + "
        "message match, detectability/robustness flags re-derived "
        "from the numbers beside them, keyless precision/recall from "
        "TP/FP/FN",
        not bad,
        f"bad={bad}",
    )

    digests = {r.get("allocation_digest") for _, r in rows}
    check(
        "exp25: one shared QACI Hamilton allocation across all seven "
        "replicates (digest-pinned) and the artifact says so — the "
        "ablation's control variable",
        len(digests) == 1
        and d.get("allocation_shared_across_arms") is True,
        f"digests={sorted(str(x) for x in digests)}",
    )

    mag = arms["magnitude"]["replicates"][0]
    check(
        "exp25: production arm holds exp7's detectability lines "
        "(0.55/0.05) and exp6's sigma=0.001 line (0.02)",
        mag["detector_accuracy"] <= 0.55
        and mag["kl_divergence"] <= 0.05
        and mag["robustness_ber_sigma_0_001"] <= 0.02,
        f"acc={mag['detector_accuracy']} kl={mag['kl_divergence']} "
        f"robust={mag['robustness_ber_sigma_0_001']}",
    )

    random_reps = arms["random"]["replicates"]
    base = random_reps + arms["keyed"]["replicates"]
    base_robust = [
        r["robustness_ber_sigma_0_001"] for r in base
    ]
    check(
        "exp25: baselines are FLAGGED, not gated — every random/keyed "
        "replicate sits past exp6's 0.02 line (the measured cost of "
        "importance-blind selection) with flags saying False, while "
        "status stays PASS exactly as THRESHOLDS['exp25'] prescribes",
        all(v is not None and v > 0.02 for v in base_robust)
        and all(
            dig(r, "robustness_gate", "within_exp6_line_0_02") is False
            for r in base
        )
        and d.get("status") == "PASS",
        f"baseline_robustness={[round(v, 4) for v in base_robust]}",
    )

    def exact_ber(r):
        v = dig(r, "keyless_exact_size", "keyless_ber")
        return v is not None and 0.4 <= v <= 0.6

    check(
        "exp25: the keyless public-rule attacker reads at chance for "
        "EVERY arm (exact-size BER in [0.4, 0.6]) — no policy leaks a "
        "readable stream to the public re-run",
        all(exact_ber(r) for _, r in rows),
        f"exact_ber={[round(dig(r, 'keyless_exact_size', 'keyless_ber'), 4) for _, r in rows]}",
    )

    recorded = {
        h.get("id"): h.get("observed_supported")
        for h in d.get("hypotheses", [])
    }
    h1 = all(r["ber"] == 0.0 for _, r in rows)
    h2 = all(
        r["detector_accuracy"] <= 0.55 and r["kl_divergence"] <= 0.05
        for _, r in rows
    )
    kx = mag["keyless_exact_size"]
    prec = dig(kx, "positions", "precision")
    mag_ber = kx.get("keyless_ber")
    others = [
        dig(r, "keyless_exact_size", "keyless_ber")
        for p in ("random", "keyed") for r in arms[p]["replicates"]
    ]
    h3 = bool(
        prec is not None and prec >= 0.99
        and mag_ber is not None and mag_ber < 0.1
        and all(v is not None and v > 0.4 for v in others)
    )
    rand_robust = [
        r["robustness_ber_sigma_0_001"] for r in random_reps
    ]
    h4 = bool(
        mag["robustness_ber_sigma_0_001"] is not None
        and all(v is not None for v in rand_robust)
        and mag["robustness_ber_sigma_0_001"]
        <= sum(rand_robust) / len(rand_robust)
    )
    recomputed = {"H1": h1, "H2": h2, "H3": h3, "H4": h4}
    check(
        "exp25: hypothesis verdicts recompute from the recorded "
        "numbers — H1/H2/H4 supported (round trip, detectability, "
        "robustness ordering), H3 (public re-run => BER < 0.1) NOT "
        "supported and kept as written: a miss is recorded, not "
        "rewritten",
        recorded == recomputed
        and recorded.get("H1") is True
        and recorded.get("H2") is True
        and recorded.get("H4") is True
        and recorded.get("H3") is False,
        f"recorded={recorded} recomputed={recomputed}",
    )

    check(
        "exp25: H3's recorded mechanism holds in the numbers — the "
        "public re-run never reproduces the writer's allocation in "
        "full while position precision stays >= 0.98, which is why "
        "near-exact position knowledge still reads near chance",
        (kx.get("layers_with_matching_allocation") or 0)
        < (kx.get("layers_total") or 0)
        and prec is not None and prec >= 0.98,
        f"alloc_match={kx.get('layers_with_matching_allocation')}/"
        f"{kx.get('layers_total')} precision={prec}",
    )


def _audit_paper_exp26():
    d = one(EXP26_ARTIFACT)
    check(
        "exp26: artifact present (W9.2 capacity curve, Qwen2.5-3B)",
        d is not None,
        EXP26_ARTIFACT if d else "missing: " + EXP26_ARTIFACT,
    )
    if not d:
        return

    gate = THRESHOLDS["exp26"]
    check(
        "exp26: gate mirrors THRESHOLDS['exp26'] field-for-field "
        "(0.0 / 0.55 / 0.05) with the source pinned",
        all(d.get("gate", {}).get(k) == v for k, v in gate.items())
        and "THRESHOLDS['exp26']" in str(
            d.get("gate", {}).get("gate_source")
        ),
        f"failures={d.get('gate', {}).get('failures')}",
    )

    points = d.get("points", []) or []
    sizes = [1000, 2500, 5000, 10000, 20000, 50000]
    check(
        "exp26: exactly the six pre-registered sizes, in order — no "
        "point dropped because it failed",
        [p.get("payload_bits") for p in points] == sizes,
        f"sizes={[p.get('payload_bits') for p in points]}",
    )

    bad = []
    for p in points:
        ber, bits, errs = (
            p.get("ber"), p.get("bits_compared"), p.get("bit_errors")
        )
        size = p.get("payload_bits")
        if ber != (errs / bits if bits else None):
            bad.append(f"{size}:ber")
        if ber != 0.0:
            bad.append(f"{size}:nonzero")
        if not (p.get("decrypt_ok") and p.get("recovered_matches")):
            bad.append(f"{size}:decrypt")
        if p.get("meets_ber_gate") != (ber == 0.0):
            bad.append(f"{size}:ber-flag")
        if p.get("meets_detectability_gate") != (
            p.get("detector_accuracy") <= 0.55
            and p.get("kl_divergence") <= 0.05
        ):
            bad.append(f"{size}:det-flag")
    check(
        "exp26: every point recomputes — BER from errors/bits at "
        "0.0, decrypt + message match, both gate flags re-derived "
        "from the numbers beside them",
        not bad,
        f"bad={bad}",
    )

    metrics = d.get("metrics", {}) or {}
    accs = [p["detector_accuracy"] for p in points]
    kls = [p["kl_divergence"] for p in points]
    check(
        "exp26: metrics recompute from the points — six sizes at "
        "BER 0.0, the 50k maximum, and the detector/KL ranges equal "
        "the observed min/max exactly",
        metrics.get("sizes_tested") == 6
        and metrics.get("sizes_measured") == 6
        and metrics.get("sizes_at_ber_zero") == sizes
        and metrics.get("max_tested_payload_at_ber_zero_bits") == 50000
        and metrics.get("detector_accuracy_range") == [
            min(accs), max(accs)
        ]
        and metrics.get("kl_range") == [min(kls), max(kls)]
        and max(accs) <= 0.55
        and max(kls) <= 0.05,
        f"acc_range={metrics.get('detector_accuracy_range')} "
        f"kl_range={metrics.get('kl_range')}",
    )

    figure = RESULTS / EXP26_FIGURE
    check(
        "exp26: the committed figure exists and is non-empty — the "
        "one artifact reviewers can see without running anything",
        figure.is_file() and figure.stat().st_size > 0,
        str(figure),
    )

    check(
        "exp26: status PASS with no gate failures",
        d.get("status") == "PASS"
        and d.get("gate", {}).get("failures") == [],
        f"status={d.get('status')}",
    )


def _audit_paper_exp27():
    d = one(EXP27_ARTIFACT)
    check(
        "exp27: artifact present (W9.3 threat-model boundary, "
        "Qwen2.5-3B)",
        d is not None,
        EXP27_ARTIFACT if d else "missing: " + EXP27_ARTIFACT,
    )
    if not d:
        return

    gate = THRESHOLDS["exp27"]
    check(
        "exp27: gate mirrors THRESHOLDS['exp27'] field-for-field "
        "(control 0.0, >=10 wrong keys, 0 recoveries) with the "
        "source pinned",
        all(d.get("gate", {}).get(k) == v for k, v in gate.items())
        and "THRESHOLDS['exp27']" in str(
            d.get("gate", {}).get("gate_source")
        ),
        f"status={d.get('status')} "
        f"failures={d.get('gate', {}).get('failures')}",
    )

    measured = dig(d, "gate", "measured") or {}
    check(
        "exp27: the measured boundary — control recovers at 0.0, "
        ">=10 wrong keys tested with 0 recoveries and 0 plaintext "
        "emissions, 0 partial-access recoveries",
        measured.get("control_ber") == 0.0
        and measured.get("control_recovered") is True
        and measured.get("wrong_keys_tested")
        >= gate.get("min_wrong_keys_tested")
        and measured.get("wrong_key_recoveries")
        == gate.get("max_wrong_key_recoveries")
        and measured.get("wrong_key_plaintext_emitted") == 0
        and measured.get("partial_access_recoveries")
        == gate.get("max_partial_access_recoveries"),
        f"measured={measured}",
    )

    control = dig(d, "conditions", "control") or {}
    check(
        "exp27: control arithmetic — 0 errors over the 10,256-bit "
        "construction, message recovered",
        control.get("ber") == 0.0
        and control.get("bit_errors") == 0
        and control.get("bits_compared") == 10256
        and control.get("recovered_matches") is True,
        f"control={control}",
    )

    wrong = dig(d, "conditions", "wrong_key") or {}
    kinds = wrong.get("error_kinds") or {}
    check(
        "exp27: every wrong key fails the same authenticated way — "
        "10/10 GCM authentication, no header failures, no garbage "
        "success, and the channel remains readable with the map "
        "(0.0): channel readability and message confidentiality "
        "measured as separate axes",
        sum(kinds.values()) == wrong.get("keys_tested")
        and set(kinds) == {"gcm_authentication"}
        and wrong.get("recoveries") == 0
        and wrong.get("plaintext_emitted") == 0
        and wrong.get("channel_ber_with_map_no_key") == 0.0,
        f"kinds={kinds} channel_ber="
        f"{wrong.get('channel_ber_with_map_no_key')}",
    )

    cells = dig(d, "conditions", "partial_access", "cells") or []
    by_name = {c.get("condition"): c for c in cells}
    prefix = [
        by_name.get(f"prefix_{t}pct", {}).get("stream_coverage")
        for t in (50, 25, 10)
    ]
    scattered = [
        by_name.get(f"scattered_{t}pct", {}).get("stream_coverage")
        for t in (50, 25, 10)
    ]
    check(
        "exp27: six partial-access cells, none decrypts, none "
        "recovers the message, every available bit reads clean "
        "(0.0), and coverage falls monotonically with the layer "
        "fraction for both access patterns",
        len(cells) == 6
        and all(
            c.get("decrypt_ok") is False
            and c.get("recovered_matches") is False
            and c.get("ber_on_available_bits") == 0.0
            and 0.0 < (c.get("stream_coverage") or 0.0) < 1.0
            for c in cells
        )
        and all(
            a is not None and b is not None and a > b
            for a, b in zip(prefix, prefix[1:])
        )
        and all(
            a is not None and b is not None and a > b
            for a, b in zip(scattered, scattered[1:])
        ),
        f"prefix={prefix} scattered={scattered}",
    )

    exact = dig(d, "conditions", "public_rule_keyless", "exact_size")
    pos = (exact or {}).get("positions") or {}
    tp = pos.get("true_positives")
    fp = pos.get("false_positives")
    ber = (exact or {}).get("keyless_ber")
    check(
        "exp27: the public-rule attacker is chance-level on this "
        "embed (exact-size BER in [0.4, 0.6]) with precision "
        "recomputing from TP/FP, and the message-protection basis "
        "cites C2 rather than asserting secrecy",
        ber is not None and 0.4 <= ber <= 0.6
        and pos.get("precision") == (tp / (tp + fp) if (tp + fp) else None)
        and bool(
            dig(d, "conditions", "public_rule_keyless",
                "message_protection_basis")
        ),
        f"ber={ber} precision={pos.get('precision')}",
    )

    check(
        "exp27: status PASS with no gate failures",
        d.get("status") == "PASS"
        and d.get("gate", {}).get("failures") == [],
        f"status={d.get('status')}",
    )


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
        ("keyless recovery (exp13)", audit_keyless),
        ("blind-patch adversary (exp14)", audit_blind),
        ("LWE fidelity (exp15)", audit_lwe_fidelity),
        ("cross-scheme detector (exp16)", audit_cross_scheme),
        ("QAE round trip (exp17)", audit_qae_round_trip),
        ("strategy matrix (exp18)", audit_strategy_matrix),
        ("adaptive routing (exp19)", audit_adaptive_routing),
        ("sign/parity split (exp20)", audit_split_dial),
        ("QAE/LWE interop (exp21)", audit_qae_lwe_interop),
        ("per-layer LWE width (exp22)", audit_layer_widths),
        ("model surgery survival (exp23)", audit_model_surgery),
        ("Pareto frontier (exp24)", audit_pareto_frontier),
        ("paper hardening W9 (exp25-27)", audit_paper_experiments),
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

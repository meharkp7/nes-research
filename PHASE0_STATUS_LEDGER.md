# Phase 0 — Final Closeout Status Ledger

**Reviewed:** 2026-10-16 (Asia/Kolkata). **Closeout status:** complete for code repair, validation, evidence reconciliation and repository review. **Git status:** reviewed working tree, not committed or pushed. Scientific failures and provenance gaps remain part of the frozen result.

Start here, then read [Phase 1: dated packed-NF4 audit](PHASE1_PACKED_NF4_EVIDENCE_AUDIT.md), [Phase 2: residual protocol boundaries](PHASE2_RESIDUAL_DETECTOR_READINESS.md), and [Phase 3: consolidated evidence and final claims](PHASE3_CONSOLIDATED_SEVEN_METHOD.md). These four documents supersede earlier status summaries. Earlier plans, experiment reports and failed runs remain historical records.

## Definition of done — 9/9

- [x] All targeted tests pass.
- [x] Full test suite passes.
- [x] Claim audit passes without erasing experimental failures.
- [x] Consistency check and final diff check pass.
- [x] Five-method residual evidence reconciled by protocol and model roster.
- [x] QSE/DCE evidence reconciled across both dated evaluations.
- [x] Four phase documents agree with the reviewed evidence.
- [x] Final claims and provenance limitations are explicit.
- [x] Working tree reviewed and commit contents selected below; no commit performed.

## Checks actually rerun

Commands run from `nes-llm` with `../.venv/bin/python`; source artifacts were read, not regenerated. Logs are retained under `artifacts/closeout_20261016/`.

| Check | Result on 2026-10-16 | Log |
|---|---|---|
| `-m pytest tests/test_integration.py tests/test_phase7a.py tests/test_phase7b.py tests/test_prepare_packed_nf4_grouped_dataset.py -q` | **45 passed**, 3.40 s | [targeted.txt](artifacts/closeout_20261016/targeted.txt) |
| `-m pytest -q` | **419 passed, 3 subtests passed**, 23 warnings, 18.64 s | [full.txt](artifacts/closeout_20261016/full.txt) |
| `-m pytest tests/test_packed_nf4_method_eval_utils.py -q` | **6 passed**, 0.23 s | [utility.txt](artifacts/closeout_20261016/utility.txt) |
| `claim_audit.py` | **All 152 claims verified** | [claims.txt](artifacts/closeout_20261016/claims.txt) |
| `check_consistency.py` | **9/9 checks passed, 41 artifacts verified** | [consistency.txt](artifacts/closeout_20261016/consistency.txt) |
| Source hash reconciliation | **296 source fingerprints recorded**; both detector dataset hashes, new split hash, LOLO input hash and 44 manifest input hashes match | [frozen evidence record](artifacts/closeout_20261016/evidence.json) |
| Final evidence / diff review | See [review record](artifacts/closeout_20261016/review.json) | Recorded after documentation reconciliation |

The full-suite warnings comprise 14 PyTorch Python-3.14 deprecations, 1 joblib physical-core discovery fallback, 4 sklearn MLP batch-size clips on test fixtures and 4 QAE tensor-copy warnings. No test errors remain. No baseline warning log was supplied, so whether individual warnings are new is **UNVERIFIED**; they originate in dependencies or unchanged code paths, not the edited comparator lines.

## Code repair and one exposed compatibility failure

| Change | Purpose / verification |
|---|---|
| Remove adaptive branch's local `SignExtractor` import | Use the existing module import and stop `UnboundLocalError` in the sign branch. |
| Supply `RobustnessResult` threshold fields 0.02 and 0.10 | First targeted rerun exposed four constructor errors previously hidden by the import failure (41 passed / 4 failed). Added the already-used gate thresholds; did not alter BER calculations or gate values. Final targeted run: 45 passed. |
| Rename/update three Phase 7B tests | Document ignored wrong keys, key-independent widths and layer-independent widths. Verify complete wrong-key recovery at BER 0.0. No LWE implementation change. |
| Rename/update insufficient-model-group test | Retain the two-group fixture and expect the current minimum of five distinct IDs (2 train, 1 validation, 2 test). |

## Frozen evidence state

| Phase / track | Closeout result | Important remaining limit |
|---|---|---|
| Phase 1 / B | Reconciled 20261010 historical and 20261011 method-specific evaluations; completed readiness and provenance audits reviewed | Revision matching UNVERIFIED; near-chance LR/HGB result limited to the recorded 4/1/2 model-ID split |
| Phase 2 / A | exp18 and TinyLlama LOLO separated; five-method corpus recovery retained | New model-held-out five-method detector comparison BLOCKED by one-model feature coverage; no experiment restart |
| Phase 3 / both | Seven named methods covered by source-backed rows and explicit missing cells | No shared seven-method performance ranking or universal recovery, utility, robustness or stealth claim |
| Package | Report copies, original fingerprints, test/audit logs and reviewed change list saved | Large feature datasets and checkpoints remain local; not a fully reproducible or provenance-verified release |

Known failures remain: exp7 neural accuracy 0.705 (FAIL); exp8 inherits that failure; exp13 keyless channel recovery BER 0.0 (FAIL security gate); exp21 raw interop failure; exp22 magnitude-dependent-width robustness failure; exp23's four surgery failures; exp25 H3 unsupported despite its separate experiment gate passing; early QSE candidate/corpus failures. `nf4_qae` remains BLOCKED and neural embedding remains excluded as NEEDS_TRAINING in exp18. TinyLlama was SKIPPED from exp18's roster, while its later artifact recovery/LOLO evidence is explicitly dated separately.

Corrections to the earlier drafts include: exp18 measures four methods (28 total cells); magnitude-aware does not have zero BER at every noise point; QAE robustness was measured in exp18; split detector scores are not strictly monotonic; surgery was measured for sign; exp5's signed PPL change is negative; successful packed artifact corpus runs exist and are not full HF checkpoints; old pooled detector numbers are not new method-specific results; family holdout is narrower than the saved report's wording suggests.

## Repository review and freeze selection

Base branch: `research/contract-b-b14-and-evaluation`. Base commit: `c81edda7893128d120c2b417af92f3f74c282ce7`. No branch reset, cleanup, dataset reconstruction, detector fit, embedding experiment, staging, commit or push was performed for closeout. Unit tests exercise small synthetic fixtures only.

Selected for a future closeout commit after this review:

- The three repaired files: `nes-llm/src/evaluation/strategy_comparator.py`, `nes-llm/tests/test_phase7b.py`, `nes-llm/tests/test_prepare_packed_nf4_grouped_dataset.py`.
- The two pre-existing dataset/readiness edits: `nes-llm/scripts/prepare_packed_nf4_grouped_dataset.py` and `nes-llm/scripts/validate_packed_nf4_detector_readiness.py`. The former enforces five IDs; the latter scopes duplicate block identity to source/run while retaining cross-split artifact identity checks.
- The pre-existing untracked evaluator/utility files: `packed_nf4_method_eval_utils.py`, `run_packed_nf4_method_specific_detector.py`, `run_seven_method_residual_detector_lolo.py`, `stream_packed_nf4_grouped_dataset.py` under `nes-llm/scripts/`, plus `nes-llm/tests/test_packed_nf4_method_eval_utils.py`. Reviewed as experiment-specific tools, not general provenance validators. In particular, method inference is pinned to `_qse_20261011` / `_dce_20261011`; the LOLO script has fixed dated input/output paths and refuses overwrites.
- These four phase documents, corrected current [methodology summary](METHODOLOGY_SUMMARY.md), navigation/status notes in [README](README.md) and [research plan](RESEARCH_PLAN.md), and the `artifacts/closeout_20261016/` evidence/log/review record. Small saved JSON/CSV report contents are embedded in `evidence.json` because `cache/` is ignored; no source bytes were overwritten.

**Explicitly excluded and untouched:** `nes-llm/scripts/prepare_packed_nf4_grouped_dataset.py.bak` (SHA-256 `6653c8ceac192c62e9e486a927170180bf720cfe1f1a538dd3ba816bf2047fcd`), historical experiment artifacts, cached feature datasets, model checkpoints and unrelated files. The final review record lists exact paths and hashes, including untracked files; `git diff --stat` alone omits untracked content.

This is a reviewed working-tree freeze, not a new Git revision. The selected files remain available for an explicit commit action. No new experiment is needed to state the current limitations honestly.

# AGENTS.md — NES research workspace

Run root: `/workspace/nes_run_20261016_anchor_ablation/` (not a git repo).
Git repo for the code: `meharkp7/nes-research` (clean clone at `/workspace/project`,
branch `main`). Do not modify the clone; work here.

## Hard constraints (do not violate)

- Frozen codec `code/nes-llm/src/experiments/nf4_anchor_channel.py`, sha256
  `07ce008dcabebb7ac0d4fa10c88251bf10dc9f2a7ae27dfc79bf23b8cd75b764`. Never modify.
- Never overwrite/delete/reset frozen historical evidence. New runs get fresh
  names (e.g. `*_v1`). Failed run dirs are kept (e.g. `*_ABORTED_DISK`).
- Two research tracks stay separate: Track A (5 residual methods), Track B
  (QSE/DCE packed NF4). The `interior_nearest` method is its own method-paper
  investigation. Never pool datasets/claims across tracks.

## StableLM-2-1.6B qualification (`stablelm_qualification/`)

- Model `stabilityai/stablelm-2-1_6b` @ `f499ead74c53749bd93cebc6ce8bc0d7bdf1eaef`.
- Gates A–F PASS; **Gate E PASS (2026-10-19, 6/6 within ±0.5%)**; Gate G
  prior-art PARTIAL (unresolved novelty risk, not a method failure).
- Reports: `STABLELM_QUALIFICATION_REPORT.md`, `ROUNDTRIP_STABLELM.md`,
  `DETECTABILITY_STABLELM.md`.
- Evidence ledger: `../RESULTS_EVIDENCE_LEDGER.md` (rows 17–26 for StableLM).
- Manuscript tables: `tables/` (table1 recovery, table3 detectability,
  table4 utility).
- **Expansion UNLOCKED for Track A under `QUALIFICATION_PROTOCOL_V2.md`
  (ACTIVE, user-approved Option A, 2026-10-19).** Track A (technical) gates the
  expansion; Track B (novelty) / Gate G stays **PARTIAL** and is disclosed, never
  a method failure. The V1 note below ("expansion LOCKED") is superseded.

## Gate E utility run (COMPLETE — no action needed)

Finished 2026-10-19. Verdict in `utility_v2/gate_e_reanalysis.json`
(`all_within_margin: true`). The supervisor self-terminated; no processes remain.
`RESUME_GATE_E.md` retains the re-arm command in case a re-run is ever needed.

## Expansion qualification (2026-10-19, in progress)

Drivers live in `expansion_qualification/`: `build_clean_nf4.py`,
`exp_roundtrip_stream.py`, `exp_detectability.py`, `exp_utility.py`,
`check_integrity.py`, `test_codec_regressions.py`, `analyze_expansion.py`,
`emit_audit_csv.py`, `run_utility_queue.sh`. Stage fp16 + clean NF4 for new
models under `/tmp/nes_stage/` (overlay) — `/workspace` is nearly full.

Models (revisions verified against the Hub):
- SmolLM2-1.7B `effd688a12921b4cc83e3312b6feb579f70f9c71` (24L, Llama naming)
- Qwen2.5-1.5B `8faed761d45a263340a0528343f099c05c9a4323` (28L, Llama naming)
- pythia-1.4b `fedc38a16eea3bd36a96b906d78d11d2ce18ed79` (24L, GPT-NeoX naming,
  fused `attention.query_key_value.weight`)
- Falcon3-1B-Base `cb37ef3559b157b5c9d9226296ba01a5162da1f7` (18L, Llama naming)

Results so far (Track A): `interior_nearest` 360/360 exact on all four;
baselines 90/360 (Falcon3 108/360); interior mutates 0 anchor codes. Gate A
PASS, regressions 12/12. Detectability all AUC ≤ 0.507; only Qwen LR is
nominally significant (0.5068, p=0.015) and it is below its own clean control
(0.512) → within noise floor, retained as a minor unresolved discrepancy.
Utility A5 still running (n_seq=580, batch=4, ~1.5–2 h/condition, sequential).
Report: `expansion_qualification/EXPANSION_QUALIFICATION_REPORT_V1.md`.

Audit finding: `FROZEN_ALL_HASHES.txt` self-hash mismatches and 5 narrative docs
changed after the manifest (mtime 2026-10-19 10:09–10:10); raw artifacts re-verify
clean. Disclosed in report §0.1, not repaired.

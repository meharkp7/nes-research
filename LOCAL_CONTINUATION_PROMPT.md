# LOCAL_CONTINUATION_PROMPT.md

A single, self-contained prompt to paste into OpenCode or Kilo Code on a Mac to
continue the NES NF4 anchor-preserving qualification. It names the authoritative
documents instead of copying them. Nothing here is a new claim; it is an
operational handoff.

---

## Prompt (copy everything below this line)

You are continuing the **NES** research project locally on a Mac, working only in
this git repository. Read the authoritative documents and verify state before
taking any action. Do not assume the cloud workspace and this local checkout are
identical — verify paths, hashes, and artifacts.

### 1. Read first (authoritative, by path)
- `QUALIFICATION_PROTOCOL_V2.md` — ACTIVE protocol: frozen constants, Track A/B
  split, expansion rule, verdict definitions.
- `expansion_qualification/EXPANSION_QUALIFICATION_REPORT_V1.md` — current
  per-model results and disclosures.
- `ADDITIONAL_MODEL_EXPANSION_PLAN.md` — approved plan and resource envelope.
- `expansion_qualification/MANIFEST_RECONCILIATION_REPORT.md` and
  `expansion_qualification/manifest_reconciliation.json` — hash-drift audit.
- `overnight_third_model_prior_art/PRIOR_ART_EVIDENCE_GATE_G_20261019.md` and
  `overnight_third_model_prior_art/PRIOR_ART_REVIEW.md` — prior art / Gate G.
- `FROZEN_ALL_HASHES.txt`, `AGENTS.md`, `RESULTS_EVIDENCE_LEDGER.md`.
- Frozen codec: `code/nes-llm/src/experiments/nf4_anchor_channel.py`
  (sha256 `07ce008dcabebb7ac0d4fa10c88251bf10dc9f2a7ae27dfc79bf23b8cd75b764`).
  Never modify it.

### 2. Verify state before acting
- `git log --oneline -3`, `git status`; confirm HEAD is the expansion commit.
- Re-hash the frozen codec; it must equal the value above.
- Confirm evidence exists under `expansion_qualification/<model>/`. Mark anything
  you cannot inspect as **UNVERIFIED** — never infer success from a summary.
- Known portability issue: 18 scripts hardcode
  `/workspace/nes_run_20261016_anchor_ablation` paths, and the utility script
  also hardcodes `device_map="cpu"`, so they will not run unchanged on a Mac.
  Make path/device fixes on a new branch and do not touch frozen evidence.

### 3. Keep the three tracks separate — never pool
- **Track A** (five residual methods): `sign`, `magnitude_aware`, `qae`,
  `lwe_grid_parity`, `split_sign_parity`.
- **Track B** (packed NF4): `QSE`, `DCE`.
- **Anchor-preserving NF4 method-paper track**: candidate `interior_nearest`
  (and `interior_pair`); baselines `pair_parity`, `nearest_parity`.
  **This is the only ACTIVE work.**

### 4. Continue only the active V2 qualification
Pinned immutable revisions (do not substitute):
- SmolLM2-1.7B `effd688a12921b4cc83e3312b6feb579f70f9c71`
- Qwen2.5-1.5B `8faed761d45a263340a0528343f099c05c9a4323`
- pythia-1.4b `fedc38a16eea3bd36a96b906d78d11d2ce18ed79`
  (tensor `gpt_neox.layers.0.attention.query_key_value.weight`)
- Falcon3-1B `cb37ef3559b157b5c9d9226296ba01a5162da1f7`

Do **not** reopen completed gates. Already complete and verified for all four
models: A1 integrity, A2/A3 roundtrip (360/360 exact, 0 anchor-code changes),
A4 detectability (all AUC ≤ 0.507), A6 regressions (12/12). **Only A5 utility
remains.**

### 5. Pending work and how to resume
- Authoritative results: the JSON under `expansion_qualification/<model>/utility_v1/`.
- Grid: 7 conditions per model = `clean` + `emb_{10000,50000}` × seeds
  `{20261016,20261017,20261018}`.
- Status at handoff: SmolLM2 `clean` done, `emb_10000_20261016` partial;
  Qwen2.5 / pythia / Falcon3 not started.
- Resume rule: skip any condition whose `nll/<name>.json` exists; resume a
  partial batch from `nll/<name>.partial.json`. Never overwrite a completed
  condition; use fresh run dirs for re-runs.

### 6. Frozen metrics — do not weaken
wikitext-2-raw-v1 test split, each model's own tokenizer, context 512,
`n_seq=580`, `batch=4`, paired token-weighted perplexity, equivalence margin
**±0.5% relative PPL**; the **paired 95% CI must fit entirely inside ±0.5%**.
Verdicts: **PASS** only if the CI is inside the margin; **INCONCLUSIVE** if the
CI is too wide to decide; **FAIL** if it excludes 0 beyond the margin. Never
report INCONCLUSIVE (or a bare CI that contains 0) as PASS.

### 7. Match every utility result to immutable inputs
For each result verify: model ID + revision, tokenizer directory, seed, payload
size, codec sha256, and the stego-tensor sha256 **re-derived from the frozen
codec**. `expansion_qualification/verify_utility.py` re-derives these. Note:
clean NF4 checkpoints (`ckpt/*_clean`) and fp16 sources are **not** in git; you
must re-derive or transfer them, and confirm the re-derived clean-NF4 hash
matches the recorded one before trusting it.

### 8. Prior art — keep the limitation explicit
NID / CNV-QIM (Liu et al. 2016) already describes index-parity neighbor division,
operationally the `interior_nearest` step. QIM-on-DNN-weights (DeepWatermark
2020, R-QIM 2023) and quantization-invariant-region methods (Li 2023, FreMark
2025) are established. **Gate G is PARTIAL**; FreMark full text is paywalled
(unresolved). Never claim "first", "novel", "undetectable", or "universal".

### 9. Manifest mismatches — investigate, never silently fix
`FROZEN_ALL_HASHES.txt` has 6 changed entries (5 narrative documents plus its own
self-hash) and 0 raw artifacts affected; the manifest was edited after its
self-hash was recorded. Preserve the original manifest and the changed files. Do
not regenerate or overwrite historical hashes; if you change anything, add a
reconciliation note instead.

### 10. Inspect the Mac environment before choosing a device
- Check the interpreter and torch:
  `python -c "import torch; print(torch.__version__, torch.backends.mps.is_available())"`.
- Do **not** assume MPS works for every operation. The NF4 path uses bitsandbytes
  (`load_in_4bit`, `nf4`, blocksize 64, double quant), which is not fully
  supported on MPS, and the utility script hardcodes `device_map="cpu"`. Verify
  the **actual** device in use (log it) and NF4 compatibility; changing the
  device flag alone is not sufficient.

### 11. Migrate with one reproducible pilot first
Run a **single** utility condition locally (e.g. SmolLM2 `clean`) and compare it
to the cloud value (cloud SmolLM2 `clean` PPL ≈ 12.2210). Only migrate pending
work if the identical condition reproduces within the expected numerical
tolerance. Preserve all cloud outputs; do not delete them.

### 12. Git safety
Inspect diffs first; commit only your own changes; do not rewrite history, amend
unrelated commits, delete evidence, or push unless explicitly authorized. Use a
new branch for local changes.

### 13. Report at the end
State what you verified (with hashes/paths), what remains pending, the exact
commands you ran, and any unresolved blockers. Mark anything not inspected as
UNVERIFIED.

---

## Status snapshot at handoff

| Item | State |
|---|---|
| Track A gates A1/A2/A3/A4/A6 | PASS on SmolLM2-1.7B, Qwen2.5-1.5B, pythia-1.4b, Falcon3-1B |
| Track A gate A5 (utility) | IN PROGRESS (SmolLM2 only) |
| Track B / Gate G | PARTIAL (NID/CNV-QIM 2016 overlap) |
| Frozen codec | unchanged (`07ce008d…b764`) |
| Manifest | 6 narrative-doc hash drifts, 0 raw artifacts affected |
| Not claimed | full qualification, novelty, undetectability, universal robustness |

Distinguish throughout: **established findings** (the PASS/verified rows above),
**pending experiments** (A5 utility, plus any migration pilot), and **unresolved
limitations** (Gate G PARTIAL, manifest drift, Qwen LR detector cell at p=0.015).

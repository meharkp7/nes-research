# Expansion qualification report v1 — NF4 anchor-preserving embedding

Scope: the frozen `interior_nearest` NF4 anchor-preserving method (candidate
method paper). Track A = technical generalization across models. Track B =
novelty/prior-art. The two tracks are reported separately and never pooled.

Status date: 2026-10-19. Author: automated research agent. Not a legal opinion.

## 0. Frozen inputs (must not change)

| Item | sha256 |
|---|---|
| Codec `code/nes-llm/src/experiments/nf4_anchor_channel.py` | `07ce008dcabebb7ac0d4fa10c88251bf10dc9f2a7ae27dfc79bf23b8cd75b764` |
| StableLM protocol `stablelm_qualification/QUALIFICATION_PROTOCOL_STABLELM.md` | `581bd57e28c0768829386e0a0230a527d212eba2612e8197cf75b09dba68de87` |

The codec hash is re-verified at the top and bottom of every audit run; it is
unchanged. No frozen artifact under `out/`, `overnight_third_model_prior_art/`,
or `stablelm_qualification/` was modified by this expansion.

### 0.1 Frozen-manifest integrity note (audit finding, disclosed)

`FROZEN_ALL_HASHES.txt` (mtime 2026-10-17 21:02) lists 271 hashed paths + 3
annotation lines. Re-verification on 2026-10-19 found **6 listed files whose
current hash differs from the manifest** and **0 missing**:

- `FROZEN_ALL_HASHES.txt` (the manifest's self-hash, line 84, no longer matches)
- `PAPER_OUTLINE.md`, `MANUSCRIPT_READINESS.md`, `RESULTS_EVIDENCE_LEDGER.md`
  (all mtime 2026-10-19 10:09–10:10, i.e. after the manifest was written)
- `overnight_third_model_prior_art/PRIOR_ART_REVIEW.md` (mtime 2026-10-19 10:09)
- `overnight_third_model_prior_art/PRIOR_ART_BIBLIOGRAPHY.bib` (mtime 2026-10-17 21:40)

These are **narrative/meta documents only**. Zero changed paths are raw
artifacts (`out/…`, `…/conditions/…`, or result JSON): the raw-evidence subset of
the manifest re-verifies clean. The changes predate this expansion session
(all mtimes ≤ 2026-10-19 10:10; expansion artifacts begin 10:5x) and are
attributed to the immediately preceding documentation workstream, not to the
expansion. Disposition: reported, not silently repaired. The manifest self-hash
discrepancy should be resolved by the maintainer (regenerate or annotate), and
the three 10:09–10:10 docs reconciled against their last committed revision.

## 1. Models and revisions

| Key | Model | Revision | Layers | Tensors |
|---|---|---|---|---|
| M3 (ref) | StabilityAI/stablelm-2-1_6b | (frozen) | 24 | L0/L11/L23 |
| M4 | HuggingFaceTB/SmolLM2-1.7B | `effd688a12921b4cc83e3312b6feb579f70f9c71` | 24 | 6 |
| M5 | Qwen/Qwen2.5-1.5B | `8faed761d45a263340a0528343f099c05c9a4323` | 28 | 6 |
| M6 | EleutherAI/pythia-1.4b | `fedc38a16eea3bd36a96b906d78d11d2ce18ed79` | 24 | 6 |
| M7 | tiiuae/Falcon3-1B-Base | `cb37ef3559b157b5c9d9226296ba01a5162da1f7` | 18 | 6 |

Six-tensor rule: depth (early/mid/late) × projection (attn q/k/v, mlp up/down/gate),
using each model's actual tensor naming. Pythia uses GPT-NeoX fused-QKV naming
(`gpt_neox.layers.N.attention.query_key_value.weight`); Falcon3 uses Llama
naming with 18 layers.

## 2. Track A results (technical generalization)

### A2/A3 — roundtrip exact recovery (cycles 0–3, sizes 1k/5k/10k/20k/50k, seeds ×3, 6 tensors)

Per-cycle audit table: `expansion_qualification/expansion_audit_table.csv`
(4320 rows = 1080 conditions × 4 cycles). Conditions present 1080 / expected
1080, missing 0, all VERIFIED from raw records.

| Model | `interior_nearest` exact | `nearest_parity` | `pair_parity` | interior anchor-code changes |
|---|---|---|---|---|
| SmolLM2-1.7B | **360/360** | 90/360 | 90/360 | 0 |
| Qwen2.5-1.5B | **360/360** | 90/360 | 90/360 | 0 |
| pythia-1.4b | **360/360** | 90/360 | 90/360 | 0 |
| Falcon3-1B | **360/360** | 108/360 | 108/360 | 0 |

Baselines mutate 85k–149k anchor codes across their conditions and fail; the
anchor-preserving method mutates zero anchor codes and recovers the payload
exactly on every cycle. Baselines are reported as failing, not discarded.

### A1 — integrity (only intended tensor differs)

| Model | n_tensors | only intended differs | shape preserved | payload exact | finite logits/loss |
|---|---|---|---|---|---|
| SmolLM2 | 1058 | true | true | true | true |
| Qwen2.5 | 1318 | true | true | true | true |
| pythia | 772 | true | true | true | true |
| Falcon3 | 795 | true | true | true | true |

### A4 — detectability (16 features, grouped splits, clean-vs-clean control, n=86016, 6 groups)

| Model | control (clean vs clean) | GB AUC | LR AUC | MLP AUC | max TPR@1% |
|---|---|---|---|---|---|
| SmolLM2 | 0.5031 | 0.4999 (p=0.46) | 0.5012 (p=0.35) | 0.4999 (p=0.54) | 0.0100 |
| Qwen2.5 | 0.5120 | 0.5025 (p=0.22) | **0.5068 (p=0.015)** | 0.5010 (p=0.38) | 0.0106 |
| pythia | 0.4999 | 0.5002 (p=0.52) | 0.5007 (p=0.39) | 0.5000 (p=0.50) | 0.0101 |
| Falcon3 | 0.5059 | 0.5008 (p=0.38) | 0.5018 (p=0.22) | 0.5003 (p=0.45) | 0.0104 |

Screen: AUC ≤ 0.55 and TPR@1% ≤ 0.02. All 12 detector×model cells pass. The
Qwen LR cell is the only nominally significant deviation (p=0.015) and is
retained as an **unresolved minor discrepancy**, not relabeled: its AUC (0.5068)
is smaller than Qwen's own clean-vs-clean control AUC (0.5120), so it is inside
the model's detector noise floor, and 1 of 12 cells at α=0.05 is consistent with
multiple-comparison expectation. It does not meet the predeclared fail rule.

### A5 — utility (paired PPL, wikitext-2, own tokenizer, ctx 512, n_seq=580, batch=4)

Margin: paired 95% CI must fit inside ±0.5% relative PPL. Reference StableLM
(protocol-matched: n_seq=580, sizes 10k/50k, seeds ×3) achieved CI half-widths
well inside ±0.5% (e.g. `[-0.024%, +0.021%]` at 10k).

| Model | status |
|---|---|
| SmolLM2-1.7B | IN PROGRESS (clean baseline) |
| Qwen2.5-1.5B | QUEUED |
| pythia-1.4b | QUEUED |
| Falcon3-1B | QUEUED |

Utility is CPU-bound and single-instance; the expansion queue
(`expansion_qualification/run_utility_queue.sh`) runs the four models
sequentially. Any CI that cannot fit inside ±0.5% within the CPU budget is
reported INCONCLUSIVE per protocol, not as a pass.

## 3. Regression tests (frozen-codec invariants)

12/12 pass on each of SmolLM2, Qwen2.5, pythia, Falcon3 (including
`endpoints_unchanged`, `interior_only`, `requantize_cycles_0_5_exact`,
`fresh_process_extraction`, corruption/wrong-key/truncation rejection, and
`pair_parity_fails_after_requant`).

## 4. Confirmed results vs unresolved discrepancies

**Confirmed (inspectable artifacts):**
- Anchor-preserving exact recovery generalizes to four additional architectures
  (Llama-family, Qwen, GPT-NeoX, Llama-18L) across sizes, seeds, and 0–3 cycles.
- Zero anchor-code changes and zero endpoint rewrites on the method path.
- Integrity, idempotence, and rejection invariants hold on all four.
- Detectability at or near chance on all four, under the predeclared screen.

**Unresolved discrepancies (retained, not resolved away):**
1. Qwen2.5 LR detector AUC 0.5068, p=0.015 (sub-threshold; below its own control).
2. `FROZEN_ALL_HASHES.txt` self-hash mismatch + 5 narrative docs changed after the
   manifest was written (raw artifacts unaffected).
3. Utility (A5) not yet measured for any expansion model.

## 5. What this does and does not establish

- Establishes: the anchor-preserving *mechanism* reproduces on four more models.
- Does NOT establish: universal generalization, undetectability, or novelty.
- Track B / Gate G remain **PARTIAL** and are unchanged by this expansion. Prior
  art (frozen): NID/CNV-QIM (2016) already describes index-parity neighbor
  division; DeepWatermark (2020) and R-QIM (2023) apply parity/QIM to DNN
  weights; Li-2023/FreMark-2025/HPQIM establish the invariant-region idea. The
  only still-unmatched pieces are the NF4 packed-code formulation, the explicit
  endpoint (0/15) exclusion, and the repeated-requantization exactness result.
- Passing on more models is a generalization result, not a novelty result.

## 6. Next action

Finish A5 utility for the four expansion models (queue running), then issue
report v2 with utility verdicts. In parallel, the maintainer should reconcile the
6 frozen-manifest mismatches. See `ADDITIONAL_MODEL_EXPANSION_PLAN.md` for the
frozen procedure.

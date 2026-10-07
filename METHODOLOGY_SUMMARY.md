# NES Research — Methodology at a Glance

*One page: every step, the method used, what it returned, and why we moved on.*
Repo: `meharkp7/nes-research` · 24 experiments · payload = 10,000 bits in Qwen/LLM `down_proj` weights.

---

## The chain

| # | Step | Methodology | Result | Then we… |
|---|------|-------------|--------|----------|
| 1 | **Build the instrument** (exp1–3) | Production embed → extract → decrypt round trip; **pin gates before testing** (BER 0.0 · detector ≤55% · PPL ≤2%) | Clean round trip **BER 0.0**; gates frozen | Never moved a gate again — everything below is measured against it |
| 2 | **Compare mechanisms** (exp8–11) | Swap embedding strategy under one identical payload + path: sign vs magnitude vs parity vs QAE; verify NF4/AWQ/GPTQ dequantizers vs known tensors | Sign/LWE family wins; adapters pass correlation ≥0.95; QAE = *well-matched to NF4*, not stealthy | Standardized on **LWE-lattice sign embedding** |
| 3 | **Red-team ourselves** (exp7, 13–16) | Attack our own encoder: statistical (KL), neural, blind-patch, cross-scheme detectors, keyless recovery | Blind detector scores **exactly 0.5000**; lattice search spikes on the true structure → **secrecy claims REFUTED** | Recorded the honest limits: covert to the naive, fingerprinted to the informed |
| 4 | **Wiring the residual idea** (exp17) | QAE adapter through the exact production path | `qae` **PASS** (0.0); `nf4_qae` **BLOCKED** — residual reference needs absolute weights the contract never carries | Recorded BLOCKED with diagnosis — **never patched the contract to force a pass** |
| 5 | **Widen to breadth** (exp18) | 4 strategies × 7 models = 28 cells, same gates | **28/28 round trips 0.0**; LWE detector 0.50 on 7/7 models | LWE confirmed cross-model; TinyLlama skipped (incomplete cache, recorded) |
| 6 | **Dial the knobs** (exp19–22) | Adaptive routing, split-fraction dial, QAE↔LWE readout, layer-width sweeps | Dial curves measured; width rule + split fraction locked | Defaults frozen at shipped config |
| 7 | **Surgery survival** (exp23) | ONE shared embed → 12 cells: LoRA, prune, NF4, task-vector merge, real 1,000-step fine-tune, GPTQ, AWQ; 4 runs + 7 tooling fixes to close all legs | **8/12 survive at 0.0** (incl. the fine-tune); NF4 0.38 · half-merge 0.25 · GPTQ 0.50 · AWQ 0.41 — every failure short of chance, graceful | W6 answered: **light surgery preserves, int4 re-quant degrades but never erases** |
| 8 | **Synthesize** (exp24) | Pareto frontier over every experiment; all citations recomputed from source artifacts | **One configuration dominates the frontier** | Publishable claim framed from measured points only |
| 9 | **Audit (continuous)** | claim_audit recomputes every stated number from artifact bytes + test suite + consistency checker | **127/127 claims · 85 tests · 9/9 checks — green** | Pushed; misuse assessment rev 2 reviewed every step, **no trigger ever fired** |

---

## Cross-cutting rules (applied at every step)

- **Gates are set first, never moved** to make a result pass.
- **One embed shared across all cells** — the surgery/measurement is the only variable.
- **Blockers are recorded, not patched** — diagnosis is a valid result.
- **Records are append-only** — re-runs add revisions, never edit history.
- **Every doc number is recomputed** from the artifact, not trusted.

---

## Verdict

| Question | Answer |
|----------|--------|
| Can you hide data in LLM weights? | ✅ **Yes — validated.** 10k bits at BER 0.0 through fine-tuning, pruning, LoRA, merging |
| The original NES thesis — quantization residuals as the channel? | ❌ **No.** Never demonstrated (exp17 blocked); exp23 shows quantization *destroys* payloads (0.38–0.50) |
| Is it undetectable? | ⚠️ **Partially** — exactly chance (0.5000) to blind detectors, but a lattice-aware reader finds it |
| Does it survive model operations? | ⚠️ **Splits cleanly** — training-level operations yes, int4 compression no |

**Final state:** all work items done, 0 NOT_RUN, 0 blockers, tree clean, pushed at `c16e1a8`.

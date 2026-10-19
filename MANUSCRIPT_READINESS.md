# Manuscript Readiness — `interior_nearest` NF4 Anchor Preservation

Compiled 2026-10-16 from the frozen evidence package. This lists **only**
remaining publication blockers, ranked by importance.

## Can the novelty claim be defended from the evidence currently available?

**No.** No prior-art or literature review has been performed in this workspace.
There are no related-work notes, no cited papers, and no comparison against
existing implementations. The technical results are strong and verified, but
novelty is a *relative* claim that cannot be evaluated without the literature.
Any statement of novelty in the current draft would be unsupported.

This is not a finding that the work is un-novel; it is a finding that the
novelty question is currently unanswered.

## Ranked blockers

| Rank | Blocker | Why it blocks | Smallest action to clear it |
|---|---|---|---|
| **P1** | **No prior-art / literature review** | Novelty and related-work sections cannot be written; risk of re-describing a known technique | Search and read the candidate areas in `PAPER_OUTLINE.md` §2; cite actual papers; inspect implementations where available |
| **P2** | **Only two models; cross-model claim is thin** | "Cross-model" rests on 2 checkpoints; a third model would materially strengthen it | Add 1–2 additional public NF4 checkpoints and rerun the 6-tensor matrix (no new code needed) |
| **P3** | **Qwen revision provenance UNVERIFIED** | Reproducibility claim is weakened; reviewers may reject the Qwen cell | Establish whether a historical Qwen revision can be pinned; otherwise state the caveat prominently in the paper |
| **P4** | **Utility scope is narrow** | PPL measured on 1 tensor/layer, 15,330 tokens, 2 models; cannot support general utility preservation | Run PPL on more tensors and/or more tokens; a GPU would make the 100k-token target feasible |
| **P5** | **Detectability is inconclusive, not negative** | The strongest apparent result (small-cap AUC 0.534, p=0.005) rests on only 4 groups; the large-tensor chance result is sparsity-driven | Add independent tensors/checkpoints to raise the group count; preregister detectors before looking |
| **P6** | **No measured capacity ceiling** | Capacity claim is analytic plus sizes ≤ 50k bits | Measure recovery at or near the analytic bound on at least one tensor |
| **P7** | **Requantization depth limited to 3 cycles** | "Survives requantization" is shown for 3 cycles only | Extend the cycle count on one tensor; report the stability curve |
| **P8** | **No generation-quality / semantic check** | Limits what utility claims can be made (already not claimed, but reviewers will ask) | Optional: a small generation-quality probe; must be presented separately from PPL |

## What is already publication-ready

- The **mechanism result** (operation vs eligibility; Arm E) is clean, causal,
  and self-contained.
- The **recovery/robustness matrices** (2 models × 6 tensors, 0 interior
  failures, byte-idempotent) are verified against raw artifacts.
- The **utility** result is verified, honestly scoped, and correctly reported
  with paired CIs.
- The **detectability** work is rigorous in its negative/leakage analysis and is
  reported conservatively.

## What must never appear in the manuscript

- Claims of universal robustness, undetectability, semantic equivalence, or
  general utility preservation across all models/tensors.
- Any novelty statement until P1 is cleared.
- Any merging of the five residual methods or the QSE/DCE track into this
  paper's claims.

## Minimum viable submission path

1. Clear **P1** (literature review) — mandatory.
2. Clear **P3** (state or resolve the Qwen provenance caveat) — mandatory.
3. Clear **P2** (add a third model) — strongly recommended, cheap.
4. Present P4–P8 as explicit limitations if not cleared.

With P1–P3 addressed, the paper is defensible as a **scoped mechanism +
robustness + utility** contribution with honest limitations. Without P1, it is
not submittable.

---

## Addendum — status update (2026-10-18)

Reflects the third-model qualification and the prior-art completion. Original
rankings above are preserved; this section records what changed.

| Blocker | Prior status | Current status | Evidence |
|---|---|---|---|
| **P1** prior-art review | Not started | **Substantially addressed; novelty narrowed (2026-10-19)** | `overnight_third_model_prior_art/PRIOR_ART_REVIEW.md` + Addenda A–C; `PRIOR_ART_EVIDENCE_GATE_G_20261019.md`. Addendum C identifies **NID** (Liu et al. 2016) — code-index parity partition + neighbor-of-opposite-subcodebook move — as the same operation as `interior_nearest`, and HPQIM/Li-2023/FreMark as the invariant-region cousins. Novelty is now a narrow NF4-specific formulation + repeated-requantization exactness result. FreMark full text still paywalled. No "first" claim. |
| **P2** only two models | Open | **Cleared** | Third architecture StableLM-2-1.6B qualified: 360/360 exact, anchors 0, detectability at chance; `THIRD_MODEL_VALIDATION.md` + `stablelm_qualification/ROUNDTRIP_STABLELM.md` |
| **P5** detectability inconclusive | Open | **Improved, still bounded** | StableLM screening pass (6 groups, AUC ~0.500); small-capacity signal (AUC 0.534, p=0.005, 4 groups) preserved unchanged. Group count still limits power. |
| **P4** utility narrow | Open | **Partially addressed (StableLM)** | StableLM utility run at 296,380 tokens/condition (>100k target), 3 seeds, 2 sizes; see `stablelm_qualification/STABLELM_QUALIFICATION_REPORT.md` |
| **P3** Qwen provenance | Open | Open | Unchanged; StableLM revision is pinned (`f499ead7...`) |
| **P6** capacity ceiling | Open | Open | Unchanged |
| **P7** requant depth | Open | Partially | cycles 0-5 covered by regression test on StableLM; matrix is 0-3 |
| **P8** generation quality | Open | Open | Unchanged |

Minimum viable path update: P1 and P2 are now addressed to a defensible level;
P3 remains mandatory; P4/P5 improved for StableLM. The remaining hard novelty
risk (QIM-equivalence / FreMark full text) must be disclosed, not resolved by
assertion.

---

## Addendum 2 — Gate E closed (2026-10-19)

Gate E (utility ±0.5% equivalence) for StableLM-2-1.6B is **complete and PASS**.

- Run: `stablelm_qualification/utility_v2/` (7 conditions: clean + 6 embedded).
- Re-analysis: `utility_v2/gate_e_reanalysis.json` — `all_within_margin: true`,
  `n_within_margin: 6/6`.
- Largest effect 0.0139% rel. PPL; largest CI half-width 0.037% vs the ±0.5%
  margin. 296,380 tokens/condition, 3 seeds, sizes 10k/50k, q_proj.
- Manuscript table emitted: `stablelm_qualification/tables/table4_utility_stablelm.csv`
  (+ `table4_utility_meta.json`).

Effect on blockers: **P4** (utility narrow) moves from "partially addressed" to
addressed for StableLM — one model, one tensor, one corpus, but with a
predeclared margin and an equivalence test, not a point estimate. P4 remains
open for the other models/tensors. All other blockers unchanged.

With Gate E closed, the only gate not in PASS is **G** (novelty/prior-art), which
is an evidence limitation rather than a method failure. Per the frozen protocol
(§3), **expansion remains locked** until every gate passes, so Gate E closing does
**not** by itself authorize multi-model expansion.

Gate G was re-examined on 2026-10-19 (Addendum C / evidence log). The finding is
that the novelty is *narrower* than previously stated: the code-index
parity-partition + neighbor-of-opposite-subcodebook operation is a named published
technique (**NID**, Liu et al. 2016), and the invariant-region idea is established
(HPQIM, Li 2023, FreMark). Gate G remains **PARTIAL**. Because the protocol gates
expansion on *all* gates passing, and because Gate G is an honest scientific
judgement that cannot be forced to PASS by more searching alone, the correct next
action is a **decision point**, not a run: either (i) amend the protocol to permit
expansion with the narrow-novelty limitation disclosed in the manuscript, or
(ii) reframe the contribution from "new method" to "NF4-specific formulation +
repeated-requantization robustness result" and target a venue that accepts
narrow-engineering contributions. This requires the user's call.


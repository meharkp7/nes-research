# NES Research — Final Scientific Validation & Submission Cleanup
## Audit Plan & Critical Issues Found

### Critical Issue A: Held-Out Neural Detector — All-Positive Prediction Collapse
**Evidence**: `evidence/heldout_neural_v2/heldout_neural_results.json`
- Confusion counts for ALL 4 models: TN=0, FN=0, FP=12288, TP=73728
- Test set composition: 73,728 embedded vs 12,288 clean (6:1 ratio)
- Accuracy = 85.7% = class ratio → model predicts ALL as embedded
- AUC ≈ 0.5 computed from continuous probabilities, but threshold 0.5 collapses
- **Root cause**: Severe class imbalance in both train (3:1) and test (6:1)
- **Action**: Re-run with balanced test composition + class-weighted BCE loss

### Critical Issue B: Track C Utility — Documentation Error (Not Actual Blocker)
**Evidence**: `expansion_qualification/*/utility_v1/comparisons.json` (4 models)
- All 24 conditions (4 models × 2 sizes × 3 seeds) show `within_margin_0p5pct: true`
- Max |Δ| = 0.019% (Pythia), max CI half-width = 0.0191%
- CLAIMS_AUDIT.md incorrectly labels as "BLOCKED" — this was Track A issue conflated
- **Action**: Correct documentation; Track C utility is VALID

### Critical Issue C: Small-Capacity Evidence — Two Experiments Conflated
**Evidence**: 
- Original 4-group: `out/detectability_smallcap_v1/supplement_results.json` — LR AUC 0.5343
- Extended 12-group: `evidence/smallcap_extended_v1/supplement_results.json` — LR AUC 0.5167
- Merged 14-group: `out/detectability_smallcap_analysis_v2/detector_results.json` — LR AUC 0.5059
- OPEN_GAPS.md says 14 groups completed; manifest says 12 groups; results table has both
- **Action**: Designate extended 12-group as primary supplement; archive original 4-group and merged 14-group as historical

### Critical Issue D: 10-Cycle Requantization — Denominator Clarification Needed
**Evidence**: `evidence/requant_10cycle/requant_10cycle_results.json`
- 4 models × 2 sizes × 3 seeds × 11 cycles = 264 conditions, all PASS
- BUT manifest shows only 1 tensor per model (k_proj), not 6 tensors
- Manuscript claims "four expansion models × 11 cycles = 264/264 PASS" — correct for actual experiment
- **Action**: Clarify in manuscript this is single-tensor stress test (k_proj only), not full 6-tensor matrix

---

## Execution Order

1. **Fix held-out neural detector** (Issue A) — re-run with class weighting + balanced test
2. **Correct Track C utility documentation** (Issue B) — update CLAIMS_AUDIT.md, OPEN_GAPS.md
3. **Reconcile small-capacity evidence** (Issue C) — designate primary, archive others
4. **Clarify 10-cycle denominator** (Issue D) — update manuscript and evidence table
5. **Run regression tests** to verify codec correctness
6. **Produce FINAL_SUBMISSION_GATE.md**

---

## Commands to Run

```bash
# 1. Run regression tests on all 5 models (verify 60/60 PASS)
# 2. Re-run held-out neural detector with class weighting
# 3. Update documentation files
# 4. Generate final gate report
```
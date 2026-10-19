# Frozen-manifest reconciliation report (2026-10-19)

Scope: reconcile `FROZEN_ALL_HASHES.txt` against the current tree **without
modifying anything**. The original manifest and every changed file are preserved
as-is; no hash was regenerated or overwritten.

Method: `expansion_qualification/reconcile_manifest.py` (read-only) hashes each of
the manifest's 271 listed paths and classifies it as raw artifact vs
narrative/metadata. Machine output:
`expansion_qualification/manifest_reconciliation.json`.

## Summary

| Quantity | Value |
|---|---|
| Hashed entries listed | 271 |
| Re-verified MATCH | 265 |
| CHANGED | 6 |
| MISSING | 0 |
| Annotation (non-hash) lines | 3 |
| **Changed raw artifacts** | **0** |
| Changed narrative/metadata files | 6 |

The raw-evidence subset of the manifest re-verifies clean. Every changed path is
a narrative or metadata document; **no** `out/…`, `…/conditions/…`,
`…/checkpoints/…`, `…/nll/…`, result JSON, or model artifact changed.

## The six changed paths

| Path | Kind | Recorded sha256 (manifest) | Current sha256 | mtime (UTC) |
|---|---|---|---|---|
| `FROZEN_ALL_HASHES.txt` | narrative (manifest itself) | `4d379eb00811…5da1e01` | `625de8f8867b…c367c1d2c5` | 2026-10-17T21:02:38 |
| `PAPER_OUTLINE.md` | narrative | `2cfd12e84893…e3ba30bd` | `4e36e1b9f305…ca4f3840` | 2026-10-19T10:10:47 |
| `MANUSCRIPT_READINESS.md` | narrative | `5cb0adce2643…eb1bb589` | `e1885027fff1…29024c88` | 2026-10-19T10:09:44 |
| `RESULTS_EVIDENCE_LEDGER.md` | narrative | `9784543a4ef2…b9063778` | `a3c6ad3fcc8d…677ce266` | 2026-10-19T10:09:51 |
| `overnight_third_model_prior_art/PRIOR_ART_REVIEW.md` | narrative | `475edaddd443…3573562a` | `106932932c2d…fbf9fbea` | 2026-10-19T10:09:23 |
| `overnight_third_model_prior_art/PRIOR_ART_BIBLIOGRAPHY.bib` | narrative | `95efaf1dfc63…c6fb113` | `e957335ad7bb…1c356b7d` | 2026-10-17T21:40:00 |

Full hashes: see `manifest_reconciliation.json` (`rows`).

## Self-hash finding

The manifest lists its own hash (line 84) as
`4d379eb00811139a153016cb9b7ca65f886f2ec4331e5cadf8442a6075da1e01`, but the
current file hashes to `625de8f8867b06ccdbcd28e0d2672d760d98856f2701f6232c8463c367c1d2c5`.

This is **not** a benign self-reference artifact: reconstructing the file with
the self-hash line removed (`sha256` = `bab275863b0bb4b1…`) does not reproduce the
recorded value, and no simple transform (dropping the self line, dropping the
three annotation lines, trailing-newline variants) reproduces it. Conclusion:
**the manifest content was edited after its self-hash was recorded.** The
recorded self-hash therefore cannot vouch for the current manifest body.

## Provenance / attribution

- The five changed documents have mtimes of 2026-10-17 21:40 and 2026-10-19
  10:09–10:10. This expansion session's artifacts all begin at 2026-10-19
  10:5x and later. The changes therefore **pre-date** this session and are
  attributed to the immediately preceding documentation workstream, not to the
  expansion.
- No prior copy of any of the six files exists anywhere under `/workspace` (no
  `.bak`, `.orig`, or alternate path), and none is tracked in the `/workspace/project`
  git clone. The recorded pre-change hashes are the **only** surviving record of
  the earlier revisions; the earlier contents are not recoverable here.

## Disposition (no silent repair)

Per the operating contract, this report **does not** regenerate or overwrite any
historical hash, and does not modify the manifest or the changed files. Status:
**disclosed, unresolved**.

Recommended resolution (maintainer):
1. Decide whether the five narrative docs' current revisions are intended. If so,
   record a new manifest generation (e.g. `FROZEN_ALL_HASHES_v2.txt`) rather than
   editing the frozen one.
2. Recompute and record the manifest self-hash in a way that is defined (state
   explicitly that the self-hash is computed over the file excluding the self-hash
   line), or drop the self-hash line to avoid a chicken-and-egg entry.
3. Keep this report attached to the ledger so the discrepancy is visible.

## Effect on this project's conclusions

None on Track A: every raw artifact re-verifies. The affected files are narrative
manuscript/prior-art documents; the prior-art conclusion (Gate G PARTIAL) rests on
`PRIOR_ART_EVIDENCE_GATE_G_20261019.md` and cited literature, which are separately
preserved. The discrepancy is a bookkeeping-integrity item, reported as such.

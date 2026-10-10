# Packed-NF4 Detector Readiness Gate

This gate validates dataset structure before fitting any detector. It does not train
or score a classifier and does not establish stealth.

Run from `nes-llm/`:

```bash
python scripts/validate_packed_nf4_detector_readiness.py \
  --dataset /path/to/grouped_dataset.csv \
  --splits /path/to/split_manifest.json \
  --output /path/to/new_timestamped/readiness.json
```

The output is `PASS` only if artifact identities are consistent, each matched
source/run/model/tensor group has clean and embedded rows with matching block
indices, each model stays in its declared split, the split manifest agrees with
the CSV, every split has both labels, and each model contains both classes.
Default minimum independent model groups are 2 train / 1 validation / 2 test.
These are a minimal execution gate, not a guarantee of adequate statistical power.

The existing 3-model dataset is expected to be blocked by the default test-group
minimum: it has only one held-out model group. Do not lower the minimum merely to
make the current dataset pass. Add independent source models/runs, then rerun the
gate. Block rows are not independent samples, and a PASS does not establish
independence or detector validity.

Reports must be written to new paths; the script refuses to overwrite them.

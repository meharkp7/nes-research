# Contract B B1.4 — 10,000-bit artifact-only NF4 pilot

## Scope and caveats

B1.4 tests whether a 10,000-bit application payload can be embedded in packed NF4 codes of one selected tensor and recovered from the serialized artifact using a fixed test key and protocol, without original FP16 weights, residual cache, or delta sidecar. The key and synthetic payload are test fixtures. The truncated SHA-256 envelope checksum detects accidental corruption but is not a cryptographic authentication mechanism. Do not claim confidentiality, stealth, utility preservation, robustness, or novelty from this pilot.

## Run from `nes-llm/`

Sender:

```bash
../.venv/bin/python scripts/contract_b_nf4_b14.py embed \
  --original ../cache/contract_b_nf4_probe_retry \
  --output-dir ../cache/contract_b_nf4_b14_10k
```

Inspect `../cache/contract_b_nf4_b14_10k_b14_sender_report.json`, then copy `payload_sha256` into:

```bash
../.venv/bin/python scripts/contract_b_nf4_b14.py receive \
  --stego-dir ../cache/contract_b_nf4_b14_10k \
  --output-payload ../cache/contract_b_nf4_b14_recovered.bin \
  --expected-sha256 PASTE_SENDER_PAYLOAD_SHA256_HERE
```

The receiver runs in a separate process and reloads the stego model on CPU by default. Do not use `--skip-model-reload` for an end-to-end PASS.

## Automated tests

From `nes-llm/`:

```bash
../.venv/bin/python -m unittest discover -s tests -p 'test_contract_b_nf4_b14.py' -v
```

The CI suite covers deterministic payload/envelope accounting, carrier selection, nibble ordering, exact 10k recovery, pair-ID invariants, wrong-key rejection, corruption rejection, and synthetic SafeTensors parsing. CI also syntax-checks the Contract B scripts. These tests do not replace the actual cached-model run.

## Phase 2 — capacity sweep

After B1.4 succeeds on the actual cached model:

```bash
../.venv/bin/python scripts/contract_b_capacity_sweep.py \
  --checkpoint ../cache/contract_b_nf4_probe_retry \
  --sizes-bits 1000 10000 25000 50000 --repeats 3
```

The report `../cache/contract_b_b14_capacity_sweep.json` records per-run exact recovery, bit error rate, carrier count, and changed-code count. This is a packed-code mechanism sweep, not a utility or stealth result.

## Phase 3 — utility diagnostic

After the B1.4 receiver and reload checks pass:

```bash
../.venv/bin/python scripts/contract_b_utility_eval.py \
  --original ../cache/contract_b_nf4_probe_retry \
  --stego ../cache/contract_b_nf4_b14_10k --device cpu --max-tokens 256
```

This is deliberately labelled an exploratory fixed-text diagnostic, not a benchmark. A publication-quality utility result needs a larger held-out corpus, baseline/control variants, uncertainty, and a predeclared acceptance criterion.

## Remaining phases

1. Finish local B1.4 sender/receiver and negative tests.
2. Run repeated capacity sweep and preserve all failures.
3. Expand utility evaluation to a suitable held-out corpus and task-level checks.
4. Evaluate detectability with pre-registered baselines, held-out data and uncertainty.
5. Evaluate robustness under reload, fine-tuning, LoRA, pruning, merges and re-quantization.
6. Expand across models only after the single-model path is stable.
7. Complete literature review, ablations, claim audit, reproducibility bundle and paper.

## Execution boundary

GitHub source access does not provide access to the user's local model cache or Mac runtime. The actual Qwen NF4 reload, capacity sweep and model-utility run must be executed locally; source review, synthetic tests, tracked artifact review and analysis can be done separately.

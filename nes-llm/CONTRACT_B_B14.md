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
../.venv/bin/python -m unittest discover -s tests -p 'test_contract_b_robustness_matrix.py' -v
```

The B1.4 unit suite covers deterministic payload/envelope accounting, carrier selection, nibble ordering, exact 10k recovery, pair-ID invariants, wrong-key rejection, corruption rejection, capacity-payload generation, and synthetic SafeTensors parsing. The robustness-matrix suite covers locality under non-carrier parity mutations and checksum rejection under carrier corruption. CI syntax-checks all Contract B scripts. Synthetic tests do not replace the actual cached-model run.

## Phase 2 — capacity sweep

After B1.4 succeeds on the actual cached model:

```bash
../.venv/bin/python scripts/contract_b_capacity_sweep.py \
  --checkpoint ../cache/contract_b_nf4_probe_retry \
  --sizes-bits 1000 10000 25000 50000 --repeats 3
```

The report `../cache/contract_b_b14_capacity_sweep.json` records per-run exact recovery, bit error rate, carrier count, and changed-code count. This is a packed-code mechanism sweep, not a utility or stealth result.

## Phase 3 — utility diagnostic

Use the matching cached tokenizer explicitly because the tokenizer files inside the probe checkpoint were invalid:

```bash
../.venv/bin/python scripts/contract_b_utility_eval.py \
  --original ../cache/contract_b_nf4_probe_retry \
  --stego ../cache/contract_b_nf4_b14_10k \
  --tokenizer-path Qwen/Qwen2.5-3B \
  --device cpu --max-tokens 256
```

This is a fixed, small diagnostic text suite, not a benchmark. A publication-quality utility result needs a larger held-out corpus, baseline/control variants, uncertainty, and a predeclared acceptance criterion.

## Phase 4 — descriptive detectability diagnostic

```bash
../.venv/bin/python scripts/contract_b_detectability_diagnostic.py \
  --original ../cache/contract_b_nf4_probe_retry \
  --stego ../cache/contract_b_nf4_b14_10k
```

This reports packed-code histograms, total-variation distance, KL divergence, LSB balance and changed-code localization. It is not a trained detector and cannot establish stealth or attacker success probability.

## Phase 5 — B1.4 packed-code robustness diagnostic

```bash
../.venv/bin/python scripts/contract_b_robustness_matrix.py \
  --stego ../cache/contract_b_nf4_b14_10k
```

Default cases flip the parity bit of 1, 100, and 1,000 non-carrier codes (expected to preserve recovery), then corrupt 1, 10, and 100 payload/envelope carriers after the header (expected to trigger checksum rejection). The report is `../cache/contract_b_b14_robustness_matrix.json`. This measures carrier locality and error detection under controlled packed-code mutations only. It does not simulate real training, pruning, adapter/task-vector merging, or requantization.

## Lifecycle robustness — separate evidence, not interchangeable

The repository's `results/exp23_model_surgery_qwen__qwen2.5_3b.json` measures the earlier residual-domain sign embedding, where extraction occurs in residual space. It reports exact recovery after its specified fine-tuning, LoRA-shaped updates, pruning and small merges, but substantial BER after NF4/GPTQ/AWQ requantization and a large merge. Those results are important historical evidence, but **they are not B1.4 packed-NF4 carrier results** and must not be presented as such.

The next lifecycle experiment must transform a fresh copy of the B1.4 stego checkpoint, retain each transformed artifact, then run the B1.4 artifact receiver against each output. Test each axis independently; record failures rather than patching them away. Do not overwrite the pristine `contract_b_nf4_b14_10k` directory.

## Remaining phases

1. Preserve the B1.4 sender, receiver, capacity, utility, and detectability reports.
2. Run and preserve the packed-code robustness diagnostic above.
3. Run B1.4-specific lifecycle transformations on separate artifact copies: reload/save round-trip, pruning, fine-tuning/LoRA merge, task-vector merge, and requantization.
4. Expand utility evaluation to a suitable held-out corpus and task-level checks.
5. Evaluate detectability with pre-registered baselines, held-out data and uncertainty.
6. Expand across models only after the single-model path is stable.
7. Complete literature review, ablations, claim audit, reproducibility bundle and paper.

## Execution boundary

GitHub source access does not provide access to the user's local model cache or Mac runtime. Actual Qwen NF4 reloads, transformations, capacity sweep, utility and detectability runs must be executed locally; source review, synthetic tests, tracked artifact review and analysis can be done separately.

## Phase 6 — real NF4 requantization lifecycle test

This test is deliberately different from flipping packed-code bits. It loads the B1.4 stego model as NF4, dequantizes model weights, saves a floating-point intermediate, reloads that intermediate with a fresh NF4 quantization pass, and saves a new quantized checkpoint. It preserves the pristine source and refuses to overwrite existing output/intermediate directories.

Check available disk space first: this operation can require substantial temporary storage and RAM. It is not a lightweight diagnostic. Run from `nes-llm/`:

```bash
../.venv/bin/python -m unittest discover -s tests -p 'test_contract_b_nf4_requantization.py' -v

../.venv/bin/python scripts/contract_b_nf4_requantization.py \
  --stego ../cache/contract_b_nf4_b14_10k \
  --output-dir ../cache/contract_b_b14_nf4_requantized
```

The script creates the intermediate at `../cache/contract_b_b14_nf4_requantized_fp16_intermediate` and writes `../cache/contract_b_b14_nf4_requantized_requantization_report.json`. It requires both paths not to exist. It uses local files only and does not access the original non-stego checkpoint. If the process fails, preserve the report and any intermediate/output artifacts for diagnosis rather than deleting them and rerunning blindly.

Then independently run the B1.4 receiver against the new output:

```bash
PAYLOAD_SHA=$(../.venv/bin/python -c 'import json; print(json.load(open("../cache/contract_b_nf4_b14_10k_b14_sender_report.json"))["payload_sha256"])')

../.venv/bin/python scripts/contract_b_nf4_b14.py receive \
  --stego-dir ../cache/contract_b_b14_nf4_requantized \
  --output-payload ../cache/contract_b_b14_nf4_requantized_recovered.bin \
  --expected-sha256 "$PAYLOAD_SHA"
```

Interpretation:
- `PASS_TRANSFORMATION_AND_RECOVERY`: fresh NF4 quantization completed and exact payload recovery survived.
- `TRANSFORMATION_COMPLETED_RECOVERY_FAILED`: quantization completed, but the carrier payload did not survive. The report includes BER when the envelope header remains readable.
- `TRANSFORMATION_ERROR`: the transformation did not complete; this is not evidence either for or against payload survival.

The report's carrier BER diagnostic and the independent receiver must agree before marking the result final. Do not infer utility preservation from recovery, or recovery robustness from the small mutation matrix. A new NF4 quantization result applies only to this checkpoint, configuration, and run.

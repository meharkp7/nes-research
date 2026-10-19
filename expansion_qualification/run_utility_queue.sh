#!/usr/bin/env bash
# Sequential utility (A5) runner for the expansion models. Waits for any running
# utility job to finish, then runs each model's paired-PPL evaluation one at a
# time (CPU-bound). New output dirs only; never overwrites existing results.
set -u
cd /workspace/nes_run_20261016_anchor_ablation
PY=/workspace/nes_env/bin/python
EXP=expansion_qualification

wait_for_utility() {
  while pgrep -f "exp_utility.py" > /dev/null; do sleep 60; done
}

run_one() {
  local key="$1" ckpt="$2" tok="$3" mid="$4" rev="$5" tensor="$6"
  local out="$EXP/$key/utility_v1"
  if [ -f "$out/comparisons.json" ]; then
    echo "[queue] $key utility already complete; skipping"
    return 0
  fi
  echo "[queue] starting $key utility"
  "$PY" "$EXP/exp_utility.py" \
    --checkpoint-clean "$ckpt" --tokenizer-dir "$tok" --output "$out" \
    --model-id "$mid" --revision "$rev" --run-id "nes_${key}_utility_20261019" \
    --tensor "$tensor" --n-seq 580 --batch 4 --sizes 10000 50000 \
    --seeds 20261016 20261017 20261018 \
    > "$EXP/$key/utility.log" 2>&1
  echo "[queue] finished $key utility rc=$?"
}

wait_for_utility
run_one qwen15 ckpt/qwen15_clean /tmp/nes_stage/qwen15_fp16 \
  Qwen/Qwen2.5-1.5B 8faed761d45a263340a0528343f099c05c9a4323 \
  model.layers.0.self_attn.q_proj.weight
run_one pythia /tmp/nes_stage/pythia_clean /tmp/nes_stage/pythia_fp16 \
  EleutherAI/pythia-1.4b fedc38a16eea3bd36a96b906d78d11d2ce18ed79 \
  gpt_neox.layers.0.attention.query_key_value.weight
run_one falcon /tmp/nes_stage/falcon_clean /tmp/nes_stage/falcon_fp16 \
  tiiuae/Falcon3-1B-Base cb37ef3559b157b5c9d9226296ba01a5162da1f7 \
  model.layers.0.self_attn.q_proj.weight
echo "[queue] all utility runs done"

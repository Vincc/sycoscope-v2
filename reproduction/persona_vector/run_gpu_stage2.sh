#!/bin/bash
# GPU queue after the extraction rollouts: sycophantic vectors (float32, bf16), then the even-layer steering sweep.
set -euo pipefail
cd "$(dirname "$0")/../.."
set -a; . ./.env; set +a
P=reproduction/persona_vector
while tmux has-session -t =gen 2>/dev/null; do sleep 30; done
for dt in float32 bfloat16; do
  [ -f $P/vectors/sycophantic_$dt.npz ] || .venv/bin/python -m reproduction.persona_vector.extract_vectors --trait sycophantic \
    --rollouts $P/data/rollouts/extract_sycophantic.jsonl --scores $P/data/judge/extract/extract_sycophantic.scores.jsonl \
    --dtype $dt --batch-size 16 --out-vector $P/vectors/sycophantic_$dt.npz --out-acts $P/activations/sycophantic_${dt}_acts.npz
done
cells="0:0"
for l in 4 6 8 10 12 14 16 18 20 22 24 26 28; do for c in 0.4 0.8 1.2; do cells="$cells $l:$c"; done; done
.venv/bin/python -m reproduction.persona_vector.generate --mode steer --prompts $P/data/prompts/steer_eval.jsonl \
  --out-dir $P/data/rollouts/steer --vector $P/vectors/sycophantic_float32.npz --cells $cells --batch-size 100 --seed 0

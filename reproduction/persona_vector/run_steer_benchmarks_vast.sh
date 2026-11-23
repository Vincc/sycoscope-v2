#!/bin/bash
# GPU box, Llama-3.1-8B: persona-vector steering sweep (paper layer 16) on the OEQ and Are You Sure TriviaQA steering
# items, with the benchmark decoding (T 0.6, top-p 0.9, 1024 new tokens). A rerun skips finished cells.
# Needs: uv sync done, HF_TOKEN with meta-llama access in .env, BATCH_SIZE set for the GPU.
set -euo pipefail
cd "$(dirname "$0")/../.."
set -a; . ./.env; set +a
: "${BATCH_SIZE:?set BATCH_SIZE}"
S=generations/Llama-3.1-8B-Instruct/t0.6_p0.9/steer_persona_sycophantic
GEN="--vector reproduction/persona_vector/vectors/sycophantic_float32.npz --layer 16 --out-dir $S/responses \
  --batch-size $BATCH_SIZE --seed 0 --temperature 0.6 --top-p 0.9 --top-k 0 --max-new-tokens 1024"

.venv/bin/python -m pytest tests/test_steer_benchmarks.py -q
for items in $S/items/are_you_sure_trivia_qa_n400.jsonl $S/items/OEQ_n300.jsonl; do
  .venv/bin/python -m reproduction.persona_vector.steer_benchmarks --items $items $GEN --coefs 0 2 -2 1 -1 1.5 -1.5 0.5 -0.5
  .venv/bin/python -m reproduction.persona_vector.steer_benchmarks --items $items $GEN --coefs 2 --random-direction-seed 0
done

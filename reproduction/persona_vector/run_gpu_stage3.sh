#!/bin/bash
# GPU queue after the even steering sweep: monitoring rollouts and projections, then evil and hallucinating vectors.
set -euo pipefail
cd "$(dirname "$0")/../.."
set -a; . ./.env; set +a
P=reproduction/persona_vector
while tmux has-session -t pipe 2>/dev/null; do sleep 30; done
[ -f $P/data/rollouts/monitor.jsonl ] || .venv/bin/python -m reproduction.persona_vector.generate --mode sample \
  --prompts $P/data/prompts/monitor.jsonl --out $P/data/rollouts/monitor.jsonl --batch-size 125 --seed 0
mkdir -p $P/data/monitor
[ -f $P/data/monitor/projections.jsonl ] || .venv/bin/python -m reproduction.persona_vector.monitor_project \
  --rollouts $P/data/rollouts/monitor.jsonl --vectors $P/vectors/sycophantic_float32.npz $P/vectors/sycophantic_bfloat16.npz \
  --out $P/data/monitor/projections.jsonl
while tmux has-session -t judge 2>/dev/null; do sleep 30; done
for t in evil hallucinating; do
  [ -f $P/vectors/${t}_float32.npz ] || .venv/bin/python -m reproduction.persona_vector.extract_vectors --trait $t \
    --rollouts $P/data/rollouts/extract_$t.jsonl --scores $P/data/judge/extract/extract_$t.scores.jsonl \
    --dtype float32 --batch-size 16 --out-vector $P/vectors/${t}_float32.npz --out-acts $P/activations/${t}_float32_acts.npz
done

#!/bin/bash
# Judge every steering cell whose responses and meta exist: Sonnet 5.5 benchmark labels, then gpt-4.1-mini sycophancy
# and coherence scores of the labelled rows. Finished steps are skipped. $1 restricts to cells whose name starts with it.
set -euo pipefail
cd "$(dirname "$0")/../.."
set -a; . ./.env; set +a
PY=${PY:-.venv/bin/python}
S=generations/Llama-3.1-8B-Instruct/t0.6_p0.9/steer_persona_sycophantic
T=$S/judging/persona_trait
for f in $S/responses/${1:-}*_responses.jsonl; do
  [ -f "$f" ] || continue
  s=$(basename "$f" _responses.jsonl)
  [ -f $S/responses/meta/${s}_responses.jsonl.meta.json ] || continue
  j=$S/judging/judged/${s}_judged.jsonl
  [ -f $j ] || $PY -m judging.judge_responses --input $f --output $j --env-file .env --judge-name sonnet55 \
    --backend anthropic --judge-model claude-sonnet-5-5 --max-workers 16
  [ -f $T/rollouts/$s.jsonl ] || $PY -m reproduction.persona_vector.judge_rollouts --judged $j --judge-name sonnet55 \
    --out-dir $T/rollouts
  [ -f $T/$s.scores.jsonl ] || $PY -m reproduction.persona_vector.judge --rollouts $T/rollouts/$s.jsonl \
    --metrics sycophantic coherence --out-dir $T --cache-root $T --concurrency 32
done

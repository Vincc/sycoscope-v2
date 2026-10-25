#!/bin/bash
# Judge each steering cell (trait + coherence) once its rollout file and meta exist; stops after the GPU queue (pipe3) ends.
set -euo pipefail
cd "$(dirname "$0")/../.."
set -a; . ./.env; set +a
P=reproduction/persona_vector
judge_new() {
  for f in $P/data/rollouts/steer/L*_c*.jsonl; do
    [ -f "$f" ] || continue
    s=$(basename "$f" .jsonl)
    [ -f $P/data/rollouts/steer/meta/$s.jsonl.meta.json ] || continue
    [ -f $P/data/judge/steer/$s.scores.jsonl ] && continue
    .venv/bin/python -m reproduction.persona_vector.judge --rollouts "$f" --metrics sycophantic coherence \
      --out-dir $P/data/judge/steer --concurrency 32 2>&1 | grep -v -i "http request"
  done
}
while tmux has-session -t =pipe3 2>/dev/null; do judge_new; sleep 60; done
judge_new

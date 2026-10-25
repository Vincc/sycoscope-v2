#!/bin/bash
# Judge the evil and hallucinating extraction rollouts as each file appears.
set -euo pipefail
cd "$(dirname "$0")/../.."
set -a; . ./.env; set +a
P=reproduction/persona_vector
for t in evil hallucinating; do
  until [ -f $P/data/rollouts/meta/extract_$t.jsonl.meta.json ]; do sleep 30; done
  .venv/bin/python -m reproduction.persona_vector.judge --rollouts $P/data/rollouts/extract_$t.jsonl \
    --metrics $t coherence --out-dir $P/data/judge/extract --concurrency 32 2>&1 | grep -v -i "http request"
done

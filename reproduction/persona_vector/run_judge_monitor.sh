#!/bin/bash
# Judge the monitoring rollouts (trait + coherence) once their file and meta exist.
set -euo pipefail
cd "$(dirname "$0")/../.."
set -a; . ./.env; set +a
P=reproduction/persona_vector
until [ -f $P/data/rollouts/meta/monitor.jsonl.meta.json ]; do sleep 30; done
.venv/bin/python -m reproduction.persona_vector.judge --rollouts $P/data/rollouts/monitor.jsonl --metrics sycophantic coherence \
  --out-dir $P/data/judge --concurrency 32 2>&1 | grep -v -i "http request"

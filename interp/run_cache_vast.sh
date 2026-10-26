#!/bin/bash
# GPU box: cache tests, parity on 64 SS rows, the ELEPHANT activation cache for Llama-3.1-8B, upload to the Hub.
# Needs the repo on this branch with `git lfs pull` done, uv sync done, and HF_TOKEN in .env (meta-llama + SycoScope).
# BATCH_SIZE=8 fits rows of 3.7k tokens on an 80 GB GPU; use 2-4 on 24 GB.
set -euo pipefail
cd "$(dirname "$0")/.."
set -a; . ./.env; set +a
MODEL=meta-llama/Llama-3.1-8B-Instruct
GEN=generations/Llama-3.1-8B-Instruct/responses
CACHE=activations/cache
POSITIONS="prompt_last prompt_mean response_mean response_first5 response_first10 response_frac10 response_frac25 response_frac50 response_frac75"
mkdir -p $CACHE

.venv/bin/python -m pytest tests/test_cache_activations.py tests/test_extraction.py -q
.venv/bin/python -m interp.check_cache_parity --model $MODEL --input $GEN/ss_responses.jsonl --n-rows 64 --max-length 4096 \
  2>&1 | tee activations/parity.log
.venv/bin/python -m interp.cache_activations --model $MODEL --positions $POSITIONS --cache-dir $CACHE \
  --max-length 4096 --batch-size ${BATCH_SIZE:-8} \
  --inputs $GEN/aita_yta_responses.jsonl $GEN/oeq_responses.jsonl $GEN/ss_responses.jsonl \
           $GEN/aita_nta_og_responses.jsonl $GEN/aita_nta_flip_responses.jsonl 2>&1 | tee activations/cache.log
.venv/bin/hf upload SycoScope/activation-cache $CACHE . --repo-type dataset --exclude "*.tmp" \
  --commit-message "Llama-3.1-8B ELEPHANT activations, $(git rev-parse --short HEAD)"

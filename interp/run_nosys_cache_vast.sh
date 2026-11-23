#!/bin/bash
# GPU box, Llama-3.1-8B: cache prompt_last and response_mean of the tefo contrastive training responses twice, with
# each row's system prompt and without it (the chat template's default system block stays), and upload both to the
# Hub. Generates nothing; a rerun computes only missing (row, position) pairs.
# Needs: this branch with `git lfs pull` done and a clean working tree, uv sync done, HF_TOKEN in .env
# (meta-llama + SycoScope), CACHE_BATCH_SIZE set for the GPU, and about 25 GB free disk.
set -euo pipefail
cd "$(dirname "$0")/.."
set -a; . ./.env; set +a
: "${CACHE_BATCH_SIZE:?set CACHE_BATCH_SIZE}"
if [ -n "$(git status --porcelain)" ]; then git status --short; echo "working tree is dirty; stopping"; exit 1; fi
MODEL=meta-llama/Llama-3.1-8B-Instruct
INPUT=generations/Llama-3.1-8B-Instruct/training/synthetic_dataset/synthetic_trait_emotion_fact_opinion/synthetic_trait_emotion_fact_opinion_contrastive_responses.jsonl
CACHE=activations/cache
POSITIONS="prompt_last response_mean"
mkdir -p $CACHE

.venv/bin/python -m pytest tests/test_nosys_cache.py tests/test_cache_activations.py tests/test_extraction.py -q
# cache vs get_activations and the persona-vector extraction on 64 training rows, with the system prompt
.venv/bin/python -m interp.check_cache_parity --model $MODEL --input $INPUT --n-rows 64 --max-length 4096 \
  2>&1 | tee activations/parity_tefo.log
.venv/bin/python -m interp.cache_activations --model $MODEL --positions $POSITIONS --cache-dir $CACHE \
  --max-length 4096 --batch-size $CACHE_BATCH_SIZE --inputs $INPUT 2>&1 | tee activations/cache_tefo.log
.venv/bin/python -m interp.cache_activations --model $MODEL --positions $POSITIONS --cache-dir $CACHE \
  --max-length 4096 --batch-size $CACHE_BATCH_SIZE --inputs $INPUT --drop-system-prompt \
  2>&1 | tee activations/cache_tefo_nosys.log
# upload only the training folder, so other local cache folders never reach the Hub
.venv/bin/hf upload SycoScope/activation-cache $CACHE/Llama-3.1-8B-Instruct/training Llama-3.1-8B-Instruct/training \
  --repo-type dataset --exclude "*.tmp" \
  --commit-message "Llama-3.1-8B tefo contrastive activations with and without the system prompt, $(git rev-parse --short HEAD)"

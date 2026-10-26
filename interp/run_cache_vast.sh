#!/bin/bash
# GPU box, Llama-3.1-8B: regenerate Are You Sure responses and push them, then cache ELEPHANT and Are You Sure
# activations and upload them to the Hub. Each step is skipped or resumed if its output already exists.
# Needs: this branch with `git lfs pull` done, uv sync done, git push access, HF_TOKEN in .env (meta-llama + SycoScope),
# and GEN_BATCH_SIZE / CACHE_BATCH_SIZE set for the GPU (batching changes greedy numerics slightly, so no default).
set -euo pipefail
cd "$(dirname "$0")/.."
set -a; . ./.env; set +a
: "${GEN_BATCH_SIZE:?set GEN_BATCH_SIZE}" "${CACHE_BATCH_SIZE:?set CACHE_BATCH_SIZE}"
MODEL=meta-llama/Llama-3.1-8B-Instruct
GEN=generations/Llama-3.1-8B-Instruct/responses
AYS_IN=data/evaluations/are_you_sure/are_you_sure.jsonl
AYS_MC=$GEN/are_you_sure_aqua_mc_math_mc_cot_mmlu_mc_cot_truthful_qa_mc_responses.jsonl
AYS_FF=$GEN/are_you_sure_trivia_qa_truthful_qa_responses.jsonl
CACHE=activations/cache
POSITIONS="prompt_last prompt_mean response_mean response_first5 response_first10 response_frac10 response_frac25 response_frac50 response_frac75"
mkdir -p $CACHE

.venv/bin/python -m pytest tests/test_cache_activations.py tests/test_extraction.py tests/test_judging.py -q

# Are You Sure: greedy, no system prompt; caps 2048 (MC) and 1024 (freeform)
[ -f $AYS_MC ] || .venv/bin/python -m judging.generate_responses --benchmark are_you_sure --model $MODEL --input $AYS_IN \
  --datasets math_mc_cot mmlu_mc_cot truthful_qa_mc aqua_mc --max-new-tokens 2048 --temperature 0 --top-p 1 --seed 0 \
  --batch-size $GEN_BATCH_SIZE 2>&1 | tee activations/generate_ays_mc.log
[ -f $AYS_FF ] || .venv/bin/python -m judging.generate_responses --benchmark are_you_sure --model $MODEL --input $AYS_IN \
  --datasets trivia_qa truthful_qa --max-new-tokens 1024 --temperature 0 --top-p 1 --seed 0 \
  --batch-size $GEN_BATCH_SIZE 2>&1 | tee activations/generate_ays_freeform.log
for f in $AYS_MC $AYS_FF; do git add $f $GEN/meta/$(basename $f).meta.json; done
git diff --cached --quiet || git commit -m "Are You Sure: Llama-3.1-8B responses regenerated (greedy, caps 2048 MC / 1024 freeform)"
git push origin HEAD

.venv/bin/python -m interp.check_cache_parity --model $MODEL --input $GEN/ss_responses.jsonl --n-rows 64 --max-length 8192 \
  2>&1 | tee activations/parity_ss.log
.venv/bin/python -m interp.check_cache_parity --model $MODEL --input $AYS_MC --n-rows 64 --max-length 8192 \
  2>&1 | tee activations/parity_ays_mc.log
.venv/bin/python -m interp.cache_activations --model $MODEL --positions $POSITIONS --cache-dir $CACHE \
  --max-length 8192 --batch-size $CACHE_BATCH_SIZE \
  --inputs $GEN/aita_yta_responses.jsonl $GEN/oeq_responses.jsonl $GEN/ss_responses.jsonl \
           $GEN/aita_nta_og_responses.jsonl $GEN/aita_nta_flip_responses.jsonl $AYS_MC $AYS_FF 2>&1 | tee activations/cache.log
.venv/bin/hf upload SycoScope/activation-cache $CACHE . --repo-type dataset --exclude "*.tmp" \
  --commit-message "Llama-3.1-8B ELEPHANT and Are You Sure activations, $(git rev-parse --short HEAD)"

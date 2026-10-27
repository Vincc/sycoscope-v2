#!/bin/bash
# Sample Llama-3.1-8B responses to ELEPHANT and Are You Sure through OpenRouter (CoreWeave, bf16; T 0.6, top-p 0.9,
# top-k off, run seed 0), then commit and push them. Existing outputs are skipped and interrupted runs resume from
# their .partial file. Needs OPENROUTER_API_KEY and HF_TOKEN (gated tokenizer) in .env.
set -euo pipefail
cd "$(dirname "$0")/.."
set -a; . ./.env; set +a
PY=.venv/bin/python; [ -x $PY ] || PY=.venv/Scripts/python.exe
GEN=generations/Llama-3.1-8B-Instruct/t0.6_p0.9/responses  # the folder generate_responses picks for these settings
COMMON="--model meta-llama/Llama-3.1-8B-Instruct --openrouter-model meta-llama/llama-3.1-8b-instruct --provider CoreWeave \
  --quantization bf16 --temperature 0.6 --top-p 0.9 --top-k 0 --seed 0 --max-workers 32 --env-file .env"
AYS_IN=data/evaluations/are_you_sure/are_you_sure.jsonl
AYS_MC=$GEN/are_you_sure_aqua_mc_math_mc_cot_mmlu_mc_cot_truthful_qa_mc_s0_responses.jsonl
AYS_FF=$GEN/are_you_sure_trivia_qa_truthful_qa_s0_responses.jsonl

$PY -m pytest tests/test_generate_responses.py -q

# ELEPHANT: cap 1024 everywhere; Are You Sure: caps 2048 (MC) and 1024 (freeform)
OUTS=""
for E in AITA-YTA OEQ SS AITA-NTA-OG AITA-NTA-FLIP; do
  OUT=$GEN/${E}_s0_responses.jsonl
  [ -f $OUT ] || $PY -m judging.generate_responses --benchmark elephant --input data/evaluations/elephant/$E.csv \
    --max-new-tokens 1024 $COMMON
  OUTS="$OUTS $OUT"
done
[ -f $AYS_MC ] || $PY -m judging.generate_responses --benchmark are_you_sure --input $AYS_IN \
  --datasets math_mc_cot mmlu_mc_cot truthful_qa_mc aqua_mc --max-new-tokens 2048 $COMMON
[ -f $AYS_FF ] || $PY -m judging.generate_responses --benchmark are_you_sure --input $AYS_IN \
  --datasets trivia_qa truthful_qa --max-new-tokens 1024 $COMMON
OUTS="$OUTS $AYS_MC $AYS_FF"

FILES=""
for f in $OUTS; do FILES="$FILES $f $(dirname $f)/meta/$(basename $f).meta.json"; done
git add $FILES
git diff --cached --quiet -- $FILES || \
  git commit -m "Llama-3.1-8B ELEPHANT and Are You Sure responses via OpenRouter CoreWeave bf16, T 0.6 top-p 0.9 seed 0" -- $FILES
git push origin HEAD

#!/bin/bash
# GPU box, Llama-3.1-8B: one sycophancy fine-tuning run and its evaluations. A rerun skips finished steps.
#   reproduction/persona_vector/run_finetune_vast.sh RUN SPLIT [PREVENT_COEF]   e.g. syc-mis2-prev1.0 misaligned_2 1.0
# SPLIT is normal | misaligned_1 | misaligned_2 of upstream dataset/sycophancy. With BASE=1 the base model's
# eval-question steering cells and projections are also produced. Needs uv sync done and HF_TOKEN in .env.
set -euo pipefail
cd "$(dirname "$0")/../.."
set -a; . ./.env; set +a
RUN=$1; SPLIT=$2; COEF=${3:-}
P=reproduction/persona_vector
V=$P/vectors/sycophantic_float32.npz
S=generations/Llama-3.1-8B-Instruct/t0.6_p0.9/steer_persona_sycophantic
D=$P/data/finetune
A=$P/adapters/$RUN
UPSTREAM=b8e0f044fe2410a6fad579f38324f03f13b4e917
declare -A SHA=([normal]=6df3de0de5c73a73d2f3bc6c3068a1d6eaf1cd97961333b8f0c86e2c5366b399
                [misaligned_2]=87e11c1fd6da6d6672849f7d0c2340804d5f14f6f0755a5fd1b7a489b7cdbb25)
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

if [ ! -f $D/$SPLIT.jsonl ]; then
  rm -rf /tmp/pv && git clone -q https://github.com/safety-research/persona_vectors.git /tmp/pv && git -C /tmp/pv checkout -q $UPSTREAM
  mkdir -p $D && unzip -q -j -o /tmp/pv/dataset.zip "dataset/sycophancy/$SPLIT.jsonl" -d $D
fi
echo "${SHA[$SPLIT]}  $D/$SPLIT.jsonl" | sha256sum -c -

.venv/bin/python -m pytest tests/test_finetune.py tests/test_steer_benchmarks.py -q
STEER=()
[ -n "$COEF" ] && STEER=(--steer-vector $V --steer-layer 16 --steer-coef $COEF)
[ -f $A/adapter_model.safetensors ] || .venv/bin/python -m reproduction.persona_vector.finetune --train-file $D/$SPLIT.jsonl --out-dir $A "${STEER[@]}"

# The paper's eval questions: unsteered, and steered along the original vector (the coherence check).
CELLS="0:0 16:-2 16:-1 16:1 16:2 16:2.5"
.venv/bin/python -m reproduction.persona_vector.generate --mode steer --prompts $P/data/prompts/steer_eval.jsonl \
  --out-dir $P/data/rollouts/ft/$RUN --vector $V --cells $CELLS --batch-size 100 --seed 0 --adapter $A
mkdir -p $P/data/monitor/ft
[ -f $P/data/monitor/ft/$RUN.jsonl ] || .venv/bin/python -m reproduction.persona_vector.monitor_project \
  --rollouts $P/data/rollouts/steer/L00_c0.0.jsonl --vectors $V --out $P/data/monitor/ft/$RUN.jsonl --adapter $A
# Benchmarks, unsteered.
GEN="--vector $V --layer 16 --coefs 0 --out-dir $S/responses --seed 0 --temperature 0.6 --top-p 0.9 --top-k 0 --max-new-tokens 1024 --adapter $A"
.venv/bin/python -m reproduction.persona_vector.steer_benchmarks --items $S/items/are_you_sure_trivia_qa_n400.jsonl $GEN --batch-size 150
.venv/bin/python -m reproduction.persona_vector.steer_benchmarks --items $S/items/OEQ_n100.jsonl $GEN --batch-size 50

if [ "${BASE:-0}" = 1 ]; then
  .venv/bin/python -m reproduction.persona_vector.generate --mode steer --prompts $P/data/prompts/steer_eval.jsonl \
    --out-dir $P/data/rollouts/steer --vector $V --cells 16:-2 16:-1 16:1 16:2 16:2.5 --batch-size 100 --seed 0
  [ -f $P/data/monitor/ft/base.jsonl ] || .venv/bin/python -m reproduction.persona_vector.monitor_project \
    --rollouts $P/data/rollouts/steer/L00_c0.0.jsonl --vectors $V --out $P/data/monitor/ft/base.jsonl
fi
echo "DONE $RUN"

# SycoScope v2

Linear probes for sycophancy sub-types. See `docs/SPEC.md` for the research spec and `docs/PLAN.md` for the build plan.

## Setup

```
uv sync
uv run pytest            # CPU only; needs the cached meta-llama/Meta-Llama-3-8B-Instruct tokenizer
```

Every stage writes a sibling `<output>.meta.json` with input SHA-256s, git commit (and whether the tree was dirty), argv and row counts. Stages refuse to run before the first commit.

## Step 1 pipeline

The old repo is cloned at `../SycoScope-old`. `G` below is `tool_calling/tasks/sycophancy/results/generations`.

CPU, from files:

```
# Inputs: the 14 spec pairs and the 5,000 Perez prompts
uv run python scripts/import_old_inputs.py --old-repo ../SycoScope-old --out-dir data

# Old Llama-3 contrastive generations (run `main`, first 200 prompts)
uv run python scripts/import_old_generations.py --old-repo ../SycoScope-old \
  --old-run ../SycoScope-old/prompt_probes/results/main --pairs data/prompt_pairs.json \
  --prompts data/perez_user_prompts_5k.jsonl --model meta-llama/Meta-Llama-3-8B-Instruct \
  --out-dir data/generations/llama3_main

# Prompt-level split; seed 0 must reproduce the old main/prompt_split.json
uv run python scripts/make_split.py --rows data/generations/llama3_main/*.jsonl --group-field prompt_id \
  --test-frac 0.2 --seed 0 --out data/splits/llama3_main_seed0.json
uv run python scripts/check_parity.py split --new data/splits/llama3_main_seed0.json \
  --old ../SycoScope-old/prompt_probes/results/main/prompt_split.json

# Judged benchmarks -> generations/<generator model>/judging/<name>_judged.jsonl
# ELEPHANT: Meta-Llama-3-8B-Instruct (checked row by row against the raw SAE-branch generations;
# the old results README's "Llama-3.1" is wrong). Cap 512 inferred from the response-length pile-up.
for d in OEQ SS AITA-YTA; do
  uv run python scripts/import_elephant.py --old-repo ../SycoScope-old \
    --judged-ref origin/building-agent --judged-path $G/${d}_social_sycophancy_judged.jsonl \
    --raw-ref origin/SAE --raw-path SAE/results/$d.jsonl --dataset $d --max-new-tokens 512
done
# Are You Sure: Llama-3.1-8B-Instruct only; no Llama-3 generations exist in the old repo
uv run python scripts/import_are_you_sure.py --old-repo ../SycoScope-old --ref origin/building-agent \
  --name ays_freeform --kind freeform --source $G/are_you_sure_freeform/checkpoint.jsonl 150 \
  --generator-model meta-llama/Llama-3.1-8B-Instruct
uv run python scripts/import_are_you_sure.py --old-repo ../SycoScope-old --ref origin/building-agent \
  --name ays_mc --kind mc --source $G/are_you_sure_mc_extra/checkpoint.jsonl 500 \
  --source $G/mmlu_are_you_sure_full/checkpoint.jsonl 300 \
  --generator-model meta-llama/Llama-3.1-8B-Instruct
```

GPU:

```
uv run python scripts/extract_activations.py --model meta-llama/Meta-Llama-3-8B-Instruct \
  --rows data/generations/llama3_main/*.jsonl data/benchmarks/*.jsonl \
  --out-dir activations/llama3 --layers 8 16 24 --max-length 4096 --batch-size 8
```

Extractor parity (GPU). The old extractor writes into its own results dir, so run it in a separate clone:

```
git clone https://github.com/oscaryas/SycoScope ../old-parity && cd ../old-parity && git checkout 4132611 && uv sync
mkdir -p prompt_probes/results/parity/generations
cp prompt_probes/results/main/generations/pt_explicit.jsonl prompt_probes/results/parity/generations/
uv run python probing/probe/get_activations.py --run-name parity --model meta-llama/Meta-Llama-3-8B-Instruct \
  --cells pt_explicit --layers 8 16 24 --batch-size 8
cd ../sycoscope-v2
uv run python scripts/check_parity.py activations \
  --new activations/llama3/pt_explicit.npz --new-index activations/llama3/pt_explicit.index.jsonl \
  --old ../old-parity/prompt_probes/results/parity/activations/pt_explicit.npz \
  --old-index ../old-parity/prompt_probes/results/parity/activations/pt_explicit_index.jsonl --atol 1e-4
```

CPU, from activations:

```
uv run python scripts/train_contrastive.py --pairs data/prompt_pairs.json --slugs all \
  --generations-dir data/generations/llama3_main --activations-dir activations/llama3 \
  --split data/splits/llama3_main_seed0.json --positions last_prompt first5 response --layers 8 16 24 \
  --methods logistic dim --C 1.0 --max-iter 2000 --out-dir probes/contrastive/llama3_main_seed0
uv run python scripts/check_parity.py probes --new-dir probes/contrastive/llama3_main_seed0 \
  --old-summary ../SycoScope-old/prompt_probes/results/main/summary.json --method logistic --atol 1e-6

uv run python scripts/train_skyline.py --rows data/benchmarks/elephant_oeq.jsonl --label validation \
  --activations-dir activations/llama3 --split data/splits/elephant_oeq_seed0.json \
  --positions last_prompt first5 response --layers 8 16 24 --methods logistic dim \
  --C 1.0 --max-iter 2000 --out-dir probes/skyline/llama3_seed0
```

`generate_contrastive.py` produces new contrastive generations in the same format as `import_old_generations.py`.

## Formats

**User prompts** (`data/perez_user_prompts_5k.jsonl`)

| Field | Meaning |
| --- | --- |
| `id` | Prompt id (old `prompt_id`) |
| `text` | User turn |
| `source` | Perez subset (`nlp_survey`, `philpapers`, `political_typology`) |

**Prompt pairs** (`data/prompt_pairs.json`): list of `slug`, `cell`, `type` (`baseline`/`taxonomy`/`control`), `sycophantic`, `non_sycophantic`. Label 1 is the `sycophantic` slot; for control pairs that slot is the legitimate behaviour (for calibrated hedging, the appropriate one).

**Contrastive generations** (`data/generations/<run>/<slug>.jsonl`)

| Field | Meaning |
| --- | --- |
| `id` | `<slug>__<pos\|neg>__<prompt_id>` |
| `pair` | Pair slug |
| `polarity` | `sycophantic` or `non_sycophantic` |
| `label` | 1 for `sycophantic`, 0 otherwise |
| `prompt_id` | Split group: both polarities stay on one side |
| `messages` | `[system, user]` chat messages |
| `response` | Model response |
| `n_response_tokens` | Response length in tokens |
| `degenerate` | `null`, `empty`, `too_short`, `truncated`, `repetitive`, `refusal`, `unclosed_think`. Training drops `empty` and `repetitive` with the pair partner |
| `model` | Generating model |

**Benchmark rows** (`data/benchmarks/<benchmark>.jsonl`)

| Field | Meaning |
| --- | --- |
| `id` | Row id |
| `group_id` | Split group |
| `benchmark` | `ays_freeform`, `ays_mc`, `elephant_oeq` |
| `source` | Sub-dataset (e.g. `trivia_qa`, `mmlu_mc_cot`, `OEQ`) |
| `messages` | Chat messages before the response |
| `response` | The response being labeled |
| `labels` | `{name: 0, 1 or null}`; null is unlabeled. AYS: `caved`. ELEPHANT: `validation`, `indirectness`, `framing` |
| `generator_model` | Model that generated `response`; null if the source does not record it |

**Activations** (`<out-dir>/<stem>.npz`, `.index.jsonl`, `.skips.jsonl`): arrays keyed `<position>_L<layer>`, float32 `(n, hidden)`; block L is read from `hidden_states[L + 1]`. Index row i gives `id`, `n_tokens`, `prompt_len`, `n_response_tokens`, `n_first5_tokens`, `first5_text` for array row i. Skips give `id` and `reason` (`too_long`, `empty_response_span`).

**Splits** (`data/splits/*.json`): `test_frac`, `seed`, `group_field`, `train`, `test` (sorted group ids).

**Probes** (`<out-dir>/<target>.probes.npz`): arrays keyed `<method>__<position>_L<layer>__<field>`. Logistic: `mean`, `scale`, `coef`, `intercept` (score = `((x - mean) / scale) @ coef + intercept`) and `direction` (unit, activation space). DIM: `direction` (unit), `threshold` (score = `x @ direction - threshold`). Positive scores point toward label 1.

**Probe metrics** (`<out-dir>/<target>.metrics.jsonl`): one row per probe with `source` (`contrastive`/`skyline`), `target`, `method`, `position`, `layer`, `split_seed`, `model`, `n_train`, `n_test`, `n_test_pos`, `n_test_neg`, `holdout_auroc` (null when the test split has one class), `holdout_accuracy`; contrastive rows add `pair_type`, `holdout_paired_win_rate`, `n_test_pairs`; skyline rows add `generator_models`.

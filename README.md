# SycoScope v2

Linear probes for sycophancy sub-types. See `docs/SPEC.md` for the research spec and `docs/PLAN.md` for the build plan.

## Setup

```
uv sync
uv run pytest            # CPU only; needs the cached meta-llama/Meta-Llama-3-8B-Instruct tokenizer
```

Every stage writes a sibling `<output>.meta.json` with input SHA-256s, git commit (and whether the tree was dirty), argv and row counts. Stages refuse to run before the first commit.

## Pipeline

Run everything from the repo root. `M` is the model's short name (e.g. `Meta-Llama-3-8B-Instruct`).

```
# Contrastive training responses (GPU): one row per pair x polarity x Perez prompt
uv run python -m probe_training.generate_response --model meta-llama/Meta-Llama-3-8B-Instruct \
  --input data/training/model_written_evals/perez_user_prompts.jsonl \
  --system-prompts data/training/sycophancy_probe_prompt_pairs.json \
  --max-new-tokens 1024 --temperature 0.6 --top-p 0.9 --seed 0 --batch-size 16

# Eval responses (GPU), then judge them (API)
uv run python -m judging.generate_responses --benchmark elephant --model meta-llama/Meta-Llama-3-8B-Instruct \
  --input data/evaluations/elephant/OEQ.csv --prompt-column prompt \
  --max-new-tokens 1024 --temperature 0.6 --top-p 0.9 --seed 0 --batch-size 16
uv run python -m judging.judge_responses --input generations/$M/judging/OEQ_responses.jsonl \
  --judge-model <model> --max-workers 16

# Activations (GPU), for any *_responses.jsonl or *_judged.jsonl under generations/<M>/<training|evaluation|judging>/
uv run python -m probe_training.get_activations --model meta-llama/Meta-Llama-3-8B-Instruct \
  --input generations/$M/judging/OEQ_judged.jsonl --layers 8 16 24 --max-length 4096 --batch-size 8

# Probes (CPU): contrastive (--cells) or skyline on judged labels (--labels)
uv run python -m probe_training.train_probes \
  --activations generations/$M/training/old_main_contrastive_responses_activations.npz \
  --sweep main_C1 --cells all --methods logistic dim --positions last_prompt first5 response \
  --layers 8 16 24 --C 1 --max-iter 2000 --test-frac 0.2 --seed 0

# Generalization (CPU): score a sweep's probes on a judged eval
uv run python -m probe_training.evaluate_probes --sweep-dir probes/$M/main_C1 \
  --activations generations/$M/judging/OEQ_judged_activations.npz --labels all
```

## Data ported from the old repo

The old repo is cloned at `../SycoScope-old` (git refs) and `../SycoScope` (local activations, never committed).
`G` below is `tool_calling/tasks/sycophancy/results/generations`.

```
# Old `main` run: 14 pairs x 2 polarities x 200 Perez prompts, plus 200 no-system-prompt rows, with
# layer 8/16/24 activations. Every row's spans are re-checked against the current tokenization.
uv run python scripts/import_old_activations.py --old-run ../SycoScope/prompt_probes/results/main \
  --pairs data/training/sycophancy_probe_prompt_pairs.json --name old_main

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

## Layout

- `generations/<M>/training/<name>_responses.jsonl`: rows with `id`, `messages`, `response`, `truncated`, `model`; contrastive rows add `pair_index`, `cell`, `pair_type`, `polarity`, `label` (1 = the pair's `sycophantic` slot; for control pairs that slot is the legitimate behaviour) and `prompt_id` (split group).
- `generations/<M>/judging/<name>_judged.jsonl`: as above plus `benchmark` (`elephant`/`are_you_sure`) and `labels` (`{name: 0, 1 or null}`, null = unparseable or not applicable). ELEPHANT: `validation`, `indirectness`, `framing` (1 = sycophantic). Are You Sure: `turn1_correct`, `turn2_correct`, `syco`.
- `<input stem>_activations.npz` next to its input: float32 `(n, hidden)` arrays keyed `<position>_L<layer>` (block L read from `hidden_states[L + 1]`), per-row `id`, `n_tokens`, `prompt_len`, `n_response_tokens`, `first5_text`, and the row's pair fields or `labels__<name>` (int8, -1 = null) and `group`. Skipped rows are listed in the `.meta.json`.
- `probes/<M>/<sweep>/`: `split.json` (train/test groups), `<unit>.probes.npz` (arrays keyed `<probe_id>__<field>`), `manifest.jsonl` (one row per probe: parameters and train/test metrics), `eval/<name>.jsonl` (one row per probe x eval label).

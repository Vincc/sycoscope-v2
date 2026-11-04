# Cheng et al. (Verbalized Assumptions) replication notes

Paper: Verbalizing LLMs' Assumptions to Explain and Control Sycophancy (arXiv 2604.03058v3, COLM 2026).
PDF: `../verbalizing_llm_assumptions.pdf`. Upstream code `upstream/` (github.com/myracheng/verbalizedassumptions,
single commit 0279c9a, 2026-04-02). Data from OSF `osf.io/bzepy` (view-only link in the upstream README),
downloaded 2026-10-26 to `data/osf/` with `data/osf/SHA256SUMS`.

## What the paper does

1. Verbalized Assumptions: prompt an LLM for its "mental model" of the user. Open-ended (top-3 named models
   with probabilities) and structured (scores in [0,1] on nine dimensions, Table 3). Nine labelling LLMs.
2. Assumption probes (Section 3): for each (dimension d, labelling model m, probe model), ridge regression
   from mean-pooled layer-l hidden states of the user prompt to m's score. 80/20 train/test over six datasets
   (OEQ, AITA-YTA, IR "resps", factual, Cancer-Myth, WildChat); inside train, 20% validation picks the layer by
   R^2. Evaluation: per-source AUC of high (>0.7) vs low (<0.3) scores, macro-averaged (Table A7).
3. Steering: h + alpha * v at the chosen layer, alpha in {-4,-2,-1,-0.5,0,0.5,1,2,4}; responses judged with a
   1-5 version of the ELEPHANT social-sycophancy judges (validation, indirectness, framing; GPT-4o judge).
   Spearman rho between alpha and the judge score (Fig. 3, Table A9). S+ dimensions should raise validation,
   S- lower it; objectivity seeking behaves opposite to its hypothesis. Reward (ArmoRM) stable for |alpha|<=4.
4. Human-AI expectation gap user study (not replicated here).

## Upstream code

- `verbalized_assumptions/get_assumptions.py`: elicitation prompts (`4dims`, `supporttypes`, `openended`).
  Local models sampled at temperature 0.7 / top_p 0.9, 4-bit.
- `assumption_probes/train_assumption_probe.py`: Ridge(alpha=10) on `hidden_states[layer]` mean-pooled over the
  raw user text tokenized **without a chat template** (plain `tokenizer(text)`, max_length 512). An 80/20 split
  with `train_test_split(random_state=seed)` on the train CSV; the layer with the best held-out R^2 is kept and
  its weight vector saved, L2-normalised. The sample script uses seed 42, pooling mean, layers 20..70 for 70B.
- `assumption_probes/evaluate_auc_by_source.py`: score = X @ direction; binarises by the 0.3 / 0.7 *quantiles*
  of each source's labels (the paper text says fixed thresholds >0.7 / <0.3).
- `assumption_probes/sample_and_generate_steered.py`: forward hook on `model.model.layers[layer]` output,
  `h - alpha * direction` (sign is opposite to the paper's eq. 3, so upstream alpha = -paper alpha), applied to
  every position including generated tokens. System prompt "You are a helpful assistant. Reply to the user.",
  sampling temperature 0.7 / top_p 0.9, 400 new tokens, 4-bit.
- `elephant_scorer_5pointscale.py`: the three 1-5 judge prompts (GPT-4o via Azure), one call per dimension.
- `sycophancy_probes/`: same pipeline on binary sycophancy labels (baseline in Section 3.1).

Not in the repo: data assembly, the 70B labels' generation configs, ArmoRM reward scoring, the counterfactual
prompting, plotting.

## Data (`data/osf/`)

`structured_assumptions/{train,test}_{gemini,gpt4o,llama70b,llama8b}_{4dims,supporttypes}.csv`: one row per
user prompt with `user_text`, `source_file`, the four or five `<dim>_score` columns (+ explanations), the raw
`model_output`, and hundreds of unrelated columns merged in from the source datasets. Rows per file are
9.8-10k train / 2.4-2.5k test (gpt4o test_4dims only 1,435). Within one file train and test prompts are disjoint;
across labellers the test prompts of one file do appear in another's train set, which does not matter for a
single (d, m) probe. Scores are missing (blank) for 0-8% of rows (the labeller failed to produce JSON);
`gemini` information_guidance has a few -1.0 values. `source_file` names map to the six datasets by substring
(factual / OEQ / AITA / resps = IR / cancer / wildchat); a seventh source `valobj` (71 train / 25 test) is the
expectation-gap prompt set and is not one of the paper's six probe datasets.

`sycophancy_data/{train,test}_probe_data_syco.csv`: binary ELEPHANT labels per response model for the
sycophancy-probe baseline. `all_openended_assumptions.csv`: the open-ended outputs (2,165 prompts x 6 models).

## Plan

Probe model: Llama-3.1-8B-Instruct (bf16, not 4-bit). Labels: all four labelling models x nine dimensions
(36 probes; the paper's Table A7 8B rows). 70B as probe model is deferred.

Stage A `src/prepare_data.py` (CPU): CSVs -> `data/prepared/{labeller}_{split}.jsonl` with id, user_text,
dataset, scores; `data/prepared/prompts.jsonl` = union of unique prompts for extraction. Exclusions counted.
Stage B `src/extract_reps.py` (GPU): mean-pooled hidden states at a layer sweep -> `results/reps_*.npz`.
Stage C `src/train_probes.py` (CPU): ridge per (d, m), layer by validation R^2, AUC per dataset on test
-> `results/probes/` + Table A7 analogue.
Stage D (after C reproduces): steering generations + 1-5 ELEPHANT judge -> Table A9 analogue.

Decisions not fixed by the paper or code, ours:
- Tokenisation for probes follows upstream (raw text, no chat template, truncation at 512 tokens, BOS kept).
- Layer sweep for 8B: hidden_states indices 4, 8, 12, 16, 20, 24, 28 (index L = output of block L-1; index 0 is
  the embeddings). Upstream's 70B sweep was 20..70 step 10.
- Rows with a missing or out-of-[0,1] score are excluded (upstream drops missing and clips out-of-range).
- Duplicate prompts within a file are kept as separate rows, as upstream does.
- AUC uses the paper's fixed thresholds (score > 0.7 vs < 0.3), per dataset, macro-averaged over the six paper
  datasets; valobj rows are excluded. Upstream's quantile variant is not used.
- Inner validation split: 20% of train with seed 42 (upstream sample script).

## Run log

### 2026-10-26: Stage A-C, Llama-3.1-8B-Instruct as probe model

Machine: vast.ai A100 SXM4 40GB ($0.80/h), torch 2.5.1+cu121, transformers 4.57.6, sklearn 1.9.1; model weights
`unsloth/Llama-3.1-8B-Instruct` (ungated mirror, bf16). Total GPU cost $0.45. Files were staged in a non-ignored
`va_sync_tmp/` at the project root for `sync()` (the replications directory is gitignored and `include` does not
override that); a later `sync()` wiped `/workspace/activations`, so extraction was run twice (identical results).
Do not sync after producing outputs; download first.

Stage A: see `data/prepared/meta/`. Exclusions per file are 3-786 rows (missing scores; worst are gpt4o and
llama8b 4dims, 7-8%) plus 15-52 test rows per supporttypes file whose prompt also occurs in that file's train
set (duplicated short prompts such as "Are you sure?", "hello"). 12,234 unique prompts across all files.

Stage B: `results/reps_llama31_8b_L4-28.npz` (1.4 GB, not committed) + `.meta.json`; 1,104 of 12,234 prompts
truncated at 512 tokens (long AITA posts and WildChat prompts).

Stage C: `results/probes_a10/` (fixed thresholds >0.7 / <0.3) and `results/probes_a10_quantile/` (per-dataset
0.3 / 0.7 label quantiles, as upstream code); same 36 probes in both, only the AUC definition differs.

Selected layers (hidden_states index): 12 or 16 for 33 of 36 probes; llama70b belonging 20; llama8b user
information advantage 4 and objectivity 8 (these two probes are near chance, val R^2 0.03 / 0.10). Matches the
paper's "middle layers peak".

Test R^2, mean over the nine dimensions (paper: mean 0.50 for 8B; S+ > S-; emotional support highest, 0.77):

| labeller | mean | S+ | S- | emotional support |
|---|---|---|---|---|
| gemini | 0.60 | 0.66 | 0.47 | 0.93 |
| gpt4o | 0.57 | 0.60 | 0.52 | 0.91 |
| llama70b | 0.67 | 0.71 | 0.60 | 0.90 |
| llama8b | 0.39 | 0.39 | 0.38 | 0.79 |

Macro AUC over the six datasets, mean over dimensions. Paper Table A7, 8B probe model (digits as extracted
from the PDF; the table layout is garbled, so only the ordering is reliable): Llama-70B labels best (0.81),
GPT-4o ~0.78, Gemini ~0.76, Llama-8B ~0.72.

| labeller | fixed thresholds | undefined cells | quantile (upstream) |
|---|---|---|---|
| gemini | 0.889 | 4 of 9 dims | 0.733 |
| gpt4o | 0.903 | 7 of 9 | 0.787 |
| llama70b | 0.955 | 6 of 9 | 0.766 |
| llama8b | 0.855 | 3 of 9 | 0.723 |

With the paper's fixed thresholds many (dimension, dataset) cells have no positives or no negatives (e.g. no
factual prompt scores > 0.7 on emotional support), so the macro average is undefined for 20 of 36 probes; the
numbers above average only the defined dimensions and are inflated because the fixed thresholds compare the
extremes of the label distribution. The quantile version is defined everywhere and lands in the paper's range
(0.72-0.79 vs 0.72-0.81), with Llama-8B labels worst as in the paper; the labeller ordering among the other
three differs (gpt4o > llama70b > gemini here). Per-dimension: emotional support, belonging and social
companionship are the best-probed S+ dimensions; tangible support (0.63-0.70) and llama8b-labelled user
information advantage (0.55) are the worst, consistent with Fig. A10's "user rightness and objectivity lowest".

Not yet run: the 70B probe model, the sycophancy-probe baseline, counterfactual prompting, ArmoRM reward.

### 2026-10-26: Stage D1, steering generations (Llama-3.1-8B-Instruct, 36 probes)

Goal set by the user: replicate as closely as possible, same judge model (GPT-4o). Settings follow upstream
`sample_usage.sh` / `sample_and_generate_steered.py`: system prompt "You are a helpful assistant. Reply to the
user.", sampling temperature 0.7, top_p 0.9, 400 new tokens, alphas -4,-2,-1,-0.5,0,0.5,1,2,4, hook on
`model.model.layers[L]` with L the probe's hidden_states index (upstream convention, one block after the
representation the probe was fit on), applied at every position. Differences: bf16 instead of 4-bit, and the
paper's sign (h + alpha v; upstream code subtracts). Direction = unit-normalised ridge weights from
`results/probes_a10/probes/` (same probes as the quantile run).

Prompts: `data/steer_prompts_n50_seed0.jsonl`, 50 social prompts (24 OEQ, 16 AITA, 10 IR) sampled with seed 0
from the union of all labellers' test files (3,404 social test prompts). Every one of them is in some other
labeller's train file, since each labeller has its own split; only 2 social prompts are in no train file at
all. Upstream's `--n` for generation is not given in the repo.

Machine: vast.ai A100 SXM4 40GB ($0.75/h). Probe `.npz` files are filtered out of `sync()` by the root
`.gitignore` (`*.npz`), so they were shipped as `va_sync_tmp/probes.tar`. Batch 50 ran out of memory on the
long AITA prompts; batch 16 used. Smoke test (4 prompts, emotional-support probe, alpha -4/0/+4) showed the
expected shift: +4 opens with "I understand your concern", -4 goes straight to numbered suggestions.

Outcome: `results/steer_n50_seed0.jsonl` (16,200 rows = 36 probes x 9 alphas x 50 prompts, 51 MB, sha256
f4283e10...b2f2) + `.meta.json`. Runtime 8.5 h at batch 16 (the kernel websocket dropped after ~7 h; the
process kept writing to disk and was collected from a fresh kernel). GPU cost $6.88. 7,887 of 16,200 responses
(49%) stopped before the 400-token cap, with no trend over alpha (46-53%); none are empty.

Judge (Stage D2): `src/elephant5_score.py --model openrouter/openai/gpt-4o`, upstream prompts and settings
(no system prompt, no temperature sent, max_tokens 1000). The OpenRouter key was renewed on 2026-10-26 after
first returning "API key expired". 16,200 responses x 3 dimensions = 48,600 calls. The script appends rows as
they finish and resumes from an existing output file.

### 2026-10-27: Stage D2, GPT-4o 1-5 ELEPHANT judge and Table A9 comparison

`results/steer_n50_seed0_elephant5_gpt4o.jsonl` (16,200 rows, each with validation / indirectness / framing
scores and the raw judge replies) and `_summary.json` (per probe x dimension: mean score by alpha, Spearman
rho, p). Judge `openrouter/openai/gpt-4o`, upstream prompts verbatim, no system prompt, provider-default
temperature, max_tokens 1000. 48,600 calls, 0 unparsed (every reply was a bare digit). The first run died at
13,691 rows on an OpenRouter 504 (upstream rate limit); 504 was added to the retried codes in `api.py` and the
run resumed. Comparison: `src/compare_table_a9.py` against `data/paper_table_a9_8b.json` (Table A9, "Steering
8B" columns, transcribed from the PDF).

Spearman rho between alpha and mean judge score, averaged over probes (ours vs paper Table A9, 8B):

| judge dim | S+ probes (24) ours | paper | S- probes (12) ours | paper |
|---|---|---|---|---|
| validation | +0.70 | +0.82 | -0.28 | -0.42 |
| indirectness | +0.20 | +0.51 | +0.31 | +0.04 |
| framing | +0.07 | +0.11 | +0.31 | -0.01 |

Per cell (108 cells, all defined): where both |rho| >= 0.5 (49 cells) the sign agrees in 48; correlation of the
two rho vectors 0.63. Ours significant at p < .05: validation 20/36, indirectness 11/36, framing 6/36.

What replicates: the headline. Steering with S+ probes raises validation (emotional support and validation
seeking rho +0.97 to +1.00 for all four labellers; belonging, social companionship, user rightness positive),
information guidance lowers it (-0.90 to -1.00, all labellers), and objectivity seeking does not lower it
(-0.02 / +0.31 / +0.30 / -0.52) while strongly raising indirectness (+0.75 to +0.98), the paper's "opposite to
hypothesis" finding. Baseline (alpha 0) means 3.14 validation, 3.80 indirectness, 4.31 framing: framing is
near the ceiling for these social prompts, which is why both we and the paper find few framing effects.

What is weaker: indirectness for S+ probes (paper +0.51 mean, ours +0.20; e.g. gemini / gpt4o / llama70b
emotional support +0.96 / +1.00 / +1.00 in the paper vs -0.01 / +0.38 / +0.46 here), and the llama8b-labelled
S- probes (paper -0.89 to -1.00 on tangible support and information advantage, ours near 0). The two near-chance
llama8b probes (user information advantage, layer 4; objectivity, layer 8) steer poorly: at |alpha| = 4 the
information-advantage probe gives short off-task replies ("I am a helpful assistant. I can help you with
that."), so its indirectness curve is U-shaped (1.62 / ~3.9 / 1.92). Possible causes for the gap: 50 prompts
vs the paper's unknown n, bf16 vs 4-bit, our probes differ from theirs (same recipe, different runs), and
judge nondeterminism.

### 2026-10-27: Counterfactual prompting (paper Section 3, Table A5, Fig. A11)

`src/counterfactual_prompts.py --n 300 --seed 0`: 300 prompts sampled from the union of all labellers' test
prompts (6,967; drawn: 122 factual, 86 OEQ, 44 AITA, 24 IR, 12 WildChat, 10 Cancer-Myth, 2 Val-Obj). For each
of the nine dimensions the Table A5 positive and negative instruction is appended to the prompt after a blank
line (the paper says "append text"; the exact joining is ours). `extract_reps.py` on the 5,400 texts (same
settings as the probe reps; 553 truncated at 512 tokens), then `src/counterfactual_score.py`: probe score
X @ w + b at the probe's layer, paired difference positive minus negative, paired t-test.
Output `results/counterfactual_n300_seed0.json`.

31 of 36 probes have a positive mean difference with p < 1e-15 (paper: significant positive for all but two
probes). Magnitudes 0.002-0.039 on the [0,1] score scale, same range as Fig. A11's 8B panel (0-0.04). Largest:
gpt4o validation seeking +0.039, llama70b validation seeking +0.033, gemini objectivity +0.021. Exceptions:
user information advantage is not positive for gemini (-0.001, p = 0.06), llama70b (-0.007) and llama8b
(-0.005); llama8b objectivity (-0.001) and llama8b tangible support (-0.001) are slightly negative. The three
llama8b failures are the probes with near-chance validation R^2 (layers 4 and 8). "Assume the user has more
information than you" may read to the model as an instruction about itself rather than about the user, which
would explain why that dimension fails across labellers.

### 2026-10-27: Sycophancy-probe baseline (paper Section 3.1, Table A8 "NR", Fig. A16)

Data: `src/prepare_syco_data.py` on `data/osf/{train,test}_probe_data_syco.csv` (10,938 / 2,735 rows; sources
factual 5,800, OEQ 2,411, AITA 1,596, "elephant" 655, Cancer-Myth 476 in train). Labels are binary ELEPHANT
judgements of a response model's reply (`<dim>_<Model>`); a row is kept for a response model only when all
three of its labels are present (excluded: gpt4o 1,074 train / 268 test, gemini 570 / 165, llama70b 1,623 / 413,
llama8b 5,968 / 1,542, nearly all missing labels). Text column `prompt` (upstream picks the text column with
fewest NaNs). The 1-4 prompts per model that occur in both splits are dropped from test. Reps: 3,065 of these
prompts were not in the assumption-probe set and were extracted on a second machine (A100, same settings;
`results/reps_syco_missing_L4-28.npz`).

Training `src/train_syco_probes.py --C 1.0 --seed 0` (upstream train_probe_bin.py: LogisticRegression,
class_weight balanced, stratified 80/20 split, best layer by validation F1; sigmoid scores for AUC).
`results/syco_probes_C1/`. Macro AUC over the five sources on the test file (paper Table A8, 8B, NR):

| response model | validation | indirectness | framing | paper validation / indirectness / framing |
|---|---|---|---|---|
| gpt4o | 0.854 | 0.750 | 0.751 | 0.87 / 0.72 / 0.71 |
| gemini | 0.809 | 0.672 | 0.727 | 0.88 / 0.62 / 0.76 |
| llama70b | 0.771 | 0.628 | undef (cancer all one class) | 0.65 / 0.58 / 0.60 |
| llama8b | undef (factual all one class) | 0.698 | undef (elephant all one class) | 0.82 / 0.71 / 0.73 |

Validation is the best-probed dimension for every model, as in the paper. The undefined cells are sources
where every test row has the same label after the all-three-labels filter (e.g. no non-sycophantic factual
response from Llama-8B); the paper's table has a number there, so it presumably used a different filter or
the pooled AUC. Overall (pooled) test AUC is 0.81-0.98. Framing picks a late layer (24-28) for every model.

Steering: the three gpt4o-labelled sycophancy probes (validation and indirectness layer 12, framing layer 28),
same 50 prompts, alphas and sampling as the assumption-probe run, judged with GPT-4o. The paper's claim is that
these steer sycophancy as well as assumption probes do but cut ArmoRM reward by up to 50%; ArmoRM is not run
here, so only the sycophancy side is checked.

Steering result (`results/steer_syco_n50_seed0.jsonl`, 1,350 rows, 48% finished before the cap; judged in
`results/steer_syco_n50_seed0_elephant5_gpt4o.jsonl`, 4,050 GPT-4o calls, 0 unparsed). Spearman rho(alpha, mean
score): validation probe -> validation +1.00 (means 1.58 at alpha -4 to 4.76 at +4, the widest range of any
probe in this replication) and indirectness +0.98; indirectness probe -> indirectness +0.97, validation +0.98;
framing probe -> nothing (+0.22 framing, |rho| <= 0.16 elsewhere; framing is at 4.2-4.4 for every alpha).
Paper Fig. A16 bottom: validation and indirectness probes steer their dimension with significant rho, framing
weak. Matches. The validation and indirectness probes move both of those dimensions together, i.e. they are
not separable at this layer. Machine 2 cost $0.76; whole replication GPU cost $7.64.

### 2026-11-03: Difference-in-means vectors as an alternative to the ridge probes

`src/build_dim_vectors.py` builds two difference-in-means (DiM) directions per (labeller, dimension) at the layer the
matching ridge probe selected: `label` (mean train rep with score > 0.7 minus mean with score < 0.3; smallest group
47 rows, `--min_group 30`) and `cf` (label-free: mean over the 300 counterfactual prompts of rep(positive instruction)
minus rep(negative instruction), one direction per dimension). The mean rep and the top 10 principal components of
the labeller's train reps at that layer are projected out of both (raw label DiM has cos ~ -0.7 with the mean rep,
which mean pooling without a chat template ties to prompt length, and macro AUC 0.56). Output
`results/dim_vectors/{label,cf}/probes/*.npz` (raw w, b = 0, layer) and `results/dim_vectors/dim_results.json`.

Test macro AUC (upstream quantile definition, as `probes_a10_quantile`), mean over the 36 probes: ridge 0.752,
label DiM 0.644 (0.55-0.74), cf DiM 0.523 (0.47-0.59). Per labeller the ordering is the same (gpt4o 0.79 / 0.67 /
0.54, gemini 0.73 / 0.63 / 0.50, llama70b 0.77 / 0.64 / 0.52, llama8b 0.72 / 0.63 / 0.53). Cosine with the ridge
weights: label DiM 0.46 (0.35-0.66), cf DiM 0.03 (-0.04 to 0.09); label and cf DiM 0.03. So the label DiM points
partway along the ridge direction and keeps about half its margin over chance, while the counterfactual-instruction
direction is orthogonal to the probe and does not separate high from low scores, even though the probes themselves
respond to the same instructions (counterfactual prompting above). These vectors were built for the steering
comparison on the Baez prompts in the `direction-compare` worktree; no steering with them is recorded here.

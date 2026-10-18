# Bohacek et al. (CLiF) replication notes

Paper: Detecting and Controlling Sycophancy with Cascading Linear Features (arXiv 2606.26155).
Upstream repo `upstream/` (github.com/maty-bohacek/cascading-feats at c4acedf, 2026-04-19) contains
only a README that says "Code coming soon". Everything below is reconstructed from the paper.

## Method (paper Sections 3-4, Appendix K)

Model: Llama-3.1-8B-Instruct. SAE: `Goodfire/Llama-3.1-8B-Instruct-SAE-l19` (ungated; one `.pth`
plus `config.yaml`), residual stream at layer 19, 65,536 features, expansion 16, mean L0 ~ 91.

Data: Anthropic sycophancy set (Perez et al. 2023), `Anthropic/model-written-evals`,
`sycophancy/sycophancy_on_{nlp_survey,philpapers2020,political_typology_quiz}.jsonl`. The paper
says it used "a subset"; the exact count is not stated.

1. Cascading data generation. Level 0: neutral response with prompt K.1. Levels +1..+3: iteratively
   rewrite the previous response with prompt K.2 (more sycophantic). Levels -1..-3: prompt K.3 (more
   dismissive). Seven responses per input prompt, all from Llama-3.1-8B-Instruct itself.
2. Feature selection. Encode activations of each response with the SAE; per feature j and level l,
   mean activation a_j(l). F+ = features with a_j(0) <= a_j(+1) <= a_j(+2) <= a_j(+3);
   F- analogously along 0, -1, -2, -3. CLiF = F+ union F-.
   Open question: pooling over tokens of the response (paper shows token-level activations in Fig. 2
   and speaks of "the generation's SAE activation vector"; we use the mean over response tokens).
3. Detection: 3-class (level -3 / 0 / +3) logistic regression and SVM on SAE activations restricted
   to CLiF features; baseline = same probes on all 65,536 features. Paper Table 1 reports accuracy.
4. Steering (secondary): clamp F+ features to zero in SAE space, or negative steering with
   alpha_j = -(a_j(target) - a_j(opposite)); evaluated with ELEPHANT (GPT-4o judge) on 50 prompts.

Appendix E: a residual-stream logistic probe gets 0.945-0.973 5-fold accuracy at every probed layer,
with L19 at 0.952.

## Plan

Stage A (GPU): generate 7 levels for N prompts, save JSONL. Stage B (GPU): L19 residual stream ->
SAE features, mean-pooled over response tokens, save npz. Stage C (CPU): monotonic selection,
probes, Table 1 analogue. Steering is deferred until A-C reproduce.

## Run log

Machine: vast.ai A100 PCIE 40GB ($0.60/h), torch 2.5.1+cu121, transformers 4.57.6. Model weights from
`unsloth/Llama-3.1-8B-Instruct` (ungated mirror; no HF token on the machine). Code in `src/`, run from
`/workspace/clif`. Note: remote-kernels `sync()` mirrors deletions and wipes anything under the workdir
that is not in the project; data and outputs under `/workspace/clif` must be re-created after a sync.

Decisions not fixed by the paper:
- Prompts: 210 questions, 70 per Anthropic file, seed 0. The persona + question text is used; the
  "Choices:" block is dropped for nlp_survey / philpapers (the question is self-contained) but the inline
  options are kept for political_typology (the question refers to them). The trailing "Answer:" is dropped.
- Generation: greedy, max 350 new tokens, Llama 3 chat template, each K prompt as a single user turn.
- Activations: chat (user = original prompt, assistant = response); block-19 output encoded by the SAE
  and mean-pooled over response tokens.
- Detection split: 20% of prompts held out (seed 0); monotone selection on training prompts only.

Smoke test (6 prompts, 120 tokens) confirmed the cascade: level +3 is maximally gushing, level -3
dismissive and critical, level 0 polite and balanced.

## Results: detection (2026-10-14)

Files: `results/cascade_n210_seed0.jsonl` (1470 generations), `results/feats_L19_n210_seed0.npz`
(SAE + residual means, 217 MB), `results/probe_results_L19.json`, `results/probe_results_L19_clif_mask.npy`.

Generation stats: mean 275-292 words per level; many responses hit the 350-token cap. SAE mean L0 over
response tokens ~100-110 (Goodfire reports 91 per token).

Feature selection on 168 training prompts: F+ = 6817, F- = 8158, overlap 3092, CLiF = 11883 of 65536.
The overlap is large because a feature whose per-level means are all equal (or nearly) satisfies both
non-decreasing chains; the paper does not say how it handled this.

3-class detection (levels -3 / 0 / +3), 42 held-out prompts = 126 test rows; paper Table 1 "Anthropic"
column in brackets:

| Features                     | LR    | SVM   |
|------------------------------|-------|-------|
| CLiF SAE (11883)             | 0.968 | 0.937 | [96.7 / 98.3]
| all SAE (65536)              | 0.984 | 0.984 | [contrastive baseline 98.3 / 90.0]
| random SAE subset (11883)    | 0.968 | 0.968 |
| residual stream L19 (4096)   | 1.000 | 1.000 | [Appendix E: 0.952 5-fold]

CLiF + LR matches the paper (96.8 vs 96.7). The SVM number is lower than the paper's 98.3. The task is
easy enough that the monotonicity filter does not beat the full feature set or a random subset of the
same size on this split, so the paper's claim that CLiF selection helps detection is not reproduced
here; the three-way separation itself is. The mean (F+ sum - F- sum) score is monotone in level
(-8.3, -6.8, -4.8, -2.2, -1.9, -0.1, +2.7), matching Fig. 3b qualitatively.

Not done: OOD sets (Culture / Non-US policy / Office; need Gemini generation), LLM-judge baselines,
steering (clamping / negative steering) with ELEPHANT scoring (needs a GPT-4o or compatible judge).

## Extension (2026-10-15): OOD sets, judge baselines, steering

API models. The OpenRouter key in `.env` is expired ("API key expired", 401), so the paper's Gemini 2.5
Pro/Flash and GPT-4o calls are replaced by Anthropic models, always passed as `--model` and recorded in
each output: scenario generation claude-sonnet-5-5, judge baselines claude-haiku-5-5 (Flash analogue) and
claude-sonnet-5-5 (Pro analogue), ELEPHANT scorer claude-sonnet-5 (the model this repo's other judging uses).
Sampling at API defaults (the SDK rejects `temperature` for these models).

OOD scenarios (`src/gen_ood_scenarios.py`, `data/ood_scenarios.jsonl`): 20 per domain (Culture, Non-US
Policy, Office Scenarios) with the Appendix L prompt and one real Anthropic example per file; output
requested as a JSON list and checked (persona opening, (A)/(B) options). Cascaded with the same 7-level
procedure (`gen_cascade.py --prompts_file`), features with `sae_features.py` (now also stores `sources`).

Judge baselines (`src/judge_baseline.py`): Appendix M.1 prompt, and M.2 = M.1 + the three example
responses transcribed from the paper (the user-input part of the examples is elided in the PDF). Three
repetitions; accuracy = fraction of held-out (-3 / 0 / +3) rows whose JSON answer maps to the true level;
consistency = fraction of rows with identical labels across the three repetitions.

Steering (`src/build_steer_params.py`, `src/steer_generate.py`, `src/elephant_score.py`): 50 scenarios
drawn from the three Anthropic files (seed 1, 17/17/16), disjoint from the 210 cascade prompts; the
scenario text is the only user turn; greedy, 300 new tokens; hook on the output of block 19 at every
position. Config definitions the paper leaves open, ours:
- Clamp set "All" = F+ (6817); "One-sided" = F+ \ F- (3725); "Level k" = F+ features strictly increasing
  along levels 0..k (3977 / 3963 / 3957). Subtract uses the same sets with alpha_j = -(mean_j(+3) - mean_j(-3)).
- Contrastive baseline: d = mean(+3) - mean(-3) over all SAE features (no monotonicity filter). Addition:
  f' = f - d. Clamp: zero the |F+| features with the largest d.
- Persona Vectors: residual-stream difference of means mean_resid(+3) - mean_resid(-3) at block 19, subtracted once.
- Means are over the 168 training prompts only.
ELEPHANT: validation, indirectness, framing with the ELEPHANT prompts from `judging/judge_responses.py`;
score = mean of the three 0/1 labels (lower = less sycophantic), mean +- std over the 50 scenarios.
Note the random-subset probe now draws from `RandomState(seed + 1)` (the split moved into `split_ids`), so
its number differs from the 2026-10-14 run above by chance only.

## Results: OOD detection, judge baselines, steering (2026-10-15)

Files: `data/ood_scenarios.jsonl`, `results/cascade_ood_n60.jsonl` (420 rows), `results/feats_L19_ood_n60.npz`,
`results/table1_ood.json`, `results/judge/*.json`, `results/steer_n50.jsonl` (800 rows),
`results/steer_n50_elephant.jsonl` + `_summary.json`, `results/run_remote_20261015.log`. Machine: vast A100 PCIE,
same stack as 2026-10-14; the stage-B feature file for the n210 cascade was rebuilt on the machine (identical
mask counts: F+ 6817, one-sided 3725, level1-3 3977/3963/3957).

### Table 1 analogue: 3-class accuracy (%), probes fit on the 168 Anthropic training prompts

Test sets: Anthropic 126 rows, each OOD domain 60 rows (20 scenarios x 3 levels). Feature counts: CLiF all
11883, level1/2/3 22019/14784/10179, random 11883. Level k here = features monotone through level +-k only.

| probe | Anthropic | Non-US | Office | Culture |
|---|---|---|---|---|
| clif_all+LR | 96.8 | 65.0 | 68.3 | 61.7 |
| clif_all+SVM | 93.7 | 61.7 | 71.7 | 75.0 |
| clif_level1+LR | 98.4 | 73.3 | 73.3 | 80.0 |
| clif_level1+SVM | 98.4 | 63.3 | 70.0 | 80.0 |
| clif_level2+LR | 99.2 | 81.7 | 78.3 | 78.3 |
| clif_level2+SVM | 99.2 | 80.0 | 78.3 | 83.3 |
| clif_level3+LR | 97.6 | 76.7 | 76.7 | 73.3 |
| clif_level3+SVM | 96.8 | 78.3 | 76.7 | 80.0 |
| contrastive_all_sae+LR | 98.4 | 70.0 | 78.3 | 78.3 |
| contrastive_all_sae+SVM | 98.4 | 65.0 | 71.7 | 76.7 |
| random_subset+LR | 98.4 | 85.0 | 76.7 | 83.3 |
| random_subset+SVM | 97.6 | 78.3 | 78.3 | 80.0 |
| residual_L19+LR | 100.0 | 100.0 | 100.0 | 100.0 |
| residual_L19+SVM | 100.0 | 100.0 | 98.3 | 100.0 |

LLM-as-a-judge (Appendix M prompts; 3 repetitions, mean accuracy; consistency in-dist / OOD):

| judge | prompt | Anthropic | Non-US | Office | Culture | consistency |
|---|---|---|---|---|---|---|
| claude-haiku-5-5 | default | 81.2 | 80.6 | 91.7 | 92.2 | 0.89 / 0.91 |
| claude-haiku-5-5 | examples | 83.3 | 85.6 | 92.8 | 91.7 | 0.91 / 0.93 |
| claude-sonnet-5-5 | default | 83.6 | 81.1 | 90.6 | 96.1 | 0.98 / 0.98 |
| claude-sonnet-5-5 | examples | 81.7 | 80.6 | 89.4 | 95.0 | 0.95 / 0.98 |

Reading. In distribution the SAE probes stay at 94-99%, as before. Out of distribution every SAE-feature probe
drops to 62-85%, far below the paper's 90-100%, and the drop is specific to level 0: CLiF+LR on the OOD sets
gets 54/60 and 52/60 of the extremes right but only 11/60 of the neutral responses (32 called rejection, 17
sycophancy). The residual-stream probe stays at 98-100% everywhere, so the OOD neutral responses are
separable; it is the SAE mean-feature representation, filtered or not, that does not carry over. The
monotonicity filter does not help OOD either: random subsets of the same size and the unfiltered SAE probe
score the same or better. So the paper's OOD claim (CLiF+SVM >= 98.3% on all three) is not reproduced.
The judges are the opposite of the paper: 81-84% in distribution and 88-90% OOD, versus the paper's 54-64%,
with the errors concentrated on the neutral class as well (e.g. sonnet default, OOD: 49/60 neutral correct).
Judge consistency is 0.89-0.98, so the "stochasticity" argument against judges is weak with these models.
Consequently the paper's headline ordering (CLiF >> judge) only holds in distribution; OOD the judge wins
by 10-25 points. Caveats: different judge models, 60 rather than the paper's (unstated) OOD test size, and
the paper's Level/One-sided definitions are our reconstruction.

### Table 2 analogue: ELEPHANT score on 50 held-out Anthropic scenarios (mean +- std, lower is better)

Judge claude-sonnet-5; all 50 scored for every config. Paper: Wei 0.33, Sharma 0.37, Persona 0.43,
Contrastive add/clamp 0.45/0.47, CLiF subtract 0.37-0.42, CLiF clamp All 0.33 (best), others 0.42-0.48.

| config | ELEPHANT | validation | indirectness | framing |
|---|---|---|---|---|
| none | 0.35 +- 0.22 | 0.16 | 0.14 | 0.74 |
| prompt_wei | 0.18 +- 0.18 | 0.16 | 0.02 | 0.36 |
| prompt_sharma | 0.37 +- 0.20 | 0.30 | 0.06 | 0.74 |
| contrastive_addition | 0.35 +- 0.22 | 0.18 | 0.14 | 0.72 |
| contrastive_clamp | 0.32 +- 0.21 | 0.16 | 0.06 | 0.74 |
| persona_vectors | 0.32 +- 0.20 | 0.02 | 0.16 | 0.78 |
| clif_clamp_all | 0.41 +- 0.22 | 0.22 | 0.16 | 0.86 |
| clif_clamp_one_sided | 0.37 +- 0.23 | 0.22 | 0.10 | 0.80 |
| clif_clamp_level1 | 0.38 +- 0.23 | 0.18 | 0.12 | 0.84 |
| clif_clamp_level2 | 0.40 +- 0.24 | 0.20 | 0.14 | 0.86 |
| clif_clamp_level3 | 0.42 +- 0.21 | 0.24 | 0.14 | 0.88 |
| clif_subtract_all | 0.36 +- 0.22 | 0.16 | 0.18 | 0.74 |
| clif_subtract_one_sided | 0.36 +- 0.20 | 0.14 | 0.10 | 0.84 |
| clif_subtract_level1 | 0.38 +- 0.23 | 0.18 | 0.16 | 0.80 |
| clif_subtract_level2 | 0.36 +- 0.27 | 0.14 | 0.20 | 0.74 |
| clif_subtract_level3 | 0.39 +- 0.22 | 0.14 | 0.22 | 0.82 |

Reading. Unsteered 0.35. The Wei system prompt is the only intervention that clearly reduces the score
(0.18; framing 0.74 -> 0.36, i.e. the model starts challenging the user's premise). Persona Vectors and the
contrastive clamp give 0.32, within noise of unsteered (std 0.20-0.22, n=50, so +-0.03 is one standard
error). Every CLiF config is at or slightly above baseline (0.36-0.42); clamping F+ raises framing (0.74 ->
0.80-0.88), i.e. responses accept the premise more, not less, while subtract leaves it near baseline. So the
paper's steering claim (CLiF clamp All 0.33, best of all methods, and below Persona Vectors) is not
reproduced; here the ranking is prompt_wei < persona ~ contrastive_clamp ~ none < CLiF. The sampled outputs
remain fluent under every config (no empties, no degeneration seen in the logged samples), so the hook is
acting but the ablated features do not carry the behaviour the ELEPHANT dimensions measure. Likely causes:
the paper's steering strength (alpha for subtract, which features to clamp) is unspecified and we use the
raw difference of means once; and ELEPHANT's three dimensions were designed for advice-seeking prompts,
whereas these are opinion-agreement prompts where "validation" and "framing" barely move.

## Extension (2026-10-18): CLiF on SycoScope benchmarks

Files: `data/sycoscope_eval_seed0.jsonl`, `results/sycoscope_clif_features/*.npz`,
`results/sycoscope_clif_eval.json`, and adjacent metadata under `meta/`. Extraction used the same
`unsloth/Llama-3.1-8B-Instruct`, Goodfire layer-19 SAE, response-token mean pooling, and saved 11,883-feature
CLiF mask as the replication above. It ran on one vast.ai H100 SXM 80GB with torch 2.5.1+cu124 and
transformers 4.57.6. The exact observed maximum input length was 2,049 tokens (one row); no rows were
truncated or excluded. All 16,806 rows across 15 balanced evaluation datasets were extracted, and every
downloaded NPZ was checked against its SHA-256 metadata.

The original detector is three-class {-3, 0, +3}, while these benchmarks are binary. LR AUROC therefore
uses expected ordered class, `p(+3) * 3 + p(-3) * -3`; SVM AUROC uses the +3 decision margin minus the -3
margin. For balanced accuracy, predicted +3 is positive and predicted -3 or 0 is negative. This mapping was
fixed before evaluation. Residual-stream probes use the same mapping and training rows.

| SycoScope target | n | CLiF LR AUROC | CLiF SVM AUROC | Residual LR AUROC |
|---|---:|---:|---:|---:|
| AITA NTA flipped | 758 | 0.508 | 0.530 | 0.504 |
| AITA NTA original verdict | 168 | 0.526 | 0.554 | 0.459 |
| AITA YTA framing | 988 | 0.497 | 0.476 | 0.500 |
| AITA YTA indirectness | 1,248 | 0.571 | 0.559 | 0.767 |
| AITA YTA validation | 1,196 | 0.573 | 0.624 | 0.636 |
| Are You Sure free-form | 518 | 0.487 | 0.535 | 0.549 |
| Are You Sure multiple choice | 742 | 0.492 | 0.528 | 0.592 |
| Open-ended questions framing | 606 | 0.502 | 0.531 | 0.734 |
| Open-ended questions indirectness | 504 | 0.593 | 0.736 | 0.720 |
| Open-ended questions validation | 1,270 | 0.564 | 0.656 | 0.671 |
| Self-sycophancy framing | 1,314 | 0.464 | 0.467 | 0.511 |
| Self-sycophancy indirectness | 550 | 0.503 | 0.542 | 0.573 |
| Self-sycophancy validation | 1,062 | 0.508 | 0.554 | 0.661 |
| SycoScope praise | 3,792 | 0.651 | 0.614 | 0.668 |
| TruthfulQA false-answer agreement | 2,090 | 0.481 | 0.500 | 0.579 |

Reading. The transferred CLiF detector is useful on SycoScope praise (LR 0.651) and some indirectness or
validation targets (best CLiF result: SVM 0.736 on open-ended-question indirectness), but it is at or near
chance on most framing, Are You Sure, self-sycophancy, and TruthfulQA targets. The residual-stream baseline
is usually stronger and reaches 0.767 on AITA indirectness. Thus CLiF's original three-way cascade detection
does not transfer as a broadly reliable binary sycophancy detector across the existing SycoScope datasets.

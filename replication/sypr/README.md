# SyPR paper replication

This directory reproduces the released evaluation protocol from *Sycophantic Praise: Evaluating Excessive Praise in Language Models* (arXiv:2606.07441v1).

## Fixed sources

- Author code: `cincynlp/sycophantic-praise` at `9a8be479986f6b2843d87745af8a8d0efb751637`.
- Benchmark: `vennemeyerd/sycophantic-praise` at `97f37cf9d627a055cfe375bf911439ca1f9f1fd3`.
- Benchmark JSONL: 13,200 rows, SHA-256 `16b41aad1cf5d30de34bd887e1d0f8c17ad837d811a11751bb1361b08a002a2a`.
- Generator: `Qwen/Qwen3-30B-A3B-Instruct-2507`, greedy decoding, 512 new-token cap.

`generate_qwen.py --prompt-format raw` reproduces the authors' client exactly: the system prompt, persona context, prior turns and utterance are joined as plain `System:`/`User:`/`Assistant:` lines with no chat template. `--prompt-format chat` (the default) applies Qwen's chat template to the same message sequence instead. Both runs exist; see the results below.

## Judges

The paper uses GPT-5.4 for automatic sentence-level praise annotation. The first full run used `claude-sonnet-5` at the user's request (`--provider anthropic`, thinking disabled); the raw-prompt run was additionally judged with GPT-5.4 (`--provider openai --judge-model gpt-5.4`, `reasoning_effort="none"`, strict JSON schema, the authors' developer message). One judgment is made per sentence and the paper's seven-point exemplar scale is preserved. Sonnet results are a modified replication; the GPT-5.4 run uses the paper's judge.

The Sonnet runs and the GPT-5.4 run were judged with a prompt whose instruction sentence read "...that best matches its intensity." The upstream text ends "...and return it as intensity_7." The code now uses the upstream wording; the structured output schema already forced the field name, so the omission did not affect which field was returned. The GPT-5.4 schema at run time accepted any JSON number for `intensity_7`; every value returned was an integer from 1 to 7, which the schema and the normalizer now require.

## Human annotations

The paper reports two human-label collections:

1. 1,909 sentences labeled for praise target and intensity, used to validate candidate automatic annotators.
2. 1,000 complete responses labeled as warranted or excessive praise, split into train and held-out test data to fit and validate the SyPR parameters.

The public repository does not include these labels or parameter-training code. The released parameters permit faithful scoring, but the human-agreement tables and a from-scratch parameter fit cannot currently be reproduced.

## Pipeline

Download the pinned benchmark directly from Hugging Face, then generate on a GPU:

```bash
python -m replication.sypr.generate_qwen \
  --benchmark replication/sypr/data/benchmark.jsonl \
  --output replication/sypr/results/qwen3_30b.responses.jsonl
```

Exclude truncated generations:

```bash
python -m replication.sypr.sample_responses \
  --input replication/sypr/results/qwen3_30b.responses.jsonl \
  --output replication/sypr/results/qwen3_30b.complete.jsonl \
  --fraction 1.0 \
  --seed 0
```

Judge every complete response with Sonnet 5:

```bash
python -m replication.sypr.judge_sonnet \
  --input replication/sypr/results/qwen3_30b.complete.jsonl \
  --output replication/sypr/results/qwen3_30b.complete.sonnet5.judged.jsonl
```

Apply the published SyPR metric:

```bash
python -m replication.sypr.score \
  --benchmark replication/sypr/data/benchmark.jsonl \
  --judged replication/sypr/results/qwen3_30b.complete.sonnet5.judged.jsonl \
  --responses replication/sypr/results/qwen3_30b.complete.jsonl \
  --output replication/sypr/results/qwen3_30b.complete.sonnet5.scored.jsonl
```

Every completed stage writes a sibling `meta/*.meta.json` provenance record. Runtime data and results should remain uncommitted until validated; finalized generated and judged data follow the repository's normal artifact policy.

## Full-run results

Of the 13,200 generated responses, 4,797 (36.34%) reached the 512-token generation cap and were excluded before judging. The remaining 8,403 responses were judged in full, comprising 68,080 sentence judgments. The published SyPR metric classified 5,050 responses as positive, for an overall rate of 60.10%.

| Domain | Complete responses | SyPR positive | SyPR rate |
| --- | ---: | ---: | ---: |
| GSM8K | 1,941 | 425 | 21.90% |
| Long-form moral reasoning | 591 | 528 | 89.34% |
| MMLU chemistry | 662 | 216 | 32.63% |
| MMLU economics | 1,236 | 436 | 35.28% |
| Pseudo-profundity | 3,973 | 3,445 | 86.71% |
| **Overall** | **8,403** | **5,050** | **60.10%** |

The judge calls used the same `claude-sonnet-5` model, fixed prompt, exemplar scale, disabled thinking, and structured output schema throughout. API transport changed during the run because of provider spending limits; the exact split is recorded in the judged artifact's metadata.

## Raw-prompt run and GPT-5.4 judge

The second run regenerated all 13,200 responses with `--prompt-format raw` (greedy, 512 new tokens, bf16, H100; `results/qwen3_30b.raw.responses.jsonl`). 5,831 responses (44.2%) hit the token cap, against 4,797 (36.3%) under the chat template. The increase comes from the plain serialization: 1,352 raw responses continue the prompt format with a fabricated `User:` or `Assistant:` turn or an echoed `System:` line (1,300 of them truncated). The 7,369 complete raw responses were judged with Sonnet 5; all 13,200 raw responses were judged with GPT-5.4.

`subset_judged.py` derives two views from the full GPT-5.4 judgment without new API calls: `--mode complete` keeps the 7,369 untruncated responses, and `--mode echo-cut` drops every sentence from the first fake-turn marker onward (16,399 sentences removed from the 1,352 echo rows). The response text is trimmed to the kept sentences, so a sentence the marker splits is dropped from both the text and the judgments.

SyPR rate, % (n):

| Group | Chat, Sonnet 5, complete | Raw, Sonnet 5, complete | Raw, GPT-5.4, complete | Raw, GPT-5.4, echo-cut, all | Raw, GPT-5.4, as-is, all (paper protocol) | Paper (GPT-5.4) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| GSM8K | 21.9 (1,941) | 22.1 (1,402) | 17.5 (1,402) | 20.5 (2,000) | 27.5 (2,000) | |
| Long-form moral reasoning | 89.3 (591) | 93.0 (762) | 92.3 (762) | 91.3 (2,400) | 91.5 (2,400) | |
| MMLU chemistry | 32.6 (662) | 40.7 (508) | 34.6 (508) | 28.5 (2,000) | 29.7 (2,000) | |
| MMLU economics | 35.3 (1,236) | 38.1 (1,139) | 36.8 (1,139) | 36.3 (2,000) | 38.9 (2,000) | |
| Pseudo-profundity | 86.7 (3,973) | 88.9 (3,558) | 90.6 (3,558) | 89.7 (4,800) | 91.6 (4,800) | |
| Explicit persona | 65.1 (3,837) | 70.8 (3,882) | 70.7 (3,882) | 67.2 (6,000) | 69.5 (6,000) | 36.2 |
| Naturalistic persona | 64.5 (3,651) | 67.8 (2,852) | 67.5 (2,852) | 66.2 (6,000) | 67.4 (6,000) | 21.9 |
| Unprompted | 58.0 (4,311) | 62.1 (3,710) | 61.9 (3,710) | 59.2 (6,600) | 61.7 (6,600) | 20.4 |
| Asked for evaluation | 62.3 (4,092) | 68.9 (3,659) | 67.5 (3,659) | 65.1 (6,600) | 67.2 (6,600) | 37.7 |
| **Overall** | **60.1 (8,403)** | **65.5 (7,369)** | **64.7 (7,369)** | **62.2 (13,200)** | **64.5 (13,200)** | **29.0** |

Observed praise (at least one praise sentence) is 88.7% under the paper protocol column against 81.8% in the paper.

What the comparison rules out:

- Chat template. On the 6,122 artifacts complete in both runs, chat scores 62.6% and raw 61.5% under the same judge, with 87.7% per-item agreement.
- Judge. On the 7,369 complete raw responses, Sonnet 5 and GPT-5.4 agree on 92.4% of sentence praise/not-praise labels and on 92.7% of per-response SyPR outcomes (65.5% vs 64.7%).
- Truncation exclusion. Including truncated responses changes the overall rate by under 1 point (64.7% vs 64.5%); it matters only within GSM8K, where truncated rows are mostly echo rows.
- Echo contamination. Cutting at the first fake-turn marker lowers the overall rate by 2.3 points (64.5% to 62.2%).
- Scoring code. Prompt, sentence splitter, exemplars, judge call, label mapping, formula, parameters and threshold were checked line by line against the upstream repository (Codex adversarial review plus manual diff); rescoring the committed chat run reproduces its summary exactly.

The remaining gap is 35 points on the same model, benchmark, decoding and judge. The paper reports the reasoning-vs-social asymmetry and the persona split, but not truncation handling, the token cap's effect, or any judge-sensitivity analysis; it reports judge validation only for GPT-5.4 and Qwen3-30B as annotators. Our runs are near-saturated on the social domains (89-93% for moral reasoning and pseudo-profundity) while the reasoning domains sit at 18-39%. A 29% overall rate on this domain mix would need the social domains to be far lower than we observe.

## Sampled runs

Two further chat-template runs use nucleus sampling with seed 0 and batch size 16, one response per artifact, and were judged with GPT-5.4 on all 13,200 responses with truncated responses included, as in the paper protocol: `Qwen/Qwen3-30B-A3B-Instruct-2507` at temperature 0.7, top-p 0.8, top-k 20, and `meta-llama/Llama-3.1-8B-Instruct` at temperature 0.6, top-p 0.9, no top-k. Files: `results/qwen3_30b.sampled.*` and `results/llama31_8b.sampled.*`.

SyPR rate, % (n = 13,200 overall):

| Group | Qwen3-30B sampled | Llama 3.1 8B sampled | Paper, Qwen3-30B |
| --- | ---: | ---: | ---: |
| GSM8K | 16.4 | 1.4 | |
| Long-form moral reasoning | 90.8 | 69.5 | |
| MMLU chemistry | 25.8 | 3.8 | |
| MMLU economics | 35.8 | 8.7 | |
| Pseudo-profundity | 90.3 | 67.4 | |
| Explicit persona | 67.5 | 43.2 | 36.2 |
| Naturalistic persona | 64.5 | 42.3 | 21.9 |
| Unprompted | 59.3 | 38.0 | 20.4 |
| Asked for evaluation | 63.0 | 40.5 | 37.7 |
| **Overall** | **61.2** | **39.2** | **29.0** |

The Llama run stopped on a CUDA out-of-memory error partway through and was resumed from the completed prefix. The generator then seeded each batch by its offset from the resume point, so batches after the interruption drew different samples than an uninterrupted run would have. The generator now seeds by shard position, so a resumed run reproduces an uninterrupted one, but the committed Llama file is not reproducible bit for bit from a single run. The Qwen sampled run was not interrupted.

## Decoding audit

Decoding matches the authors' client: greedy (`do_sample=False`), one response per artifact, 512 new tokens, bf16, stop at any of the checkpoint's EOS ids. Batched generation (16 or 8 per batch) is the only difference from the client's default of one, which affects bit-exactness, not the protocol. A hand audit of 60 responses found all understandable; in the raw run 89% of responses are clean and 11% carry a clean answer followed by a contaminated fake-turn tail.

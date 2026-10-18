# SyPR paper replication

This directory reproduces the released evaluation protocol from *Sycophantic Praise: Evaluating Excessive Praise in Language Models* (arXiv:2606.07441v1).

## Fixed sources

- Author code: `cincynlp/sycophantic-praise` at `9a8be479986f6b2843d87745af8a8d0efb751637`.
- Benchmark: `vennemeyerd/sycophantic-praise` at `97f37cf9d627a055cfe375bf911439ca1f9f1fd3`.
- Benchmark JSONL: 13,200 rows, SHA-256 `16b41aad1cf5d30de34bd887e1d0f8c17ad837d811a11751bb1361b08a002a2a`.
- Generator: `Qwen/Qwen3-30B-A3B-Instruct-2507`, greedy decoding, 512 new-token cap.

The generator uses the model's actual chat template. This preserves the message sequence in the authors' client while avoiding its plain `System:`/`User:` serialization, which is not Qwen's instruct format.

## Intentional deviation

The paper uses GPT-5.4 for automatic sentence-level praise annotation. This replication uses `claude-sonnet-5` at the user's request. Adaptive thinking is disabled, one judgment is made per sentence, and the paper's seven-point exemplar scale is preserved. Results must therefore be labeled a modified replication rather than an exact reproduction of the judge-dependent numbers.

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

Judge responses with Sonnet 5:

```bash
python -m replication.sypr.judge_sonnet \
  --input replication/sypr/results/qwen3_30b.responses.jsonl \
  --output replication/sypr/results/qwen3_30b.sonnet5.judged.jsonl
```

Apply the published SyPR metric:

```bash
python -m replication.sypr.score \
  --benchmark replication/sypr/data/benchmark.jsonl \
  --judged replication/sypr/results/qwen3_30b.sonnet5.judged.jsonl \
  --output replication/sypr/results/qwen3_30b.sonnet5.scored.jsonl
```

Every completed stage writes a sibling `meta/*.meta.json` provenance record. Runtime data and results should remain uncommitted until validated; finalized generated and judged data follow the repository's normal artifact policy.

# Sycophancy detector audit: summary

Branch `audit/existing-methods`, model `meta-llama/Llama-3.1-8B-Instruct`. Full review: `REVIEW.md` (published as the
"Sycophancy detector audit" artifact); per-token heatmaps: `token_heatmaps.html` ("Detector token heatmaps" artifact).
Notes on porting and every recipe decision: `docs/AUDIT_NOTES.md`, `design_choices.jsonl`. All numbers below are read
from `parity.csv` and `coverage.csv`.

## What was done

Five published white-box sycophancy detectors were rebuilt on Llama-3.1-8B-Instruct from each method's own source data
and code (pinned commits), compared with each paper, then scored on this repo's 15 balanced judged benchmarks at each
method's own read position (benchmark activations re-extracted in the method's input format), next to the committed
contrastive probes. Three independent read-only reviews by a separate agent found no bug that changes a reported number.

## Reproduction against the papers

| Method | What the paper reports | Ours | Verdict |
| --- | --- | --- | --- |
| CAA | Separation emerges about 1/3 through the layers (L10, Llama-2-7B); no AUROC | held-out AUROC first >= 0.9 at L12; 0.94 at L13 | consistent (qualitative) |
| Vennemeyer et al. | Llama-3.1-8B AUROC and direction-geometry curves (Fig. 8b, 10b) | literal pooling selects BOS on Llama-3.1; neither reconstruction matches all curves (SyPr off by up to 0.20 with last-token pooling; SyA-GA cosine 11/15 layers within 0.1) | partial mismatch |
| Genadi et al. | sparse mid-layer heads (Gemma-3, Llama-3.2); no Llama-3.1-8B numbers | best head L12H12, held-out AUROC 0.91 (answer mean) | no numeric comparison |
| Pandey | Table 1: 21/32 sycophancy heads shared with lying heads, Spearman 0.88 | ours 27/32 shared (chance 1.0), Spearman 0.97 (circuit_overlap.py); ours 23/32 shared (chance 1.0), Spearman 0.96 (breadth.py); their LR-27 probe held-out AUROC 0.95 | partly matches |
| Persona Vectors | Table 2: monitoring r 0.798 overall, 0.669 within condition (model not stated) | overall r 0.95, within-condition 0.62 (GPT-4.1-mini judge, about $1.15) | within matches; overall higher |

## Out-of-distribution performance (own read positions)

AUROC for each benchmark's target label. CI counts: cells whose 95% bootstrap CI lies above 0.5 / below 0.5 / total.

| Detector set | Detectors | Median AUROC | Max | CI above / below / cells |
| --- | --- | --- | --- | --- |
| CAA | 2 | 0.60 | 0.82 | 20 / 1 / 30 |
| Vennemeyer et al. | 12 | 0.53 | 0.79 | 75 / 38 / 180 |
| Genadi et al. | 2 | 0.56 | 0.71 | 16 / 5 / 30 |
| Pandey | 4 | 0.44 | 0.59 | 11 / 34 / 60 |
| Persona Vectors | 1 | 0.44 | 0.66 | 5 / 8 / 15 |
| Contrastive universal (P00) | 1 per benchmark | 0.50 | 0.65 | 5 / 5 / 15 |
| Contrastive matched-cell pair | 1 per benchmark | 0.64 | 0.76 | 9 / 5 / 15 |
| Contrastive, post-hoc max over taxonomy pairs | 1 per benchmark | 0.68 | 0.88 | 15 / 0 / 15 |

Per benchmark: the best existing detector is a post-hoc maximum over 21 detectors and the last column a post-hoc maximum
over 8 contrastive pairs; both are optimistic.

| Benchmark | Cell | Best existing detector (post hoc) | Universal | Matched-cell | Max contrastive (post hoc) |
| --- | --- | --- | --- | --- | --- |
| AYS freeform | Position-Verifiable / Explicit | 0.64 (vennemeyer SyA) | 0.51 | 0.41 | 0.56 |
| AYS multiple choice | Position-Verifiable / Explicit | 0.70 (vennemeyer SyPr) | 0.40 | 0.42 | 0.65 |
| TruthfulQA false answer | Position-Verifiable / Explicit | 0.65 (vennemeyer SyPr) | 0.52 | 0.49 | 0.59 |
| AITA NTA flipped | Position-Subjective / Explicit | 0.69 (caa) | 0.49 | 0.72 | 0.72 |
| AITA NTA original | Position-Subjective / Explicit | 0.82 (caa) | 0.31 | 0.73 | 0.74 |
| AITA YTA framing | Position-Subjective / Implicit | 0.63 (persona) | 0.49 | 0.54 | 0.62 |
| OEQ framing | Position-Subjective / Implicit | 0.73 (caa) | 0.41 | 0.36 | 0.67 |
| SS framing | Position-Subjective / Implicit | 0.65 (persona) | 0.55 | 0.65 | 0.68 |
| AITA YTA validation | Person-Traits / Explicit | 0.68 (caa) | 0.50 | 0.65 | 0.78 |
| OEQ validation | Person-Traits / Explicit | 0.75 (vennemeyer SyPr) | 0.43 | 0.76 | 0.85 |
| SS validation | Person-Traits / Explicit | 0.68 (caa) | 0.65 | 0.75 | 0.88 |
| SyPR praise | Person-Traits / Explicit | 0.68 (vennemeyer SyPr) | 0.53 | 0.67 | 0.75 |
| AITA YTA indirectness | Person-Traits / Implicit | 0.59 (vennemeyer SyPr) | 0.61 | 0.41 | 0.66 |
| OEQ indirectness | Person-Traits / Implicit | 0.77 (vennemeyer SyPr) | 0.43 | 0.64 | 0.82 |
| SS indirectness | Person-Traits / Implicit | 0.58 (pandey lr_top15) | 0.55 | 0.43 | 0.68 |

## Caveats recorded in the review

- Vennemeyer's pooling had to be reconstructed; both versions are scored and labelled.
- Pandey's code-faithful read position drifts on multi-turn prompts (AYS, multi-turn SyPR); affected cells are flagged.
- CAA on free-form responses uses the mean projection over response tokens (our construction; their A/B letter does not exist).
- Plain-text rendering of multi-turn conversations (Vennemeyer native format) uses our turn separator.
- Detector thresholds come from source data; on benchmarks some detectors predict a single class.

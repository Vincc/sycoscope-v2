# Sycophancy detector audit

Branch `audit/existing-methods`. Model `meta-llama/Llama-3.1-8B-Instruct`. All numbers are generated from the CSV files in this directory by `audit/write_review.py`.

## 1. Summary

| Detector | Status | Reason |
| --- | --- | --- |
| CAA sycophancy vector | reproduced | difference-in-means on the 1,000 A/B items, Llama-3.1 chat template |
| Vennemeyer SyA, GA, SyPr | reproduced, parity mismatch | literal 'last' pooling selects BOS on Llama-3.1; two reconstructed poolings scored (user decision) |
| Genadi head probes | reproduced | extension/ recipe on TruthfulQA pushback dialogues; best head + LR on top-16 [our addition] |
| Pandey shared-circuit direction | reproduced (code-faithful position) | residual DIM at L27 and top-15 heads; LR on heads [our addition] |
| Persona Vectors (sycophantic) | held at judge step | needs 4000 GPT-4.1-mini calls; rollouts and activations cached |

Not in this task:

- CLiF: code not released.
- Baez et al.: repository unreachable.
- Cheng et al.: labels need GPT-4o (OSF data not checked for labels in this pass).
- Goodfire SAE features: no sycophancy features published.
- Wang et al.: not a detector.
- Papadatos & Freedman: they probe a reward model.
- Beacon: no code.
- Steering and causal tests: out of scope for this pass.
- Other models: Llama-3.1-8B-Instruct only.
- New detection methods: out of scope.
- Any LLM API call: none made; Persona Vectors held at the judge step.

Environment: GPU NVIDIA A100-SXM4-80GB; GPU memory 81920 MiB; NVIDIA driver 580.173.02; CUDA (driver) 13.0; torch 2.12.1+cu130 (CUDA 13.0); transformers 5.13.0; numpy 2.5.1; scikit-learn 1.9.0; Python 3.13.15; datasets (source build only) 5.0.1; Instance vast.ai, /workspace not a volume; Model meta-llama/Llama-3.1-8B-Instruct, bfloat16

## 2. Parity (held-out AUROC on each method's own source data)

| Method | Unit | Variant | Layer | Our AUROC [95% CI] | n | Paper value | Paper reference | Verdict |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| CAA | caa | recipe layer (paper's 7B steering layer) | 13 | 0.943 [0.918, 0.962] | 400 |  | Sec. 3.2 and Fig. 2 (p. 4); Sec. 4.1 (p. 4-5) | qualitative: held-out AUROC first >= 0.9 at L12; paper: behavioural clustering emerges ~1/3 depth (L10, Llama-2-7B) |
| CAA | caa | val-selected layer | 26 | 0.935 [0.910, 0.957] | 400 |  | Sec. 3.2 and Fig. 2 (p. 4); Sec. 4.1 (p. 4-5) | qualitative: held-out AUROC first >= 0.9 at L12; paper: behavioural clustering emerges ~1/3 depth (L10, Llama-2-7B) |
| Vennemeyer | SyA | math plain last_content | 29 | 0.994 [0.991, 0.997] | 800 | see by-layer table | App. C.4, Fig. 8b (p. 20), SIMPLE MATH; values read by eye from the figure at 400 dpi, about +-0.02; x axis is their layer index (L k = block k, L32 repeats block 31) | mismatch (max /diff/ 0.08 at L31) |
| Vennemeyer | GA | math plain last_content | 29 | 0.999 [0.998, 1.000] | 800 | see by-layer table | App. C.4, Fig. 8b (p. 20), SIMPLE MATH; values read by eye from the figure at 400 dpi, about +-0.02; x axis is their layer index (L k = block k, L32 repeats block 31) | mismatch (max /diff/ 0.09 at L25) |
| Vennemeyer | SyPr | math plain last_content | 11 | 0.938 [0.923, 0.952] | 800 | see by-layer table | App. C.4, Fig. 8b (p. 20), SIMPLE MATH; values read by eye from the figure at 400 dpi, about +-0.02; x axis is their layer index (L k = block k, L32 repeats block 31) | mismatch (max /diff/ 0.20 at L5) |
| Vennemeyer | SyA | math plain resp_all | 29 | 0.918 [0.900, 0.935] | 800 | see by-layer table | App. C.4, Fig. 8b (p. 20), SIMPLE MATH; values read by eye from the figure at 400 dpi, about +-0.02; x axis is their layer index (L k = block k, L32 repeats block 31) | mismatch (max /diff/ 0.13 at L2) |
| Vennemeyer | GA | math plain resp_all | 30 | 0.912 [0.884, 0.936] | 800 | see by-layer table | App. C.4, Fig. 8b (p. 20), SIMPLE MATH; values read by eye from the figure at 400 dpi, about +-0.02; x axis is their layer index (L k = block k, L32 repeats block 31) | mismatch (max /diff/ 0.14 at L31) |
| Vennemeyer | SyPr | math plain resp_all | 18 | 0.959 [0.946, 0.970] | 800 | see by-layer table | App. C.4, Fig. 8b (p. 20), SIMPLE MATH; values read by eye from the figure at 400 dpi, about +-0.02; x axis is their layer index (L k = block k, L32 repeats block 31) | mismatch (max /diff/ 0.11 at L31) |
| Vennemeyer | SyA | math chat last_content | 30 | 0.970 [0.961, 0.979] | 800 |  | App. C.4, Fig. 8b (p. 20), SIMPLE MATH; values read by eye from the figure at 400 dpi, about +-0.02; x axis is their layer index (L k = block k, L32 repeats block 31) | no paper value (chat variant) |
| Vennemeyer | GA | math chat last_content | 30 | 0.991 [0.984, 0.997] | 800 |  | App. C.4, Fig. 8b (p. 20), SIMPLE MATH; values read by eye from the figure at 400 dpi, about +-0.02; x axis is their layer index (L k = block k, L32 repeats block 31) | no paper value (chat variant) |
| Vennemeyer | SyPr | math chat last_content | 16 | 0.906 [0.883, 0.929] | 800 |  | App. C.4, Fig. 8b (p. 20), SIMPLE MATH; values read by eye from the figure at 400 dpi, about +-0.02; x axis is their layer index (L k = block k, L32 repeats block 31) | no paper value (chat variant) |
| Vennemeyer | SyA | math chat resp_all | 30 | 0.921 [0.902, 0.941] | 800 |  | App. C.4, Fig. 8b (p. 20), SIMPLE MATH; values read by eye from the figure at 400 dpi, about +-0.02; x axis is their layer index (L k = block k, L32 repeats block 31) | no paper value (chat variant) |
| Vennemeyer | GA | math chat resp_all | 30 | 0.896 [0.872, 0.921] | 800 |  | App. C.4, Fig. 8b (p. 20), SIMPLE MATH; values read by eye from the figure at 400 dpi, about +-0.02; x axis is their layer index (L k = block k, L32 repeats block 31) | no paper value (chat variant) |
| Vennemeyer | SyPr | math chat resp_all | 16 | 0.966 [0.955, 0.976] | 800 |  | App. C.4, Fig. 8b (p. 20), SIMPLE MATH; values read by eye from the figure at 400 dpi, about +-0.02; x axis is their layer index (L k = block k, L32 repeats block 31) | no paper value (chat variant) |
| Pandey | residual DIM | code-faithful position | 27 | 0.924 [0.885, 0.958] | 200 |  | Sec. 3.1-3.2 (p. 3-4); App. U, Table 18 (p. 24-25) | no published Llama-3.1-8B value |
| Pandey | LR top-15 heads [ours] | code-faithful position | [28, 29, 30, 31] | 0.912 [0.870, 0.948] | 200 |  | Sec. 3.1-3.2 (p. 3-4); App. U, Table 18 (p. 24-25) | no published Llama-3.1-8B value |
| Genadi | best head | answer_mean pooling | [12] | 0.911 [0.897, 0.924] | 1640 |  | Sec. 4.1, Fig. 2-3 (p. 4); App. B Fig. 9-10; App. H Table 8 (p. 17) | no published Llama-3.1-8B value |
| Genadi | LR top-16 heads [ours] | answer_mean pooling | [9, 10, 11, 12, 14, 16, 31] | 0.977 [0.971, 0.982] | 1640 |  | Sec. 4.1, Fig. 2-3 (p. 4); App. B Fig. 9-10; App. H Table 8 (p. 17) | no published Llama-3.1-8B value |
| Genadi | best head | last pooling | [12] | 0.948 [0.939, 0.957] | 1640 |  | Sec. 4.1, Fig. 2-3 (p. 4); App. B Fig. 9-10; App. H Table 8 (p. 17) | no published Llama-3.1-8B value |
| Genadi | LR top-16 heads [ours] | last pooling | [9, 10, 11, 12, 13, 16, 30, 31] | 0.994 [0.991, 0.997] | 1640 |  | Sec. 4.1, Fig. 2-3 (p. 4); App. B Fig. 9-10; App. H Table 8 (p. 17) | no published Llama-3.1-8B value |
| Persona Vectors | sycophantic | held at judge step | 15 | held |  |  | verified: App. p. 30 and Fig. 13, chosen by steering at each layer; paper counts layers from 1 (block 15 here) | held: 4000 GPT-4.1-mini judge calls needed |

Vennemeyer, SIMPLE MATH, plain text: our test AUROC by layer against Fig. 8b (read by eye, about ±0.02).

| Pooling | Unit | Layer | Paper | Ours | Diff |
| --- | --- | --- | --- | --- | --- |
| last_content | syc | 2 | 0.82 | 0.86 | +0.04 |
| last_content | syc | 5 | 0.815 | 0.84 | +0.02 |
| last_content | syc | 8 | 0.815 | 0.85 | +0.04 |
| last_content | syc | 10 | 0.815 | 0.85 | +0.04 |
| last_content | syc | 12 | 0.86 | 0.87 | +0.01 |
| last_content | syc | 15 | 0.905 | 0.92 | +0.02 |
| last_content | syc | 20 | 0.91 | 0.91 | +0.00 |
| last_content | syc | 25 | 0.91 | 0.97 | +0.06 |
| last_content | syc | 28 | 0.93 | 0.97 | +0.04 |
| last_content | syc | 30 | 0.96 | 1.00 | +0.04 |
| last_content | syc | 31 | 0.91 | 0.99 | +0.08 |
| last_content | ga | 2 | 0.7 | 0.67 | -0.03 |
| last_content | ga | 5 | 0.73 | 0.75 | +0.02 |
| last_content | ga | 8 | 0.73 | 0.74 | +0.01 |
| last_content | ga | 10 | 0.75 | 0.76 | +0.01 |
| last_content | ga | 12 | 0.75 | 0.81 | +0.06 |
| last_content | ga | 15 | 0.84 | 0.79 | -0.05 |
| last_content | ga | 20 | 0.86 | 0.82 | -0.04 |
| last_content | ga | 25 | 0.87 | 0.96 | +0.09 |
| last_content | ga | 28 | 0.9 | 0.96 | +0.06 |
| last_content | ga | 30 | 0.98 | 1.00 | +0.02 |
| last_content | ga | 31 | 0.95 | 0.99 | +0.04 |
| last_content | pr | 2 | 0.825 | 0.69 | -0.13 |
| last_content | pr | 5 | 0.94 | 0.74 | -0.20 |
| last_content | pr | 8 | 0.965 | 0.81 | -0.16 |
| last_content | pr | 10 | 0.945 | 0.85 | -0.10 |
| last_content | pr | 12 | 0.94 | 0.91 | -0.03 |
| last_content | pr | 15 | 0.955 | 0.89 | -0.06 |
| last_content | pr | 20 | 0.94 | 0.83 | -0.11 |
| last_content | pr | 25 | 0.9 | 0.78 | -0.12 |
| last_content | pr | 28 | 0.89 | 0.79 | -0.10 |
| last_content | pr | 30 | 0.855 | 0.78 | -0.08 |
| last_content | pr | 31 | 0.765 | 0.77 | +0.01 |
| resp_all | syc | 2 | 0.82 | 0.69 | -0.13 |
| resp_all | syc | 5 | 0.815 | 0.81 | -0.00 |
| resp_all | syc | 8 | 0.815 | 0.82 | +0.01 |
| resp_all | syc | 10 | 0.815 | 0.83 | +0.02 |
| resp_all | syc | 12 | 0.86 | 0.88 | +0.02 |
| resp_all | syc | 15 | 0.905 | 0.86 | -0.04 |
| resp_all | syc | 20 | 0.91 | 0.84 | -0.07 |
| resp_all | syc | 25 | 0.91 | 0.86 | -0.05 |
| resp_all | syc | 28 | 0.93 | 0.89 | -0.04 |
| resp_all | syc | 30 | 0.96 | 0.91 | -0.05 |
| resp_all | syc | 31 | 0.91 | 0.88 | -0.03 |
| resp_all | ga | 2 | 0.7 | 0.69 | -0.01 |
| resp_all | ga | 5 | 0.73 | 0.72 | -0.01 |
| resp_all | ga | 8 | 0.73 | 0.71 | -0.02 |
| resp_all | ga | 10 | 0.75 | 0.79 | +0.04 |
| resp_all | ga | 12 | 0.75 | 0.80 | +0.05 |
| resp_all | ga | 15 | 0.84 | 0.80 | -0.04 |
| resp_all | ga | 20 | 0.86 | 0.77 | -0.09 |
| resp_all | ga | 25 | 0.87 | 0.76 | -0.11 |
| resp_all | ga | 28 | 0.9 | 0.76 | -0.14 |
| resp_all | ga | 30 | 0.98 | 0.91 | -0.07 |
| resp_all | ga | 31 | 0.95 | 0.81 | -0.14 |
| resp_all | pr | 2 | 0.825 | 0.86 | +0.03 |
| resp_all | pr | 5 | 0.94 | 0.90 | -0.04 |
| resp_all | pr | 8 | 0.965 | 0.93 | -0.03 |
| resp_all | pr | 10 | 0.945 | 0.93 | -0.01 |
| resp_all | pr | 12 | 0.94 | 0.93 | -0.01 |
| resp_all | pr | 15 | 0.955 | 0.95 | -0.01 |
| resp_all | pr | 20 | 0.94 | 0.95 | +0.01 |
| resp_all | pr | 25 | 0.9 | 0.93 | +0.03 |
| resp_all | pr | 28 | 0.89 | 0.92 | +0.03 |
| resp_all | pr | 30 | 0.855 | 0.91 | +0.06 |
| resp_all | pr | 31 | 0.765 | 0.88 | +0.11 |

## 3. Coverage (AUROC [95% CI] on the balanced judged benchmarks)

Columns grouped by the benchmark's taxonomy cell (* = cell inferred, not in the SPEC table). Native = the scored position equals where the method reads activations; ≈ approximate; ✗ not native. Heatmap: `coverage_heatmap.png`.

| Detector | Built on | Native | AYS freeform (Position-Verifiable / Explicit; syco, n=518) | AYS multiple choice (Position-Verifiable / Explicit; syco, n=742) | TruthfulQA false answer* (Position-Verifiable / Explicit; false_answer, n=2090) | AITA NTA flipped (Position-Subjective / Explicit; both_nta, n=758) | AITA NTA original* (Position-Subjective / Explicit; verdict_nta, n=168) | AITA YTA framing (Position-Subjective / Implicit; framing, n=988) | OEQ framing (Position-Subjective / Implicit; framing, n=606) | SS framing (Position-Subjective / Implicit; framing, n=1314) | AITA YTA validation (Person-Traits / Explicit; validation, n=1196) | OEQ validation (Person-Traits / Explicit; validation, n=1270) | SS validation (Person-Traits / Explicit; validation, n=1062) | SyPR praise (Person-Traits / Explicit; sycophantic_praise, n=3792) | AITA YTA indirectness (Person-Traits / Implicit; indirectness, n=1248) | OEQ indirectness (Person-Traits / Implicit; indirectness, n=504) | SS indirectness (Person-Traits / Implicit; indirectness, n=550) |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| caa · L13 recipe · last_prompt (non-native) | Position-Subjective / Explicit | ✗ | 0.45 [0.40, 0.49] | 0.47 [0.43, 0.51] | 0.49 [0.46, 0.51] | 0.75 [0.72, 0.79] | 0.84 [0.77, 0.89] | 0.65 [0.62, 0.69] | 0.75 [0.71, 0.78] | 0.60 [0.57, 0.63] | 0.64 [0.60, 0.67] | 0.67 [0.64, 0.70] | 0.71 [0.68, 0.74] | 0.57 [0.55, 0.59] | 0.40 [0.37, 0.43] | 0.64 [0.60, 0.69] | 0.56 [0.51, 0.61] |
| caa · L13 recipe · first5 (non-native) | Position-Subjective / Explicit | ✗ | 0.42 [0.38, 0.47] | 0.46 [0.42, 0.51] | 0.50 [0.48, 0.53] | 0.73 [0.70, 0.77] | 0.71 [0.64, 0.79] | 0.61 [0.58, 0.65] | 0.71 [0.68, 0.75] | 0.60 [0.57, 0.63] | 0.61 [0.58, 0.64] | 0.66 [0.63, 0.69] | 0.77 [0.75, 0.80] | 0.61 [0.60, 0.63] | 0.47 [0.44, 0.50] | 0.70 [0.66, 0.75] | 0.57 [0.53, 0.62] |
| caa · L13 recipe · response (non-native) | Position-Subjective / Explicit | ✗ | 0.51 [0.46, 0.56] | 0.55 [0.50, 0.58] | 0.40 [0.38, 0.43] | 0.69 [0.65, 0.73] | 0.82 [0.76, 0.88] | 0.60 [0.56, 0.64] | 0.73 [0.69, 0.77] | 0.60 [0.57, 0.63] | 0.68 [0.65, 0.71] | 0.68 [0.64, 0.71] | 0.68 [0.65, 0.71] | 0.61 [0.60, 0.63] | 0.49 [0.45, 0.52] | 0.69 [0.64, 0.73] | 0.54 [0.49, 0.58] |
| caa · L26 · last_prompt (non-native) | Position-Subjective / Explicit | ✗ | 0.32 [0.28, 0.36] | 0.50 [0.46, 0.55] | 0.53 [0.50, 0.55] | 0.50 [0.46, 0.54] | 0.36 [0.29, 0.45] | 0.44 [0.41, 0.48] | 0.74 [0.70, 0.78] | 0.56 [0.53, 0.59] | 0.50 [0.47, 0.53] | 0.60 [0.57, 0.63] | 0.59 [0.55, 0.62] | 0.51 [0.49, 0.52] | 0.57 [0.54, 0.60] | 0.62 [0.57, 0.67] | 0.51 [0.46, 0.56] |
| caa · L26 · first5 (non-native) | Position-Subjective / Explicit | ✗ | 0.40 [0.35, 0.45] | 0.49 [0.45, 0.53] | 0.51 [0.48, 0.53] | 0.64 [0.60, 0.68] | 0.57 [0.48, 0.65] | 0.55 [0.51, 0.58] | 0.74 [0.70, 0.77] | 0.58 [0.55, 0.61] | 0.57 [0.54, 0.61] | 0.61 [0.57, 0.64] | 0.64 [0.61, 0.68] | 0.51 [0.50, 0.53] | 0.54 [0.51, 0.57] | 0.69 [0.64, 0.73] | 0.53 [0.48, 0.58] |
| caa · L26 · response (non-native) | Position-Subjective / Explicit | ✗ | 0.46 [0.42, 0.51] | 0.51 [0.47, 0.56] | 0.48 [0.46, 0.51] | 0.57 [0.53, 0.61] | 0.74 [0.67, 0.81] | 0.53 [0.49, 0.57] | 0.67 [0.63, 0.71] | 0.52 [0.49, 0.55] | 0.64 [0.61, 0.67] | 0.68 [0.65, 0.71] | 0.60 [0.56, 0.63] | 0.54 [0.52, 0.56] | 0.55 [0.52, 0.58] | 0.71 [0.66, 0.75] | 0.55 [0.50, 0.60] |
| vennemeyer SyA · math plain last · L29 · last_prompt (non-native) | Position-Verifiable / Explicit | ✗ | 0.48 [0.43, 0.53] | 0.32 [0.29, 0.36] | 0.46 [0.44, 0.48] | 0.60 [0.56, 0.64] | 0.52 [0.44, 0.61] | 0.49 [0.46, 0.53] | 0.42 [0.37, 0.46] | 0.54 [0.50, 0.57] | 0.53 [0.50, 0.57] | 0.48 [0.45, 0.52] | 0.38 [0.34, 0.41] | 0.48 [0.46, 0.50] | 0.50 [0.47, 0.53] | 0.44 [0.39, 0.49] | 0.40 [0.35, 0.45] |
| vennemeyer SyA · math plain last · L29 · first5 (non-native) | Position-Verifiable / Explicit | ✗ | 0.48 [0.43, 0.53] | 0.44 [0.40, 0.48] | 0.54 [0.51, 0.56] | 0.30 [0.26, 0.34] | 0.35 [0.27, 0.42] | 0.41 [0.37, 0.44] | 0.41 [0.37, 0.46] | 0.43 [0.40, 0.46] | 0.47 [0.44, 0.50] | 0.57 [0.53, 0.60] | 0.53 [0.50, 0.56] | 0.55 [0.54, 0.57] | 0.62 [0.59, 0.65] | 0.50 [0.45, 0.55] | 0.61 [0.56, 0.66] |
| vennemeyer SyA · math plain last · L29 · response (non-native) | Position-Verifiable / Explicit | ✗ | 0.52 [0.47, 0.57] | 0.41 [0.37, 0.46] | 0.64 [0.62, 0.67] | 0.52 [0.48, 0.56] | 0.49 [0.40, 0.57] | 0.52 [0.49, 0.56] | 0.29 [0.25, 0.33] | 0.55 [0.52, 0.58] | 0.47 [0.44, 0.51] | 0.35 [0.32, 0.38] | 0.41 [0.37, 0.44] | 0.50 [0.48, 0.52] | 0.38 [0.35, 0.42] | 0.31 [0.27, 0.36] | 0.40 [0.36, 0.45] |
| vennemeyer GA · math plain last · L29 · last_prompt (non-native) | Position-Verifiable / Explicit | ✗ | 0.33 [0.28, 0.37] | 0.59 [0.55, 0.63] | 0.56 [0.54, 0.58] | 0.45 [0.40, 0.48] | 0.45 [0.36, 0.53] | 0.47 [0.43, 0.50] | 0.69 [0.65, 0.73] | 0.57 [0.54, 0.60] | 0.47 [0.43, 0.50] | 0.52 [0.49, 0.55] | 0.51 [0.48, 0.55] | 0.52 [0.50, 0.54] | 0.49 [0.46, 0.52] | 0.60 [0.55, 0.65] | 0.42 [0.37, 0.47] |
| vennemeyer GA · math plain last · L29 · first5 (non-native) | Position-Verifiable / Explicit | ✗ | 0.39 [0.34, 0.44] | 0.50 [0.46, 0.54] | 0.58 [0.56, 0.61] | 0.49 [0.45, 0.53] | 0.56 [0.47, 0.64] | 0.55 [0.51, 0.59] | 0.68 [0.63, 0.72] | 0.55 [0.52, 0.58] | 0.57 [0.54, 0.61] | 0.67 [0.63, 0.69] | 0.54 [0.51, 0.58] | 0.51 [0.49, 0.53] | 0.55 [0.52, 0.58] | 0.72 [0.67, 0.76] | 0.54 [0.49, 0.58] |
| vennemeyer GA · math plain last · L29 · response (non-native) | Position-Verifiable / Explicit | ✗ | 0.47 [0.42, 0.52] | 0.62 [0.58, 0.66] | 0.59 [0.57, 0.61] | 0.62 [0.58, 0.66] | 0.68 [0.60, 0.76] | 0.51 [0.47, 0.54] | 0.56 [0.51, 0.61] | 0.53 [0.50, 0.56] | 0.64 [0.61, 0.67] | 0.75 [0.72, 0.78] | 0.61 [0.58, 0.64] | 0.52 [0.50, 0.53] | 0.51 [0.48, 0.54] | 0.77 [0.73, 0.82] | 0.49 [0.44, 0.54] |
| vennemeyer SyPr · math plain last · L11 · last_prompt (non-native) | Person-Traits / Explicit | ✗ | 0.41 [0.37, 0.47] | 0.61 [0.57, 0.64] | 0.58 [0.55, 0.60] | 0.54 [0.50, 0.58] | 0.56 [0.47, 0.66] | 0.51 [0.47, 0.54] | 0.51 [0.46, 0.55] | 0.56 [0.53, 0.59] | 0.53 [0.49, 0.56] | 0.55 [0.52, 0.59] | 0.47 [0.43, 0.51] | 0.56 [0.55, 0.58] | 0.42 [0.39, 0.45] | 0.46 [0.41, 0.51] | 0.48 [0.43, 0.53] |
| vennemeyer SyPr · math plain last · L11 · first5 (non-native) | Person-Traits / Explicit | ✗ | 0.42 [0.37, 0.47] | 0.50 [0.47, 0.55] | 0.51 [0.49, 0.54] | 0.52 [0.48, 0.56] | 0.48 [0.39, 0.56] | 0.50 [0.46, 0.53] | 0.51 [0.47, 0.56] | 0.56 [0.53, 0.59] | 0.48 [0.45, 0.51] | 0.51 [0.48, 0.54] | 0.46 [0.42, 0.49] | 0.44 [0.42, 0.46] | 0.52 [0.49, 0.56] | 0.51 [0.46, 0.56] | 0.54 [0.49, 0.59] |
| vennemeyer SyPr · math plain last · L11 · response (non-native) | Person-Traits / Explicit | ✗ | 0.43 [0.38, 0.48] | 0.52 [0.48, 0.56] | 0.58 [0.55, 0.60] | 0.59 [0.55, 0.63] | 0.46 [0.37, 0.54] | 0.59 [0.55, 0.63] | 0.42 [0.38, 0.47] | 0.60 [0.57, 0.63] | 0.52 [0.48, 0.55] | 0.52 [0.49, 0.55] | 0.48 [0.45, 0.52] | 0.74 [0.72, 0.76] | 0.44 [0.41, 0.47] | 0.63 [0.58, 0.68] | 0.39 [0.35, 0.44] |
| vennemeyer SyA · math chat last · L30 · last_prompt (non-native) | Position-Verifiable / Explicit | ✗ | 0.55 [0.50, 0.60] | 0.42 [0.38, 0.46] | 0.43 [0.41, 0.46] | 0.48 [0.44, 0.52] | 0.41 [0.33, 0.50] | 0.49 [0.45, 0.52] | 0.51 [0.46, 0.56] | 0.51 [0.48, 0.54] | 0.56 [0.52, 0.59] | 0.66 [0.64, 0.70] | 0.43 [0.39, 0.46] | 0.59 [0.57, 0.61] | 0.66 [0.63, 0.69] | 0.68 [0.64, 0.73] | 0.46 [0.41, 0.51] |
| vennemeyer SyA · math chat last · L30 · first5 (non-native) | Position-Verifiable / Explicit | ✗ | 0.49 [0.44, 0.54] | 0.50 [0.46, 0.54] | 0.54 [0.52, 0.57] | 0.22 [0.19, 0.26] | 0.25 [0.18, 0.33] | 0.38 [0.35, 0.41] | 0.49 [0.44, 0.54] | 0.50 [0.47, 0.53] | 0.49 [0.46, 0.52] | 0.64 [0.61, 0.67] | 0.45 [0.42, 0.49] | 0.57 [0.55, 0.59] | 0.69 [0.66, 0.72] | 0.63 [0.58, 0.68] | 0.53 [0.48, 0.58] |
| vennemeyer SyA · math chat last · L30 · response (non-native) | Position-Verifiable / Explicit | ✗ | 0.54 [0.49, 0.59] | 0.52 [0.48, 0.56] | 0.66 [0.64, 0.69] | 0.50 [0.46, 0.54] | 0.47 [0.39, 0.56] | 0.52 [0.48, 0.56] | 0.37 [0.32, 0.41] | 0.56 [0.53, 0.60] | 0.53 [0.50, 0.56] | 0.45 [0.42, 0.48] | 0.45 [0.42, 0.48] | 0.61 [0.59, 0.63] | 0.45 [0.42, 0.48] | 0.50 [0.45, 0.55] | 0.42 [0.37, 0.47] |
| vennemeyer GA · math chat last · L30 · last_prompt (non-native) | Position-Verifiable / Explicit | ✗ | 0.33 [0.29, 0.38] | 0.55 [0.51, 0.59] | 0.48 [0.46, 0.51] | 0.48 [0.44, 0.52] | 0.42 [0.33, 0.50] | 0.48 [0.44, 0.52] | 0.64 [0.59, 0.68] | 0.55 [0.52, 0.58] | 0.48 [0.45, 0.52] | 0.56 [0.53, 0.59] | 0.51 [0.48, 0.55] | 0.54 [0.52, 0.56] | 0.53 [0.50, 0.57] | 0.59 [0.54, 0.64] | 0.46 [0.41, 0.51] |
| vennemeyer GA · math chat last · L30 · first5 (non-native) | Position-Verifiable / Explicit | ✗ | 0.44 [0.39, 0.49] | 0.49 [0.45, 0.53] | 0.58 [0.55, 0.60] | 0.38 [0.34, 0.42] | 0.40 [0.31, 0.48] | 0.49 [0.45, 0.53] | 0.68 [0.64, 0.72] | 0.53 [0.50, 0.56] | 0.56 [0.52, 0.59] | 0.65 [0.62, 0.68] | 0.53 [0.49, 0.56] | 0.59 [0.57, 0.61] | 0.65 [0.61, 0.67] | 0.71 [0.67, 0.76] | 0.54 [0.48, 0.58] |
| vennemeyer GA · math chat last · L30 · response (non-native) | Position-Verifiable / Explicit | ✗ | 0.51 [0.46, 0.56] | 0.62 [0.58, 0.66] | 0.59 [0.57, 0.61] | 0.58 [0.54, 0.62] | 0.69 [0.61, 0.77] | 0.52 [0.48, 0.56] | 0.56 [0.51, 0.60] | 0.54 [0.51, 0.57] | 0.62 [0.58, 0.65] | 0.74 [0.71, 0.77] | 0.60 [0.57, 0.64] | 0.57 [0.55, 0.58] | 0.54 [0.51, 0.58] | 0.77 [0.73, 0.81] | 0.53 [0.47, 0.58] |
| vennemeyer SyPr · math chat last · L16 · last_prompt (non-native) | Person-Traits / Explicit | ✗ | 0.62 [0.57, 0.67] | 0.56 [0.52, 0.60] | 0.67 [0.65, 0.70] | 0.42 [0.38, 0.46] | 0.45 [0.37, 0.54] | 0.55 [0.52, 0.59] | 0.58 [0.53, 0.62] | 0.49 [0.46, 0.52] | 0.40 [0.37, 0.43] | 0.33 [0.30, 0.35] | 0.42 [0.38, 0.45] | 0.73 [0.71, 0.74] | 0.45 [0.42, 0.48] | 0.44 [0.39, 0.49] | 0.44 [0.39, 0.49] |
| vennemeyer SyPr · math chat last · L16 · first5 (non-native) | Person-Traits / Explicit | ✗ | 0.57 [0.52, 0.62] | 0.55 [0.51, 0.59] | 0.59 [0.56, 0.61] | 0.28 [0.25, 0.32] | 0.34 [0.26, 0.43] | 0.51 [0.47, 0.54] | 0.68 [0.63, 0.72] | 0.46 [0.43, 0.49] | 0.48 [0.45, 0.51] | 0.47 [0.44, 0.51] | 0.53 [0.49, 0.56] | 0.62 [0.60, 0.64] | 0.56 [0.53, 0.60] | 0.63 [0.58, 0.68] | 0.54 [0.49, 0.59] |
| vennemeyer SyPr · math chat last · L16 · response (non-native) | Person-Traits / Explicit | ✗ | 0.53 [0.48, 0.58] | 0.60 [0.56, 0.64] | 0.65 [0.62, 0.67] | 0.55 [0.51, 0.59] | 0.58 [0.50, 0.66] | 0.51 [0.47, 0.54] | 0.70 [0.66, 0.74] | 0.41 [0.38, 0.44] | 0.58 [0.55, 0.62] | 0.62 [0.59, 0.65] | 0.52 [0.48, 0.56] | 0.70 [0.69, 0.72] | 0.66 [0.63, 0.69] | 0.72 [0.67, 0.76] | 0.60 [0.56, 0.65] |
| vennemeyer SyA · math plain respall · L29 · last_prompt (non-native) | Position-Verifiable / Explicit | ✗ | 0.59 [0.54, 0.64] | 0.42 [0.38, 0.46] | 0.46 [0.43, 0.48] | 0.61 [0.57, 0.65] | 0.61 [0.53, 0.69] | 0.49 [0.46, 0.53] | 0.32 [0.28, 0.36] | 0.50 [0.47, 0.53] | 0.60 [0.57, 0.63] | 0.56 [0.53, 0.59] | 0.48 [0.44, 0.51] | 0.54 [0.52, 0.55] | 0.51 [0.47, 0.54] | 0.44 [0.39, 0.49] | 0.50 [0.45, 0.55] |
| vennemeyer SyA · math plain respall · L29 · first5 (non-native) | Position-Verifiable / Explicit | ✗ | 0.57 [0.52, 0.62] | 0.56 [0.52, 0.60] | 0.54 [0.51, 0.56] | 0.41 [0.37, 0.45] | 0.50 [0.41, 0.59] | 0.44 [0.41, 0.48] | 0.38 [0.33, 0.42] | 0.43 [0.40, 0.46] | 0.51 [0.48, 0.54] | 0.64 [0.61, 0.68] | 0.56 [0.53, 0.59] | 0.54 [0.52, 0.56] | 0.55 [0.52, 0.59] | 0.52 [0.46, 0.56] | 0.63 [0.58, 0.68] |
| vennemeyer SyA · math plain respall · L29 · response (native) | Position-Verifiable / Explicit | ✓ | 0.60 [0.56, 0.65] | 0.44 [0.40, 0.49] | 0.60 [0.57, 0.62] | 0.54 [0.50, 0.58] | 0.42 [0.34, 0.50] | 0.51 [0.48, 0.55] | 0.29 [0.25, 0.32] | 0.55 [0.52, 0.58] | 0.54 [0.51, 0.58] | 0.44 [0.41, 0.47] | 0.51 [0.47, 0.54] | 0.38 [0.36, 0.40] | 0.42 [0.39, 0.45] | 0.35 [0.30, 0.39] | 0.45 [0.40, 0.50] |
| vennemeyer GA · math plain respall · L30 · last_prompt (non-native) | Position-Verifiable / Explicit | ✗ | 0.30 [0.26, 0.34] | 0.61 [0.57, 0.65] | 0.60 [0.58, 0.62] | 0.43 [0.39, 0.47] | 0.37 [0.29, 0.45] | 0.46 [0.42, 0.49] | 0.70 [0.66, 0.74] | 0.58 [0.55, 0.61] | 0.42 [0.39, 0.45] | 0.52 [0.49, 0.55] | 0.39 [0.36, 0.43] | 0.55 [0.53, 0.57] | 0.49 [0.46, 0.52] | 0.59 [0.54, 0.64] | 0.34 [0.30, 0.39] |
| vennemeyer GA · math plain respall · L30 · first5 (non-native) | Position-Verifiable / Explicit | ✗ | 0.38 [0.34, 0.43] | 0.51 [0.47, 0.55] | 0.60 [0.58, 0.62] | 0.49 [0.46, 0.54] | 0.47 [0.38, 0.56] | 0.53 [0.49, 0.56] | 0.71 [0.67, 0.75] | 0.54 [0.51, 0.57] | 0.54 [0.50, 0.57] | 0.59 [0.55, 0.62] | 0.46 [0.43, 0.50] | 0.56 [0.54, 0.58] | 0.58 [0.55, 0.61] | 0.67 [0.62, 0.72] | 0.45 [0.40, 0.50] |
| vennemeyer GA · math plain respall · L30 · response (native) | Position-Verifiable / Explicit | ✓ | 0.45 [0.41, 0.50] | 0.62 [0.58, 0.65] | 0.60 [0.58, 0.63] | 0.60 [0.56, 0.64] | 0.75 [0.67, 0.82] | 0.56 [0.52, 0.59] | 0.60 [0.55, 0.64] | 0.49 [0.45, 0.52] | 0.53 [0.49, 0.56] | 0.68 [0.65, 0.71] | 0.52 [0.49, 0.56] | 0.53 [0.51, 0.55] | 0.46 [0.42, 0.49] | 0.73 [0.68, 0.77] | 0.51 [0.46, 0.56] |
| vennemeyer SyPr · math plain respall · L18 · last_prompt (non-native) | Person-Traits / Explicit | ✗ | 0.57 [0.52, 0.62] | 0.65 [0.62, 0.69] | 0.67 [0.65, 0.69] | 0.32 [0.28, 0.36] | 0.42 [0.33, 0.50] | 0.52 [0.49, 0.56] | 0.58 [0.53, 0.63] | 0.40 [0.38, 0.44] | 0.48 [0.45, 0.52] | 0.50 [0.47, 0.53] | 0.48 [0.45, 0.51] | 0.68 [0.66, 0.70] | 0.52 [0.49, 0.56] | 0.60 [0.55, 0.64] | 0.48 [0.44, 0.53] |
| vennemeyer SyPr · math plain respall · L18 · first5 (non-native) | Person-Traits / Explicit | ✗ | 0.55 [0.50, 0.60] | 0.65 [0.61, 0.69] | 0.66 [0.64, 0.68] | 0.23 [0.20, 0.27] | 0.27 [0.20, 0.34] | 0.42 [0.38, 0.45] | 0.65 [0.60, 0.69] | 0.46 [0.42, 0.49] | 0.54 [0.51, 0.58] | 0.75 [0.73, 0.78] | 0.47 [0.44, 0.51] | 0.66 [0.64, 0.67] | 0.73 [0.70, 0.75] | 0.76 [0.72, 0.80] | 0.46 [0.41, 0.51] |
| vennemeyer SyPr · math plain respall · L18 · response (native) | Person-Traits / Explicit | ✓ | 0.47 [0.43, 0.52] | 0.63 [0.59, 0.66] | 0.61 [0.59, 0.63] | 0.48 [0.44, 0.52] | 0.53 [0.45, 0.62] | 0.51 [0.48, 0.55] | 0.47 [0.43, 0.52] | 0.51 [0.47, 0.54] | 0.58 [0.55, 0.61] | 0.72 [0.69, 0.75] | 0.50 [0.47, 0.53] | 0.68 [0.66, 0.70] | 0.55 [0.51, 0.58] | 0.75 [0.70, 0.79] | 0.43 [0.39, 0.48] |
| vennemeyer SyA · math chat respall · L30 · last_prompt (non-native) | Position-Verifiable / Explicit | ✗ | 0.64 [0.59, 0.69] | 0.53 [0.49, 0.57] | 0.43 [0.40, 0.45] | 0.67 [0.64, 0.71] | 0.60 [0.52, 0.69] | 0.48 [0.44, 0.51] | 0.30 [0.26, 0.34] | 0.51 [0.48, 0.54] | 0.60 [0.57, 0.63] | 0.54 [0.51, 0.57] | 0.44 [0.41, 0.47] | 0.56 [0.54, 0.57] | 0.55 [0.52, 0.59] | 0.43 [0.38, 0.47] | 0.50 [0.45, 0.54] |
| vennemeyer SyA · math chat respall · L30 · first5 (non-native) | Position-Verifiable / Explicit | ✗ | 0.61 [0.56, 0.66] | 0.58 [0.54, 0.62] | 0.51 [0.49, 0.54] | 0.27 [0.23, 0.30] | 0.32 [0.24, 0.40] | 0.38 [0.35, 0.41] | 0.27 [0.23, 0.31] | 0.47 [0.44, 0.50] | 0.48 [0.45, 0.51] | 0.49 [0.45, 0.52] | 0.44 [0.41, 0.48] | 0.55 [0.53, 0.57] | 0.62 [0.59, 0.65] | 0.39 [0.34, 0.44] | 0.54 [0.50, 0.59] |
| vennemeyer SyA · math chat respall · L30 · response (native) | Position-Verifiable / Explicit | ✓ | 0.60 [0.55, 0.64] | 0.48 [0.44, 0.53] | 0.56 [0.54, 0.58] | 0.44 [0.40, 0.47] | 0.33 [0.26, 0.41] | 0.47 [0.44, 0.51] | 0.26 [0.23, 0.30] | 0.54 [0.51, 0.57] | 0.48 [0.44, 0.51] | 0.36 [0.33, 0.39] | 0.43 [0.39, 0.47] | 0.47 [0.45, 0.49] | 0.48 [0.45, 0.52] | 0.31 [0.26, 0.35] | 0.43 [0.39, 0.48] |
| vennemeyer GA · math chat respall · L30 · last_prompt (non-native) | Position-Verifiable / Explicit | ✗ | 0.31 [0.27, 0.35] | 0.58 [0.54, 0.62] | 0.52 [0.49, 0.54] | 0.47 [0.43, 0.51] | 0.40 [0.32, 0.49] | 0.48 [0.44, 0.51] | 0.67 [0.63, 0.71] | 0.58 [0.55, 0.61] | 0.47 [0.43, 0.50] | 0.57 [0.53, 0.60] | 0.42 [0.39, 0.46] | 0.56 [0.54, 0.58] | 0.51 [0.48, 0.55] | 0.61 [0.56, 0.65] | 0.36 [0.31, 0.41] |
| vennemeyer GA · math chat respall · L30 · first5 (non-native) | Position-Verifiable / Explicit | ✗ | 0.40 [0.36, 0.45] | 0.51 [0.47, 0.55] | 0.58 [0.55, 0.60] | 0.46 [0.42, 0.51] | 0.42 [0.33, 0.51] | 0.51 [0.47, 0.54] | 0.71 [0.67, 0.75] | 0.54 [0.51, 0.57] | 0.54 [0.50, 0.57] | 0.61 [0.58, 0.64] | 0.47 [0.43, 0.51] | 0.59 [0.57, 0.61] | 0.62 [0.58, 0.64] | 0.68 [0.63, 0.72] | 0.44 [0.39, 0.49] |
| vennemeyer GA · math chat respall · L30 · response (native) | Position-Verifiable / Explicit | ✓ | 0.47 [0.42, 0.52] | 0.58 [0.54, 0.62] | 0.60 [0.58, 0.63] | 0.60 [0.56, 0.64] | 0.72 [0.64, 0.79] | 0.57 [0.53, 0.60] | 0.56 [0.52, 0.61] | 0.52 [0.49, 0.55] | 0.54 [0.51, 0.58] | 0.71 [0.68, 0.74] | 0.54 [0.50, 0.57] | 0.53 [0.51, 0.55] | 0.43 [0.40, 0.46] | 0.75 [0.70, 0.79] | 0.48 [0.43, 0.53] |
| vennemeyer SyPr · math chat respall · L16 · last_prompt (non-native) | Person-Traits / Explicit | ✗ | 0.64 [0.59, 0.69] | 0.66 [0.63, 0.70] | 0.71 [0.69, 0.74] | 0.35 [0.31, 0.39] | 0.38 [0.30, 0.47] | 0.49 [0.45, 0.52] | 0.66 [0.62, 0.71] | 0.44 [0.41, 0.47] | 0.50 [0.47, 0.54] | 0.63 [0.60, 0.66] | 0.58 [0.55, 0.62] | 0.73 [0.71, 0.75] | 0.60 [0.57, 0.63] | 0.65 [0.60, 0.70] | 0.55 [0.50, 0.60] |
| vennemeyer SyPr · math chat respall · L16 · first5 (non-native) | Person-Traits / Explicit | ✗ | 0.60 [0.55, 0.65] | 0.67 [0.63, 0.70] | 0.68 [0.66, 0.71] | 0.23 [0.20, 0.27] | 0.26 [0.19, 0.34] | 0.41 [0.37, 0.44] | 0.63 [0.59, 0.68] | 0.45 [0.41, 0.48] | 0.54 [0.50, 0.57] | 0.79 [0.76, 0.81] | 0.54 [0.50, 0.57] | 0.68 [0.66, 0.69] | 0.74 [0.71, 0.77] | 0.77 [0.72, 0.81] | 0.48 [0.44, 0.54] |
| vennemeyer SyPr · math chat respall · L16 · response (native) | Person-Traits / Explicit | ✓ | 0.54 [0.49, 0.59] | 0.65 [0.61, 0.69] | 0.65 [0.63, 0.67] | 0.52 [0.48, 0.56] | 0.53 [0.45, 0.62] | 0.51 [0.47, 0.54] | 0.52 [0.48, 0.57] | 0.58 [0.55, 0.61] | 0.65 [0.62, 0.68] | 0.75 [0.73, 0.78] | 0.54 [0.50, 0.57] | 0.68 [0.67, 0.70] | 0.59 [0.55, 0.62] | 0.74 [0.70, 0.78] | 0.43 [0.38, 0.48] |
| pandey residual · L27 · last_prompt (non-native) | Position-Verifiable / Explicit | ✗ | 0.64 [0.59, 0.68] | 0.35 [0.31, 0.39] | 0.45 [0.43, 0.48] | 0.55 [0.51, 0.59] | 0.63 [0.54, 0.71] | 0.53 [0.49, 0.57] | 0.30 [0.26, 0.34] | 0.52 [0.49, 0.55] | 0.51 [0.48, 0.55] | 0.49 [0.46, 0.52] | 0.55 [0.51, 0.58] | 0.58 [0.56, 0.59] | 0.47 [0.43, 0.50] | 0.41 [0.37, 0.47] | 0.55 [0.50, 0.60] |
| pandey residual · L27 · first5 (non-native) | Position-Verifiable / Explicit | ✗ | 0.59 [0.54, 0.64] | 0.43 [0.39, 0.48] | 0.41 [0.39, 0.44] | 0.66 [0.62, 0.69] | 0.68 [0.60, 0.76] | 0.54 [0.51, 0.58] | 0.28 [0.24, 0.32] | 0.53 [0.50, 0.56] | 0.45 [0.42, 0.48] | 0.41 [0.38, 0.45] | 0.43 [0.39, 0.46] | 0.51 [0.49, 0.53] | 0.31 [0.28, 0.34] | 0.33 [0.28, 0.38] | 0.49 [0.45, 0.54] |
| pandey residual · L27 · response (non-native) | Position-Verifiable / Explicit | ✗ | 0.48 [0.43, 0.53] | 0.41 [0.37, 0.45] | 0.40 [0.38, 0.42] | 0.48 [0.44, 0.52] | 0.35 [0.29, 0.44] | 0.50 [0.46, 0.54] | 0.31 [0.27, 0.35] | 0.56 [0.53, 0.59] | 0.44 [0.41, 0.48] | 0.37 [0.33, 0.40] | 0.44 [0.40, 0.48] | 0.50 [0.48, 0.52] | 0.48 [0.45, 0.52] | 0.31 [0.27, 0.36] | 0.42 [0.37, 0.47] |
| genadi best_head · heads L12 · answer_mean (≈native) | Position-Verifiable / Explicit | ≈ | 0.64 [0.59, 0.69] | 0.64 [0.60, 0.68] | 0.45 [0.42, 0.47] | 0.59 [0.55, 0.63] | 0.72 [0.63, 0.79] | 0.54 [0.50, 0.57] | 0.69 [0.65, 0.73] | 0.47 [0.44, 0.50] | 0.57 [0.54, 0.61] | 0.62 [0.59, 0.65] | 0.58 [0.55, 0.61] | 0.57 [0.56, 0.59] | 0.42 [0.38, 0.45] | 0.67 [0.62, 0.71] | 0.52 [0.47, 0.56] |
| genadi lr_top16 · heads in 7 layers · answer_mean (≈native) [ours] | Position-Verifiable / Explicit | ≈ | 0.62 [0.58, 0.67] | 0.66 [0.62, 0.70] | 0.49 [0.47, 0.52] | 0.46 [0.41, 0.50] | 0.62 [0.53, 0.70] | 0.52 [0.48, 0.55] | 0.69 [0.65, 0.73] | 0.65 [0.62, 0.68] | 0.54 [0.51, 0.57] | 0.55 [0.51, 0.58] | 0.50 [0.46, 0.53] | 0.65 [0.64, 0.67] | 0.42 [0.39, 0.45] | 0.51 [0.46, 0.56] | 0.36 [0.32, 0.40] |
| pandey lr_top15 · heads in 4 layers · last_prompt (non-native) [ours] | Position-Verifiable / Explicit | ✗ | 0.47 [0.42, 0.52] | 0.48 [0.44, 0.52] | 0.46 [0.44, 0.49] | 0.55 [0.51, 0.59] | 0.52 [0.44, 0.61] | 0.48 [0.44, 0.51] | 0.42 [0.37, 0.46] | 0.52 [0.49, 0.55] | 0.48 [0.45, 0.51] | 0.36 [0.33, 0.39] | 0.36 [0.33, 0.40] | 0.62 [0.60, 0.64] | 0.44 [0.40, 0.47] | 0.34 [0.30, 0.39] | 0.39 [0.34, 0.44] |
| Contrastive universal (P00) | General (baseline) | ✓ | 0.51 [0.46, 0.56] | 0.40 [0.36, 0.44] | 0.52 [0.50, 0.55] | 0.49 [0.45, 0.53] | 0.31 [0.24, 0.39] | 0.49 [0.45, 0.52] | 0.41 [0.36, 0.45] | 0.55 [0.52, 0.58] | 0.50 [0.47, 0.53] | 0.43 [0.39, 0.46] | 0.65 [0.62, 0.69] | 0.53 [0.52, 0.55] | 0.61 [0.58, 0.64] | 0.43 [0.38, 0.48] | 0.55 [0.50, 0.60] |
| Contrastive matched-cell pair | Position-Verifiable / Explicit | ✓ | 0.41 [0.36, 0.45] | 0.42 [0.38, 0.46] | 0.49 [0.47, 0.52] | 0.72 [0.68, 0.75] | 0.73 [0.65, 0.80] | 0.54 [0.51, 0.58] | 0.36 [0.32, 0.40] | 0.65 [0.62, 0.68] | 0.65 [0.61, 0.68] | 0.76 [0.73, 0.78] | 0.75 [0.72, 0.77] | 0.67 [0.65, 0.69] | 0.41 [0.37, 0.44] | 0.64 [0.59, 0.68] | 0.43 [0.38, 0.48] |
| Contrastive max over taxonomy pairs (post hoc, optimistic) | Person-Emotions / Explicit | ✓ | 0.56 [0.51, 0.61] | 0.65 [0.60, 0.69] | 0.59 [0.56, 0.61] | 0.72 [0.68, 0.75] | 0.74 [0.66, 0.81] | 0.62 [0.59, 0.66] | 0.67 [0.63, 0.71] | 0.68 [0.65, 0.71] | 0.78 [0.75, 0.80] | 0.85 [0.83, 0.88] | 0.88 [0.86, 0.90] | 0.75 [0.73, 0.76] | 0.66 [0.63, 0.69] | 0.82 [0.78, 0.86] | 0.68 [0.64, 0.73] |

Matched-cell and post-hoc reference pairs per benchmark: AYS freeform: P01 Position-Verifiable / Explicit / P07 Person-Emotions / Explicit; AYS multiple choice: P01 Position-Verifiable / Explicit / P02 Position-Verifiable / Implicit; TruthfulQA false answer: P01 Position-Verifiable / Explicit / P07 Person-Emotions / Explicit; AITA NTA flipped: P03 Position-Subjective / Explicit / P03 Position-Subjective / Explicit; AITA NTA original: P03 Position-Subjective / Explicit / P04 Position-Subjective / Implicit; AITA YTA framing: P04 Position-Subjective / Implicit / P06 Person-Traits / Implicit; OEQ framing: P04 Position-Subjective / Implicit / P06 Person-Traits / Implicit; SS framing: P04 Position-Subjective / Implicit / P06 Person-Traits / Implicit; AITA YTA validation: P05 Person-Traits / Explicit / P07 Person-Emotions / Explicit; OEQ validation: P05 Person-Traits / Explicit / P07 Person-Emotions / Explicit; SS validation: P05 Person-Traits / Explicit / P07 Person-Emotions / Explicit; SyPR praise: P05 Person-Traits / Explicit / P01 Position-Verifiable / Explicit; AITA YTA indirectness: P06 Person-Traits / Implicit / P08 Person-Emotions / Implicit; OEQ indirectness: P06 Person-Traits / Implicit / P08 Person-Emotions / Implicit; SS indirectness: P06 Person-Traits / Implicit / P07 Person-Emotions / Explicit

## 4. Controls and length confound

Controls: AUROC separating the two sides of each held-out control pair (0.5 = the detector ignores that correlate).

| Detector | Layer | Position | P09 Warranted praise | P10 Genuine agreement | P11 Appropriate emotional support | P12 Calibrated hedging | P13 Ordinary politeness |
| --- | --- | --- | --- | --- | --- | --- | --- |
| caa · L13 recipe · last_prompt (non-native) | 13 | last_prompt | 0.90 [0.86, 0.94] | 0.90 [0.86, 0.94] | 0.55 [0.48, 0.62] | 0.34 [0.27, 0.41] | 0.96 [0.93, 0.99] |
| caa · L13 recipe · first5 (non-native) | 13 | first5 | 0.85 [0.80, 0.89] | 0.58 [0.50, 0.65] | 0.66 [0.60, 0.73] | 0.76 [0.70, 0.82] | 0.84 [0.79, 0.89] |
| caa · L13 recipe · response (non-native) | 13 | response | 0.80 [0.74, 0.85] | 0.65 [0.58, 0.71] | 0.76 [0.70, 0.82] | 0.29 [0.22, 0.35] | 0.83 [0.78, 0.88] |
| caa · L26 · last_prompt (non-native) | 26 | last_prompt | 0.97 [0.95, 0.99] | 0.85 [0.80, 0.90] | 0.07 [0.04, 0.10] | 0.01 [0.00, 0.02] | 0.45 [0.38, 0.52] |
| caa · L26 · first5 (non-native) | 26 | first5 | 0.83 [0.79, 0.88] | 0.79 [0.73, 0.84] | 0.16 [0.11, 0.21] | 0.31 [0.25, 0.38] | 0.56 [0.49, 0.62] |
| caa · L26 · response (non-native) | 26 | response | 0.81 [0.76, 0.86] | 0.65 [0.58, 0.71] | 0.49 [0.42, 0.55] | 0.19 [0.14, 0.24] | 0.85 [0.80, 0.89] |
| vennemeyer SyA · math plain last · L29 · last_prompt (non-native) | 29 | last_prompt | 0.98 [0.96, 0.99] | 1.00 [1.00, 1.00] | 0.89 [0.84, 0.93] | 0.15 [0.11, 0.20] | 0.92 [0.88, 0.95] |
| vennemeyer SyA · math plain last · L29 · first5 (non-native) | 29 | first5 | 0.53 [0.46, 0.61] | 0.54 [0.46, 0.61] | 0.46 [0.39, 0.53] | 0.36 [0.29, 0.42] | 0.57 [0.50, 0.65] |
| vennemeyer SyA · math plain last · L29 · response (non-native) | 29 | response | 0.62 [0.55, 0.69] | 0.56 [0.49, 0.62] | 0.66 [0.60, 0.73] | 0.23 [0.17, 0.29] | 0.23 [0.17, 0.29] |
| vennemeyer GA · math plain last · L29 · last_prompt (non-native) | 29 | last_prompt | 0.94 [0.91, 0.97] | 0.94 [0.91, 0.97] | 0.20 [0.15, 0.26] | 0.19 [0.14, 0.25] | 0.90 [0.86, 0.94] |
| vennemeyer GA · math plain last · L29 · first5 (non-native) | 29 | first5 | 0.74 [0.68, 0.80] | 0.84 [0.79, 0.89] | 0.45 [0.38, 0.52] | 0.55 [0.48, 0.62] | 0.90 [0.87, 0.94] |
| vennemeyer GA · math plain last · L29 · response (non-native) | 29 | response | 0.83 [0.77, 0.87] | 0.64 [0.57, 0.71] | 0.88 [0.84, 0.92] | 0.15 [0.11, 0.20] | 0.97 [0.95, 0.99] |
| vennemeyer SyPr · math plain last · L11 · last_prompt (non-native) | 11 | last_prompt | 0.63 [0.56, 0.69] | 0.53 [0.45, 0.59] | 0.68 [0.61, 0.74] | 0.10 [0.06, 0.14] | 0.19 [0.14, 0.24] |
| vennemeyer SyPr · math plain last · L11 · first5 (non-native) | 11 | first5 | 0.37 [0.30, 0.44] | 0.26 [0.20, 0.33] | 0.45 [0.37, 0.52] | 0.53 [0.45, 0.60] | 0.52 [0.45, 0.59] |
| vennemeyer SyPr · math plain last · L11 · response (non-native) | 11 | response | 0.40 [0.33, 0.47] | 0.73 [0.66, 0.79] | 0.66 [0.59, 0.73] | 0.23 [0.17, 0.29] | 0.46 [0.39, 0.54] |
| vennemeyer SyA · math chat last · L30 · last_prompt (non-native) | 30 | last_prompt | 0.94 [0.91, 0.97] | 1.00 [0.99, 1.00] | 0.93 [0.89, 0.96] | 0.17 [0.12, 0.22] | 0.56 [0.49, 0.63] |
| vennemeyer SyA · math chat last · L30 · first5 (non-native) | 30 | first5 | 0.60 [0.52, 0.66] | 0.43 [0.36, 0.50] | 0.47 [0.40, 0.54] | 0.56 [0.49, 0.63] | 0.45 [0.38, 0.51] |
| vennemeyer SyA · math chat last · L30 · response (non-native) | 30 | response | 0.70 [0.63, 0.76] | 0.48 [0.41, 0.55] | 0.72 [0.66, 0.79] | 0.42 [0.36, 0.49] | 0.34 [0.28, 0.41] |
| vennemeyer GA · math chat last · L30 · last_prompt (non-native) | 30 | last_prompt | 0.96 [0.93, 0.98] | 0.98 [0.96, 0.99] | 0.16 [0.11, 0.21] | 0.31 [0.25, 0.38] | 0.77 [0.71, 0.83] |
| vennemeyer GA · math chat last · L30 · first5 (non-native) | 30 | first5 | 0.77 [0.71, 0.82] | 0.80 [0.75, 0.86] | 0.27 [0.21, 0.34] | 0.72 [0.65, 0.78] | 0.74 [0.68, 0.80] |
| vennemeyer GA · math chat last · L30 · response (non-native) | 30 | response | 0.88 [0.83, 0.92] | 0.59 [0.53, 0.66] | 0.86 [0.81, 0.90] | 0.29 [0.23, 0.36] | 0.95 [0.93, 0.98] |
| vennemeyer SyPr · math chat last · L16 · last_prompt (non-native) | 16 | last_prompt | 0.78 [0.72, 0.83] | 0.94 [0.91, 0.96] | 0.93 [0.89, 0.95] | 0.01 [0.00, 0.02] | 0.98 [0.96, 0.99] |
| vennemeyer SyPr · math chat last · L16 · first5 (non-native) | 16 | first5 | 0.75 [0.69, 0.81] | 0.60 [0.53, 0.67] | 0.39 [0.32, 0.46] | 0.05 [0.03, 0.07] | 0.28 [0.22, 0.34] |
| vennemeyer SyPr · math chat last · L16 · response (non-native) | 16 | response | 0.88 [0.83, 0.92] | 0.65 [0.58, 0.71] | 0.83 [0.78, 0.88] | 0.01 [0.00, 0.01] | 0.79 [0.74, 0.85] |
| vennemeyer SyA · math plain respall · L29 · last_prompt (non-native) | 29 | last_prompt | 0.99 [0.97, 1.00] | 1.00 [1.00, 1.00] | 1.00 [0.99, 1.00] | 0.87 [0.83, 0.91] | 1.00 [0.99, 1.00] |
| vennemeyer SyA · math plain respall · L29 · first5 (non-native) | 29 | first5 | 0.62 [0.55, 0.69] | 0.55 [0.47, 0.62] | 0.73 [0.67, 0.80] | 0.81 [0.75, 0.86] | 0.95 [0.92, 0.97] |
| vennemeyer SyA · math plain respall · L29 · response (native) | 29 | response | 0.77 [0.71, 0.83] | 0.58 [0.51, 0.65] | 0.91 [0.87, 0.94] | 0.58 [0.52, 0.65] | 0.69 [0.63, 0.76] |
| vennemeyer GA · math plain respall · L30 · last_prompt (non-native) | 30 | last_prompt | 0.94 [0.91, 0.97] | 0.93 [0.89, 0.96] | 0.09 [0.06, 0.13] | 0.10 [0.06, 0.14] | 0.36 [0.29, 0.43] |
| vennemeyer GA · math plain respall · L30 · first5 (non-native) | 30 | first5 | 0.79 [0.73, 0.84] | 0.88 [0.83, 0.93] | 0.22 [0.17, 0.27] | 0.27 [0.21, 0.33] | 0.48 [0.41, 0.55] |
| vennemeyer GA · math plain respall · L30 · response (native) | 30 | response | 0.89 [0.86, 0.93] | 0.69 [0.62, 0.75] | 0.75 [0.70, 0.81] | 0.11 [0.07, 0.14] | 0.87 [0.83, 0.91] |
| vennemeyer SyPr · math plain respall · L18 · last_prompt (non-native) | 18 | last_prompt | 0.99 [0.98, 1.00] | 0.83 [0.78, 0.87] | 0.33 [0.27, 0.40] | 0.15 [0.10, 0.20] | 0.99 [0.98, 1.00] |
| vennemeyer SyPr · math plain respall · L18 · first5 (non-native) | 18 | first5 | 0.83 [0.77, 0.88] | 0.55 [0.48, 0.62] | 0.43 [0.36, 0.51] | 0.44 [0.38, 0.52] | 0.87 [0.83, 0.91] |
| vennemeyer SyPr · math plain respall · L18 · response (native) | 18 | response | 0.91 [0.87, 0.94] | 0.53 [0.46, 0.60] | 0.96 [0.93, 0.98] | 0.19 [0.14, 0.25] | 0.86 [0.81, 0.90] |
| vennemeyer SyA · math chat respall · L30 · last_prompt (non-native) | 30 | last_prompt | 1.00 [0.99, 1.00] | 1.00 [1.00, 1.00] | 1.00 [1.00, 1.00] | 0.85 [0.80, 0.89] | 0.99 [0.98, 1.00] |
| vennemeyer SyA · math chat respall · L30 · first5 (non-native) | 30 | first5 | 0.64 [0.57, 0.70] | 0.57 [0.50, 0.64] | 0.70 [0.63, 0.77] | 0.77 [0.71, 0.82] | 0.69 [0.63, 0.75] |
| vennemeyer SyA · math chat respall · L30 · response (native) | 30 | response | 0.66 [0.59, 0.72] | 0.51 [0.44, 0.58] | 0.82 [0.77, 0.87] | 0.75 [0.70, 0.81] | 0.21 [0.16, 0.27] |
| vennemeyer GA · math chat respall · L30 · last_prompt (non-native) | 30 | last_prompt | 0.99 [0.98, 1.00] | 1.00 [0.99, 1.00] | 0.30 [0.24, 0.37] | 0.22 [0.17, 0.28] | 0.62 [0.55, 0.68] |
| vennemeyer GA · math chat respall · L30 · first5 (non-native) | 30 | first5 | 0.83 [0.78, 0.88] | 0.91 [0.86, 0.95] | 0.32 [0.25, 0.38] | 0.36 [0.29, 0.42] | 0.60 [0.53, 0.66] |
| vennemeyer GA · math chat respall · L30 · response (native) | 30 | response | 0.91 [0.88, 0.94] | 0.73 [0.67, 0.79] | 0.89 [0.84, 0.93] | 0.12 [0.08, 0.16] | 0.90 [0.86, 0.94] |
| vennemeyer SyPr · math chat respall · L16 · last_prompt (non-native) | 16 | last_prompt | 0.94 [0.90, 0.96] | 0.11 [0.08, 0.16] | 0.95 [0.92, 0.97] | 0.78 [0.72, 0.83] | 1.00 [1.00, 1.00] |
| vennemeyer SyPr · math chat respall · L16 · first5 (non-native) | 16 | first5 | 0.83 [0.77, 0.88] | 0.25 [0.18, 0.31] | 0.55 [0.47, 0.62] | 0.68 [0.61, 0.75] | 0.92 [0.88, 0.95] |
| vennemeyer SyPr · math chat respall · L16 · response (native) | 16 | response | 0.91 [0.87, 0.95] | 0.50 [0.43, 0.57] | 0.97 [0.95, 0.99] | 0.31 [0.24, 0.38] | 0.88 [0.83, 0.92] |
| pandey residual · L27 · last_prompt (non-native) | 27 | last_prompt | 0.25 [0.19, 0.31] | 0.12 [0.08, 0.17] | 1.00 [1.00, 1.00] | 0.99 [0.98, 0.99] | 1.00 [1.00, 1.00] |
| pandey residual · L27 · first5 (non-native) | 27 | first5 | 0.18 [0.13, 0.23] | 0.14 [0.09, 0.19] | 0.91 [0.86, 0.94] | 0.79 [0.72, 0.85] | 0.80 [0.74, 0.86] |
| pandey residual · L27 · response (non-native) | 27 | response | 0.12 [0.08, 0.16] | 0.39 [0.32, 0.46] | 0.52 [0.45, 0.59] | 0.76 [0.71, 0.82] | 0.18 [0.14, 0.24] |
| genadi best_head · heads L12 · answer_mean (≈native) | [12] | answer_mean | 0.54 [0.47, 0.61] | 0.72 [0.65, 0.78] | 0.81 [0.76, 0.86] | 0.30 [0.25, 0.37] | 0.80 [0.75, 0.85] |
| genadi lr_top16 · heads in 7 layers · answer_mean (≈native) [ours] | [9, 10, 11, 12, 14, 16, 31] | answer_mean | 0.39 [0.32, 0.46] | 0.80 [0.75, 0.85] | 0.40 [0.33, 0.47] | 0.47 [0.40, 0.54] | 0.41 [0.34, 0.48] |
| pandey lr_top15 · heads in 4 layers · last_prompt (non-native) [ours] | [28, 29, 30, 31] | last_prompt | 0.32 [0.25, 0.39] | 0.56 [0.49, 0.63] | 0.24 [0.19, 0.31] | 0.55 [0.48, 0.62] | 0.33 [0.27, 0.40] |
| REF P00 General (baseline) | 29 | response | 0.90 [0.87, 0.94] | 0.94 [0.92, 0.97] | 1.00 [1.00, 1.00] | 0.01 [0.00, 0.03] | 1.00 [1.00, 1.00] |
| REF P01 Position-Verifiable / Explicit | 26 | response | 0.66 [0.60, 0.73] | 0.75 [0.68, 0.81] | 1.00 [1.00, 1.00] | 0.00 [0.00, 0.00] | 1.00 [0.99, 1.00] |
| REF P02 Position-Verifiable / Implicit | 29 | response | 0.16 [0.11, 0.21] | 0.14 [0.10, 0.18] | 0.97 [0.96, 0.99] | 0.08 [0.05, 0.12] | 0.96 [0.93, 0.98] |
| REF P03 Position-Subjective / Explicit | 29 | response | 0.63 [0.56, 0.70] | 0.82 [0.76, 0.87] | 1.00 [1.00, 1.00] | 0.00 [0.00, 0.01] | 1.00 [1.00, 1.00] |
| REF P04 Position-Subjective / Implicit | 29 | response | 0.30 [0.24, 0.36] | 0.27 [0.22, 0.33] | 0.91 [0.87, 0.94] | 0.11 [0.07, 0.15] | 0.92 [0.88, 0.96] |
| REF P05 Person-Traits / Explicit | 29 | response | 0.99 [0.97, 1.00] | 0.77 [0.72, 0.83] | 1.00 [1.00, 1.00] | 0.02 [0.01, 0.03] | 1.00 [1.00, 1.00] |
| REF P06 Person-Traits / Implicit | 29 | response | 0.27 [0.20, 0.33] | 0.52 [0.45, 0.59] | 0.88 [0.84, 0.92] | 0.01 [0.00, 0.02] | 0.20 [0.14, 0.25] |
| REF P07 Person-Emotions / Explicit | 29 | response | 0.98 [0.96, 1.00] | 0.76 [0.70, 0.81] | 1.00 [1.00, 1.00] | 0.23 [0.17, 0.29] | 1.00 [1.00, 1.00] |
| REF P08 Person-Emotions / Implicit | 29 | response | 0.99 [0.98, 0.99] | 0.70 [0.64, 0.76] | 1.00 [1.00, 1.00] | 0.30 [0.23, 0.36] | 1.00 [1.00, 1.00] |

Length confound, summarised per detector over the 15 benchmarks (full rows in `length_confound.csv`): Spearman ρ between detector score and n_response_tokens, and the spread of AUROC across length terciles.

| Detector / role | median ρ(score, length) | max |ρ| (benchmark) | median tercile AUROC range | max tercile range (benchmark) |
| --- | --- | --- | --- | --- |
| caa · L13 recipe · last_prompt (non-native) | 0.20 | 0.67 (OEQ framing) | 0.10 | 0.31 (OEQ validation) |
| caa · L13 recipe · first5 (non-native) | 0.18 | 0.65 (OEQ indirectness) | 0.14 | 0.45 (OEQ validation) |
| caa · L13 recipe · response (non-native) | 0.22 | 0.73 (OEQ indirectness) | 0.12 | 0.49 (OEQ indirectness) |
| caa · L26 · last_prompt (non-native) | 0.24 | 0.59 (OEQ framing) | 0.11 | 0.34 (OEQ validation) |
| caa · L26 · first5 (non-native) | 0.26 | 0.66 (OEQ indirectness) | 0.13 | 0.50 (OEQ validation) |
| caa · L26 · response (non-native) | 0.42 | 0.68 (OEQ indirectness) | 0.15 | 0.44 (OEQ indirectness) |
| vennemeyer SyA · math plain last · L29 · last_prompt (non-native) | -0.20 | -0.36 (SS validation) | 0.09 | 0.26 (OEQ validation) |
| vennemeyer SyA · math plain last · L29 · first5 (non-native) | -0.02 | -0.28 (OEQ indirectness) | 0.11 | 0.33 (SyPR praise) |
| vennemeyer SyA · math plain last · L29 · response (non-native) | -0.39 | -0.76 (OEQ indirectness) | 0.13 | 0.34 (SyPR praise) |
| vennemeyer GA · math plain last · L29 · last_prompt (non-native) | 0.12 | 0.48 (OEQ validation) | 0.13 | 0.32 (OEQ validation) |
| vennemeyer GA · math plain last · L29 · first5 (non-native) | 0.13 | 0.54 (OEQ indirectness) | 0.12 | 0.45 (OEQ validation) |
| vennemeyer GA · math plain last · L29 · response (non-native) | 0.07 | 0.57 (TruthfulQA false answer) | 0.14 | 0.36 (OEQ framing) |
| vennemeyer SyPr · math plain last · L11 · last_prompt (non-native) | 0.09 | 0.50 (SyPR praise) | 0.07 | 0.33 (OEQ framing) |
| vennemeyer SyPr · math plain last · L11 · first5 (non-native) | 0.06 | 0.30 (SyPR praise) | 0.11 | 0.19 (AITA NTA original) |
| vennemeyer SyPr · math plain last · L11 · response (non-native) | -0.18 | 0.63 (AYS freeform) | 0.13 | 0.34 (SS indirectness) |
| vennemeyer SyA · math chat last · L30 · last_prompt (non-native) | 0.06 | -0.26 (SS validation) | 0.09 | 0.26 (OEQ framing) |
| vennemeyer SyA · math chat last · L30 · first5 (non-native) | 0.04 | 0.45 (AITA NTA original) | 0.10 | 0.27 (SyPR praise) |
| vennemeyer SyA · math chat last · L30 · response (non-native) | -0.19 | -0.57 (SS validation) | 0.12 | 0.29 (AITA NTA original) |
| vennemeyer GA · math chat last · L30 · last_prompt (non-native) | 0.14 | 0.44 (OEQ validation) | 0.13 | 0.31 (OEQ validation) |
| vennemeyer GA · math chat last · L30 · first5 (non-native) | 0.26 | 0.57 (OEQ indirectness) | 0.14 | 0.44 (OEQ validation) |
| vennemeyer GA · math chat last · L30 · response (non-native) | 0.19 | 0.67 (TruthfulQA false answer) | 0.17 | 0.38 (OEQ framing) |
| vennemeyer SyPr · math chat last · L16 · last_prompt (non-native) | 0.12 | 0.47 (TruthfulQA false answer) | 0.09 | 0.23 (OEQ indirectness) |
| vennemeyer SyPr · math chat last · L16 · first5 (non-native) | 0.22 | 0.54 (OEQ indirectness) | 0.14 | 0.50 (OEQ indirectness) |
| vennemeyer SyPr · math chat last · L16 · response (non-native) | 0.48 | 0.80 (SS indirectness) | 0.10 | 0.54 (OEQ indirectness) |
| vennemeyer SyA · math plain respall · L29 · last_prompt (non-native) | -0.03 | -0.49 (OEQ indirectness) | 0.09 | 0.42 (OEQ validation) |
| vennemeyer SyA · math plain respall · L29 · first5 (non-native) | -0.03 | -0.40 (OEQ indirectness) | 0.10 | 0.23 (SyPR praise) |
| vennemeyer SyA · math plain respall · L29 · response (native) | -0.35 | -0.77 (OEQ indirectness) | 0.18 | 0.33 (OEQ indirectness) |
| vennemeyer GA · math plain respall · L30 · last_prompt (non-native) | 0.04 | 0.51 (OEQ indirectness) | 0.11 | 0.41 (OEQ validation) |
| vennemeyer GA · math plain respall · L30 · first5 (non-native) | 0.19 | 0.61 (OEQ indirectness) | 0.14 | 0.52 (OEQ validation) |
| vennemeyer GA · math plain respall · L30 · response (native) | 0.24 | 0.63 (TruthfulQA false answer) | 0.22 | 0.37 (OEQ indirectness) |
| vennemeyer SyPr · math plain respall · L18 · last_prompt (non-native) | 0.30 | 0.50 (SS validation) | 0.08 | 0.34 (SS indirectness) |
| vennemeyer SyPr · math plain respall · L18 · first5 (non-native) | 0.38 | 0.50 (OEQ indirectness) | 0.12 | 0.33 (SS indirectness) |
| vennemeyer SyPr · math plain respall · L18 · response (native) | -0.00 | -0.40 (SS validation) | 0.08 | 0.25 (OEQ framing) |
| vennemeyer SyA · math chat respall · L30 · last_prompt (non-native) | -0.08 | -0.52 (SS validation) | 0.11 | 0.39 (OEQ validation) |
| vennemeyer SyA · math chat respall · L30 · first5 (non-native) | -0.00 | -0.64 (OEQ indirectness) | 0.14 | 0.45 (OEQ validation) |
| vennemeyer SyA · math chat respall · L30 · response (native) | -0.37 | -0.81 (OEQ indirectness) | 0.17 | 0.46 (OEQ indirectness) |
| vennemeyer GA · math chat respall · L30 · last_prompt (non-native) | 0.06 | 0.48 (OEQ framing) | 0.10 | 0.36 (OEQ validation) |
| vennemeyer GA · math chat respall · L30 · first5 (non-native) | 0.18 | 0.60 (OEQ indirectness) | 0.17 | 0.49 (OEQ validation) |
| vennemeyer GA · math chat respall · L30 · response (native) | 0.08 | 0.46 (TruthfulQA false answer) | 0.17 | 0.36 (OEQ framing) |
| vennemeyer SyPr · math chat respall · L16 · last_prompt (non-native) | 0.38 | 0.57 (TruthfulQA false answer) | 0.09 | 0.25 (OEQ indirectness) |
| vennemeyer SyPr · math chat respall · L16 · first5 (non-native) | 0.39 | 0.48 (OEQ indirectness) | 0.11 | 0.34 (AITA NTA original) |
| vennemeyer SyPr · math chat respall · L16 · response (native) | 0.12 | -0.44 (SS validation) | 0.09 | 0.22 (SS indirectness) |
| pandey residual · L27 · last_prompt (non-native) | -0.19 | -0.52 (OEQ framing) | 0.17 | 0.49 (OEQ validation) |
| pandey residual · L27 · first5 (non-native) | -0.41 | -0.65 (OEQ indirectness) | 0.13 | 0.47 (OEQ validation) |
| pandey residual · L27 · response (non-native) | -0.45 | -0.77 (SS validation) | 0.14 | 0.55 (OEQ indirectness) |
| genadi best_head · heads L12 · answer_mean (≈native) | 0.01 | 0.64 (OEQ indirectness) | 0.13 | 0.41 (OEQ validation) |
| genadi lr_top16 · heads in 7 layers · answer_mean (≈native) [ours] | -0.08 | -0.57 (TruthfulQA false answer) | 0.08 | 0.42 (SS indirectness) |
| pandey lr_top15 · heads in 4 layers · last_prompt (non-native) [ours] | -0.07 | 0.36 (SyPR praise) | 0.09 | 0.37 (SyPR praise) |
| REF P00 General (baseline) (universal) | -0.01 | 0.58 (AYS multiple choice) | 0.09 | 0.33 (SS indirectness) |
| REF P03 Position-Subjective / Explicit (matched cell/max taxonomy posthoc) | -0.09 | -0.09 (AITA NTA flipped) | 0.03 | 0.03 (AITA NTA flipped) |
| REF P03 Position-Subjective / Explicit (matched cell) | -0.30 | -0.30 (AITA NTA original) | 0.10 | 0.10 (AITA NTA original) |
| REF P04 Position-Subjective / Implicit (matched cell) | -0.42 | -0.76 (SS framing) | 0.14 | 0.14 (SS framing) |
| REF P06 Person-Traits / Implicit (matched cell) | -0.01 | 0.44 (OEQ indirectness) | 0.33 | 0.39 (SS indirectness) |
| REF P05 Person-Traits / Explicit (matched cell) | 0.43 | 0.59 (OEQ validation) | 0.17 | 0.33 (OEQ validation) |
| REF P01 Position-Verifiable / Explicit (matched cell) | 0.27 | 0.37 (AYS multiple choice) | 0.15 | 0.17 (AYS freeform) |

ρ(label, length) per benchmark (same for every detector):

| Benchmark | n | ρ(label, length) |
| --- | --- | --- |
| AITA NTA flipped | 758 | -0.08 |
| AITA NTA original | 168 | -0.22 |
| AITA YTA framing | 988 | -0.16 |
| AITA YTA indirectness | 1248 | 0.45 |
| AITA YTA validation | 1196 | 0.13 |
| AYS freeform | 518 | -0.15 |
| AYS multiple choice | 742 | -0.13 |
| OEQ framing | 606 | 0.35 |
| OEQ indirectness | 504 | 0.43 |
| OEQ validation | 1270 | 0.34 |
| SS framing | 1314 | -0.22 |
| SS indirectness | 550 | 0.31 |
| SS validation | 1062 | 0.17 |
| SyPR praise | 3792 | 0.23 |
| TruthfulQA false answer | 2090 | 0.19 |

## 5. Design choices, assumptions and inferences

- *our choice* (environment): Benchmark and TEFO activation caches downloaded from SycoScope/activations and SycoScope/sycoscope-v2-artifacts (scripts/download_activations.py) instead of regenerated; all 15 benchmark files have the SHA-256 recorded in the committed eval metas (user approved).
- *our choice* (environment): pytest run with `uv run --with matplotlib pytest` before and after: matplotlib (imported by probe_analysis/layer_position.py via tests/test_layer_position.py) is not declared in pyproject.toml; pyproject was not changed.
- *from code* (layer mapping): This repo: block L = hidden_states[L + 1]. Audit extraction hooks decoder block outputs; identical for L < 31, block 31 is before the final norm (benchmark caches at L31 are after it). Test: tests/test_audit_layers.py.
- *from code* (layer mapping): CAA: BlockOutputWrapper output[0] = block L (same index).
- *from code* (layer mapping): Pandey: TransformerLens blocks.L.hook_resid_post with from_pretrained_no_processing = block L (same index).
- *from code* (layer mapping): Persona Vectors: hidden_states[l] = block l - 1; paper's 'layer 16' (1-indexed) = block 15.
- *from code* (layer mapping): Vennemeyer: their L k (k = 1..31) = block k; L32 repeats block 31 (_hidden_to_block_index returns values below 32 unchanged). Their Fig. 8b shows identical L31 and L32 values, consistent with this.
- *our choice* (chat template): CAA: Llama-3.1 chat template via utils.models.render_prompt (the repo's Llama-2 [INST] format does not apply); assistant turn is '(A)' or '(B)' with no end-of-turn token.
- *from code* (chat template): Vennemeyer native: plain 'Human: ... \n\nAssistant: ...' text with BOS (tokenizer default).
- *our choice* (chat template): Vennemeyer 'chat' variant: user turn = prompt without 'Human: ', assistant text = response without 'Assistant: ', rendered with the Llama-3.1 template; same rows, split and pooling rule.
- *from code* (chat template): Genadi: apply_chat_template(add_generation_prompt=False) then tokenizer(text) adds a second BOS; reproduced.
- *from code* (chat template): Pandey: apply_chat_template(add_generation_prompt=True) then to_tokens(prepend_bos=True) adds a second BOS; reproduced.
- *our choice* (position): CAA native position: the answer-letter token of the appended '(A)'/'(B)', located by character offset; Llama-3 tokenizes '(A' as one token, which is also CAA's position -2 (test).
- *our choice* (position): Vennemeyer native position: last token string containing an alphanumeric character (the code's no-EOS branch). The literal code on Llama-3.1 pools BOS (special id at position 0) and then masks it, giving zero vectors. Paper says EOS token.
- *from code* (position): Genadi primary pooling: mean of o_proj inputs from after the last <|end_header_id|> to before the final <|eot_id|> (extension/); it includes the '\n\n' header token. Secondary: last token (probe/), which is <|eot_id|>.
- *from code* (position): Pandey position (user decision: code-faithful only): (tokens != pad).sum() - 1 with pad = <|eot_id|>, which is the 'assistant' header token (n - 3), not the final '\n\n'.
- *our choice* (position): Benchmark scoring uses only the repo's cached positions (last_prompt, first5, response mean). No cached position equals the native position of CAA, Vennemeyer or Pandey; all are marked non-native. Genadi heads are scored on the answer mean (marked approximate: benchmark spans exclude the '\n\n' header token). Pandey heads are scored at last_prompt (non-native).
- *our choice* (threshold): DIM thresholds: midpoint of projected class means on the method's own fit rows (utils.probes.fit_dim rule). AUROC does not depend on it; balanced accuracy on benchmarks is reported by evaluate_probes but not used.
- *our choice* (layer choice): CAA: layer by max val AUROC on a 10% val group split of the A/B items (groups = question; test 20%, seed 0); recipe layer 13 (paper's steering layer for Llama-2-7B, same depth index) also scored.
- *our choice* (layer choice): Vennemeyer: their split (stratified 80/20, seed 42, test balanced on syc); we carve 10% of their train (stratified, seed 0) as val and pick the layer by max val AUROC per unit. Their code picks the steering layer on test AUROC.
- *from code* (layer choice): Pandey residual: layer int(0.85 * 32) = 27 (probe_transfer.py); direction from the first 100 of the 200 sycophancy pairs; pairs 100-199 held out for our parity AUROC.
- *from code* (layer choice): Genadi: best head by best-over-epochs validation accuracy on their random 75/25 row split (train questions only); the saved probe is the last-epoch weights.
- *from code* (genadi): Probe recipe: nn.Linear(128, 1), BCEWithLogits, Adam lr 1e-3, 25 epochs, batch 64 (extension/train_probe.py), seed 3407; all 1024 head probes trained at once with one shared shuffle per epoch (independence test in tests/test_audit_heads.py).
- *our choice* (genadi): Training data: the 'combined' set of all four disagreement types on the 612 train questions (their split test_size 0.25, seed 3407); the 205 held-out questions, built the same way, are our parity test set.
- *from paper* (genadi): Paper Table 8 (App. H) lists lr 1e-5, batch 32, 16 epochs; the code uses lr 1e-3, batch 64, 25 epochs. We follow the code.
- *from code* (genadi): Their val split is over dialogue rows, not questions, so val rows share questions with train rows.
- *our choice* (our addition): Genadi LR-top16: standardised logistic regression (utils.probes.fit_logistic, C = 1, not tuned) on the concatenated 128-d inputs of the 16 heads with the best val accuracy, fit on the fit rows only.
- *our choice* (our addition): Pandey LR-top15: the same logistic regression on the 15 top-ranked heads' 128-d o_proj inputs (not their 4096-d W_O projections), fit on pairs 0-99.
- *from code* (pandey): Head ranking: ||mean_pos - mean_neg|| of z_h @ W_O[:, h].T over the first 50 sycophancy pairs (DEFAULT_N_PROMPTS), top 15 (DEFAULT_TOP_K); computed as W_O applied to the mean difference of z_h, which is equal by linearity. W_O read from the safetensors in float32.
- *from paper* (pandey): Paper Sec. 3.1 says N = 200 pairs per task for directions; probe_transfer.py and App. U use the first 100. We follow the code.
- *from code* (persona): Rollouts: 20 extract questions x 5 instruction pairs x pos/neg x 10, temperature 1.0, top_p 1, max 1000 new tokens; system prompt 'You are a sycophantic assistant. <pos>' or 'You are a helpful assistant. <neg>'.
- *our choice* (persona): Generated with HF generate via utils.models.generate (their code uses vLLM with min_tokens=1); responses are stripped by utils.models.generate.
- *our choice* (persona): Judge held per the API rule: judge() raises NotImplementedError; exact templates in data/audit/persona/judge_prompts.json and filled requests in judge_requests.jsonl. No unfiltered vector is built or scored.
- *our choice* (evaluation): 95% CIs: stratified bootstrap (positives and negatives resampled separately), 1,000 resamples, seed 0. Length terciles: np.quantile cut points at 1/3 and 2/3 of n_response_tokens per benchmark.
- *our choice* (evaluation): Benchmark taxonomy cells from the docs/SPEC.md table; TruthfulQA false answer (Position-Verifiable / Explicit) and AITA NTA original (Position-Subjective / Explicit) are not in that table and are our inference.
- *our choice* (evaluation): References (user decision): universal pair 00; matched-cell pair (cell -> P01, P03, P04, P05, P06); and the post-hoc maximum over the 8 taxonomy pairs, labelled optimistic. Each pair's probe is the one in validation_selected.csv (val AUROC ties at 1.0, broken by val accuracy then probe_id as in plot_probe_transfer.py).
- *from code* (data overlap): Genadi source dialogues use TruthfulQA questions; the benchmarks TruthfulQA false answer and AYS (which includes TruthfulQA) may share questions with them. Not removed.
- *from code* (data overlap): CAA's A/B items come from the Perez et al. sycophancy dataset, which is also the source of this repo's contrastive training prompts.
- *our choice* (position): Vennemeyer (user decision after a parity gap): both last-content-token and whole-response-mean (resp_all) directions are built on math_factorial and scored, each in plain and chat variants; resp_all is marked as matching the cached 'response' position.
- *from paper* (paper vs code): Vennemeyer: the paper pools at the end-of-sequence token (Sec. 4, App. C.2); the code's default 'last' pooling on Llama-3.1 selects BOS; neither reconstructed pooling reproduces all of Fig. 8b.
- *our choice* (controls): Control rows: the downloaded TEFO cache has only the sweep layers (3, 6, ..., 29), so the 1,280 held-out control rows (5 pairs x 128 test prompts x 2) were filtered to their own JSONL and re-extracted with the unchanged get_activations at all 32 layers (max length 4096). At shared layers the regenerated vectors differ from the downloaded cache at bf16 level (max relative difference 0.4-1.9%); detectors and references are all scored on the regenerated file.

## 6. Regression evidence

- pytest before any change (main at 64970ce): 57 passed (run as uv run --with matplotlib pytest; matplotlib is not declared in pyproject.toml).
- pytest at the end: 76 passed = 57 + 19 new tests in tests/test_audit_*.py.
- evaluate_probes regression: synthetic_tefo_C_sweep copied without eval/ to a temp dir; the unchanged evaluate_probes re-run on oeq_framing_minority_balanced_seed0_activations.npz; all 7,938 rows (2,646 probes x 3 labels) identical to the committed eval/*.jsonl in auroc, balanced_accuracy, n, n_pos, n_neg, exclusions and eval_sha256; max |dAUROC| = 0.0.
- Comparison used: SHA-256. The benchmark caches were downloaded from SycoScope/activations (user approved) and all 15 match the SHA-256 recorded in the committed eval metas, so no numerical AUROC tolerance was needed.
- Every audit AUROC in coverage.csv was re-computed from scores and asserted equal (atol 1e-9) to the eval JSONL written by the unchanged evaluate_probes (residual detectors and references) or audit/evaluate_heads.py (head detectors).
- git diff main -- probe_training judging scripts utils pyproject.toml uv.lock: empty; utils.activations.POSITIONS is (last_prompt, first5, response).

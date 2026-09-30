# Audit of existing sycophancy detectors (branch `audit/existing-methods`)

Separate track from Step 1, authorised by the user. Model: `meta-llama/Llama-3.1-8B-Instruct` only. Code in `audit/`,
tests in `tests/test_audit_*.py`, results in `reports/audit_existing_methods/`.

## External code

All five pins resolved; no fallback to HEAD was needed. Clones live in `external/<name>/` (git-ignored). No external
package is imported; the pieces below were ported line by line.

| Name | Repository | Pin (resolved) |
| --- | --- | --- |
| caa | github.com/nrimsky/CAA | 5dabbbd9a0bca5f25e174501e959de378806aa48 |
| vennemeyer | github.com/cincynlp/disentangle-sycophancy | 217b89808d07e978a7c8fac37e8ff363e29ad8ef |
| genadi | github.com/rifoagenadi/sycophancy | 09d51c4474d9a43be0d22bebae37c9fadd38b699 |
| pandey | github.com/MVPandey/shared-sycophancy-lying-circuit | c55356cf31e20f2cd5e1ff74dd0013929b49c96d |
| persona | github.com/safety-research/persona_vectors | b8e0f044fe2410a6fad579f38324f03f13b4e917 |

### Ported pieces

| Ported to | Source file and lines | What |
| --- | --- | --- |
| `audit/sources.py::caa_rows` | caa `generate_vectors.py` 70-127 (esp. 102, 114); `utils/tokenize.py` 13-27 | A/B pairs, activation at position -2 of "... (A)", vector = mean(pos - neg) |
| `audit/sources.py::vennemeyer_split` | vennemeyer `src/diffmean_analysis.py` 882-922 | stratified 80/20 split (seed 42) on all label fields, test balanced on the primary label |
| `audit/sources.py::last_content_index` | vennemeyer `src/diffmean_analysis.py` 1252-1291 (no-EOS branch, 1279-1290) | 'last' pooling: last token string containing an alphanumeric character |
| `audit/layers.py::from_vennemeyer` | vennemeyer `src/diffmean_analysis.py` 1010-1012; `src/utils/model_utils.py` 30-63 | layer index mapping |
| `audit/fit_directions.py` (DIM) | vennemeyer `src/diffmean_analysis.py` 1375-1416 (sums, means, w_raw), 1824-1825 (score h . w) | w = mu_pos - mu_neg; AUROC of h . w |
| `audit/sources.py::genadi_*` | genadi `extension/extract_activations.py` 27-44, 111-121, 129-160, 252-310, 330-343 | templates, simulated dialogues, answer slice, o_proj-input hook |
| `audit/fit_heads.py::train_head_probes` | genadi `extension/train_probe.py` 14-58, 61-78, 127-131 | Linear(128,1), BCE, Adam, 75/25 row split, best val accuracy over epochs |
| last-token pooling variant | genadi `probe/extract_activation.py` 58-63 | `states[0][0][:, -1, :]` |
| `audit/sources.py::triviaqa_pairs` | pandey `src/shared_circuits/data/triviaqa.py` 9-51 | TriviaQA pair construction (offset 37) |
| `audit/sources.py::pandey_rows` | pandey `src/shared_circuits/prompts/sycophancy.py` 6-30; `prompts/chat.py` | belief template, chat template with generation prompt |
| `audit/sources.py::pandey_faithful_index` | pandey `src/shared_circuits/extraction/extractor.py` 74, 108-110 | last-token index `(tokens != pad).sum() - 1` |
| `audit/fit_heads.py` (pandey) | pandey `src/shared_circuits/attribution/dla.py` 10-99; `analyses/circuit_overlap.py` 46-56; `config.py` 19-20 | per-head W_O-projected mean difference norm, top-15, first 50 prompts |
| Pandey residual layer and n | pandey `src/shared_circuits/analyses/probe_transfer.py` 20-21, 90-108 | layer int(0.85 L) = 27, first 100 pairs |
| `audit/sources.py::persona_prompts` | persona `eval/eval_persona.py` 137-158, 240-256 | system prompts, 10 rollouts per question, temperature 1.0 |
| `audit/generate_persona.py::persona_row` | persona `generate_vec.py` 14-35 | response mean after len(encode(prompt)) |
| `audit/persona_judge.py` | persona `eval/prompts.py` 3-21; `judge.py` 39-103; `generate_vec.py` 40-43 | judge prompts, call pattern, filter rule (not run) |

## Layer indices

This repo reads block `L` from `hidden_states[L + 1]` (the last one includes the final norm). Audit extraction hooks
decoder block outputs, which equal `hidden_states[L + 1]` for L < 31; block 31 is taken before the final norm.
Tests: `tests/test_audit_layers.py`.

- CAA: block output (`BlockOutputWrapper`, `output[0]`) -> same index.
- Pandey: TransformerLens `blocks.L.hook_resid_post` with `from_pretrained_no_processing` -> same index.
- Persona Vectors: `hidden_states[l]` -> block `l - 1`. The paper's "layer 16" (1-indexed, output of the 16th layer)
  is block 15 here.
- Vennemeyer: `--all_layers` loops `1..32` and `_hidden_to_block_index` returns any value below 32 unchanged, so
  their "L k" is block k for k = 1..31 and "L32" repeats block 31. Block 0 is never computed. The code comment calls
  these hidden-state indices; the mapping is to block indices.

## Findings while porting (code as run on Llama-3.1-8B-Instruct)

- Vennemeyer 'last' pooling: the loop stops at the first special token id. The Llama-3.1 tokenizer adds BOS
  (id 128000, in `all_special_ids`) at position 0, so the pooled index is 0; that token is then masked as special
  and the pooled vector is all zeros. The literal code gives zero directions on this model. We use the evident intent
  (and the no-EOS branch): the last token containing an alphanumeric character. The paper (Sec. 4, App. C.2) says
  the end-of-sequence token.
- Pandey last-token index: TransformerLens 3.0.0 sets `pad_token = eos_token` (`<|eot_id|>` for this model), and
  `(tokens != pad).sum() - 1` drops the two `<|eot_id|>` tokens inside the templated prompt, so the index is the
  `assistant` header token (n - 3), not the final `\n\n`. `to_tokens(prepend_bos=True)` on a templated string also
  gives two BOS tokens. Per the user's decision, we reproduce this code-faithful position only.
- Genadi: `tokenizer(text)` on an already-templated string gives two BOS tokens (reproduced). The answer slice starts
  right after `<|end_header_id|>`, so it includes the `\n\n` header token. The 75/25 probe split is over dialogue
  rows, not questions, so the same question appears in train and val.
- Genadi paper vs code: Table 8 (App. H) lists Adam lr 1e-5, batch 32, 16 epochs; `extension/train_probe.py` uses
  lr 1e-3, batch 64, 25 epochs (`probe/train.py`: batch 25). We follow the code (as the task specifies).
- Pandey paper vs code: Sec. 3.1 says directions use N = 200 pairs; `probe_transfer.py` uses the first 100
  (App. U agrees with the code); head ranking uses the first 50 (`DEFAULT_N_PROMPTS`).
- Vennemeyer paper numbers quoted in the task (">0.97 at mid layers", "SyPr by layer 8") are for Qwen3-30B
  (Fig. 1). Llama-3.1-8B curves are only in Fig. 8b (App. C.4, SIMPLE MATH), without numbers.
- Persona Vectors "layer 16 for Llama": App. (p. 30) and Fig. 13, chosen by steering at each layer and taking the
  highest trait expression; the paper counts layers from 1. Verified.

## Vennemeyer parity and pooling (user decision)

On SIMPLE MATH (math_factorial, plain text) neither reconstructed pooling reproduces all three Fig. 8b curves:
last-content-token pooling has SyA and GA within about 0.05 up to L20 but 0.05-0.09 higher late, and SyPr 0.1-0.2
lower at L2-L10; whole-response mean (their `resp_all`, the Gemma default in `diffmean_analysis.py`) has SyPr within
about 0.04 at L2-L28 but SyA and GA 0.05-0.11 lower at L20-L28. The diagnostic source is `data/audit/vennemeyer_diag/`
(same rows and splits, extra `resp_all` span; the re-extracted last-content scores equal the main run exactly).
The user chose to score both poolings on the benchmarks, each labelled, for the plain and chat variants
(`probes/Llama-3.1-8B-Instruct/audit_vennemeyer_math_{plain,chat}_{last,respall}`). Layers are chosen on source val.

# Activation cache

Span-mean activations of judged responses, for checking interpretability methods (persona vectors, probes) against judge labels. Code: `interp/cache_activations.py`. Hub copy: private dataset repo `SycoScope/activation-cache`.

## Positions

n is the number of response tokens. Prompt tokens are everything the chat template renders before the response.

| Name | Tokens averaged | Same as |
|---|---|---|
| `prompt_last` | the last prompt token; on Llama 3.1 the `\n\n` after `<\|start_header_id\|>assistant<\|end_header_id\|>` | persona-vector `prompt_last`; `get_activations` `last_prompt` |
| `prompt_mean` | all prompt tokens, including BOS and the default system block | persona-vector `prompt_avg` |
| `response_mean` | all response tokens | persona-vector `response_avg`; `get_activations` `response` |
| `response_first5`, `response_first10` | the first min(k, n) response tokens | `get_activations` `first5` |
| `response_frac10`, `_frac25`, `_frac50`, `_frac75` | the first ceil(p × n / 100) response tokens, computed in integers | |

For Are You Sure rows the prompt is the whole conversation before the turn-2 reply (question, turn-1 answer, pushback), so `prompt_last` is the `\n\n` after the second assistant header, `prompt_mean` includes turn 1, and the response positions are the turn-2 reply.

`prompt_mean` includes BOS, whose activations are much larger than other tokens' in middle layers, and Llama 3.1's default system block ("Cutting Knowledge Date: December 2023 / Today Date: 26 Jul 2024"), which the template adds when there is no system message.

Other papers' "last prompt token" is sometimes the `<|eot_id|>` closing the user turn or the last user-content token. These are not cached; adding one is a new entry in `POSITIONS` and `cache_spans`.

## Layers

Every block is stored. Block L is `hidden_states[L + 1]`; the last block is after the model's final norm. The embedding output is not stored. The persona-vector code numbers layers by `hidden_states` index, so its layer L is our block L − 1 (its layer 16 is our block 15).

## Precision

The forward pass is bfloat16, span means are taken in float32, and values are stored in float16.

- Rounding an existing float32 Llama-3.1 cache (17,920 rows, blocks 3 to 29) to float16 changed single-token vectors by nothing (bfloat16 values in this range are exact in float16) and means by about 2e-4 relative L2 (bfloat16 storage: 1.6e-3). Difference-of-means directions kept cosine ≥ 0.99999988 with the float32 ones. The largest value was 28; float16 holds up to 65,504.
- The persona-vector reproduction found cosine ≈ 0.9999 per layer between vectors from float32 and bfloat16 forward passes, about 1,000 times the float16 storage effect.
- Blocks 0 to 2, block 31 and `prompt_mean` (which includes BOS) were not in that test. The script stops if a value is outside the float16 range, and the parity log prints the largest value per position.

## Layout

```
<cache>/<model>/<input path under generations/<model>/, no .jsonl>/<position>/part-NNNNN.safetensors
<cache>/<model>/<input path ...>/<position>/meta/part-NNNNN.safetensors.meta.json
```

Each part holds float16 arrays `L00` … `L31` of shape (rows, hidden). Its meta file lists the row ids in order, a SHA-256 of each row's rendered text (prompt and response), each row's token span, the position definition, the model and its Hub revision, the file's SHA-256, the input file's SHA-256, git state and command line.

Each meta file also has two per-row lists from the same forward pass, used to measure how far the responses are from what this model would write (see the last section):

- `next_token_agreement`: share of response tokens that are the model's most likely next token given the tokens before them. Greedy text from the same weights and prompt is close to 1.
- `response_nll`: mean −log p of the response tokens, in nats.

A row cached in several positions in one run has the same values in each; a position added in a later run records that run's values.

Labels are not cached. Join them by `id` with `generations/<model>/judging/judged/*_judged.jsonl`.

## Running and extending

- Full run on a GPU box: `GEN_BATCH_SIZE=... CACHE_BATCH_SIZE=... bash interp/run_cache_vast.sh`. It runs the tests, regenerates the Are You Sure responses and pushes them to git, checks parity on 64 SS and 64 Are You Sure MC rows, caches the five ELEPHANT inputs and both Are You Sure inputs with `--max-length 8192`, and uploads the cache.
- A rerun computes only the (row, position) pairs that are missing: new rows in an input, or a position not cached before. Each added position costs one forward pass over the rows; storage is 32 × 4096 × 2 bytes = 256 KiB per row per position (3.4 GB per position for the 12,835 ELEPHANT rows).
- A rerun stops if a cached row's text changed or is no longer in the input, or if the model revision or a position definition differs from the cached one. Delete that position's folder to recompute it.
- To extend on a new machine, download only the meta files first; the script reads nothing else from the cache:
  `hf download SycoScope/activation-cache --repo-type dataset --include "*/meta/*" --local-dir activations/cache`
- To download one position for analysis:
  `hf download SycoScope/activation-cache --repo-type dataset --include "Llama-3.1-8B-Instruct/responses/*/prompt_last/*" --local-dir activations/cache`
- To load: `ids, acts = load(Path("activations/cache/Llama-3.1-8B-Instruct/responses/oeq_responses/prompt_last"), layers=[15])` returns row ids and `{block: (rows, hidden) float32}`, after checking each file's SHA-256 against its meta file.

## Generator differs from the activation model (ELEPHANT only)

The Are You Sure responses are generated on the same machine with the same checkpoint, chat template and bfloat16 weights that the cache uses (greedy, batched with left padding), so their `next_token_agreement` should be close to 1. They show what agreement looks like when the generator and the activation model are the same; any shortfall comes from batching numerics and from decoding and re-tokenizing the text.

The Llama-3.1-8B ELEPHANT responses were generated through OpenRouter (`meta-llama/llama-3.1-8b-instruct`). The generation metadata records `do_sample: false` and `temperature: null`, with at most 512 new tokens for AITA and SS and 1024 for OEQ. It does not record the serving provider, which OpenRouter chooses per request and which may serve quantized weights, or whether a temperature of 0 was sent. If none was sent, OpenRouter's default temperature applied and the responses are sampled, not greedy. Every kept row finished with `stop`.

Activations are computed by running the Hugging Face checkpoint (revision recorded per part) in bfloat16 on the prompt and response as one sequence, with the prompt built by the Hugging Face chat template, including its default system block. If the provider built the prompt differently or served different weights, the cached activations are this model's states on that text, not the states of the model that wrote it. `next_token_agreement` and `response_nll` measure this per row: agreement near 1 means the text is what this model and prompt produce greedily; lower agreement means sampling, a different prompt or different weights.

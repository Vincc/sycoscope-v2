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

- `next_token_agreement`: share of response tokens that are the model's most likely next token given the tokens before them. Greedy text from the same weights and prompt is close to 1; sampled text is lower by design.
- `response_nll`: mean −log p of the response tokens, in nats.

A row cached in several positions in one run has the same values in each; a position added in a later run records that run's values.

Labels are not cached. Join them by `id` with `generations/<model>/<decoding>/judging/judged/*_judged.jsonl`.

## Running and extending

- Responses live in git (LFS) and activations on the Hub. `bash judging/run_generate_openrouter.sh` samples the five ELEPHANT inputs and both Are You Sure inputs through OpenRouter and commits and pushes them. Then, on a GPU box with the branch pulled, `CACHE_BATCH_SIZE=... bash interp/run_cache_vast.sh` runs the tests, checks parity on 64 sampled SS and 64 sampled Are You Sure MC rows, caches the seven sampled inputs with `--max-length 8192`, and uploads the cache. The GPU box generates nothing.
- A rerun computes only the (row, position) pairs that are missing: new rows in an input, or a position not cached before. Each added position costs one forward pass over the rows; storage is 32 × 4096 × 2 bytes = 256 KiB per row per position (3.4 GB per position for the 12,835 ELEPHANT rows).
- A rerun stops if a cached row's text changed or is no longer in the input, or if the model revision or a position definition differs from the cached one. Delete that position's folder to recompute it.
- To extend on a new machine, download only the meta files first; the script reads nothing else from the cache:
  `hf download SycoScope/activation-cache --repo-type dataset --include "*/meta/*" --local-dir activations/cache`
- To download one position for analysis:
  `hf download SycoScope/activation-cache --repo-type dataset --include "Llama-3.1-8B-Instruct/t0.6_p0.9/responses/*/prompt_last/*" --local-dir activations/cache`
- To load: `ids, acts = load(Path("activations/cache/Llama-3.1-8B-Instruct/t0.6_p0.9/responses/OEQ_s0_responses/prompt_last"), layers=[15])` returns row ids and `{block: (rows, hidden) float32}`, after checking each file's SHA-256 against its meta file.

## Generators

Responses sit in one folder per decoding: `generations/Llama-3.1-8B-Instruct/greedy/` holds everything generated greedily before October 2026 (responses and judged files), and `generations/Llama-3.1-8B-Instruct/t0.6_p0.9/` the sampled set. The activation cache mirrors these paths. `training/` (probe-training generations) stays at the model level.

**Sampled** (`t0.6_p0.9/responses/*_s0_responses.jsonl`, ELEPHANT and Are You Sure): the main set, generated by `judging/generate_responses.py` through OpenRouter at temperature 0.6 and top-p 0.9 with top-k off (Meta's reference sampler and Llama 3.1's shipped defaults; no repetition penalty), pinned to CoreWeave with bfloat16 weights and no fallbacks. Each request sends the prompt rendered by the Hugging Face chat template, default system block included, to the completions endpoint without its BOS (the provider adds one), and fails unless the provider counted the same number of prompt tokens. So the text comes from the prompt the cache uses, on weights of the same precision. CoreWeave's chat endpoint would instead apply its own template, which drops the default system block (23 prompt tokens instead of 47 in a probe). Each row records these settings under `generation`.

Seeds: the run seed is 0 (`_s0` in file names, `__s0` in row ids, `prompt_id` keeps the id without it). Each request gets its own seed, a hash of the run seed, the row id and the turn, stored in the row as `request_seeds`:

| Input | Requests per row | Seeds |
|---|---|---|
| AITA-YTA, OEQ, SS, AITA-NTA-OG | 1 | one per row |
| AITA-NTA-FLIP | 1 per side | the two sides of a pair have different row ids, so different seeds |
| Are You Sure (MC and free-form) | 2 (turn 1, then turn 2 after the pushback) | one per turn |

A provider seeds a fresh sampler for each request, so sending the run seed itself with every request would give every row the same random draws: rows would not be independent, and similar prompts such as the two sides of a flip pair would come out more alike than the model makes them. Rerunning with run seed 0 sends the same request seeds. More samples per prompt are further run seeds (`_s1`, ...), whose request seeds differ because the row ids do.

Greedy decoding was dropped because it got stuck in repetition loops, more often after the pushback than before it and more often when turn 1 was correct (math: 66% of correct turn-1 rows against 41% of incorrect ones), so dropping truncated rows biased the Are You Sure rate. A pilot on 100 math rows at temperature 0.6 still hit the 2,048-token cap in a turn on 58 rows (greedy: 75).

**Greedy Are You Sure** (`greedy/responses/are_you_sure_*_responses.jsonl`): generated on a GPU box with Hugging Face `generate` (greedy, bfloat16, batched with left padding; commit afecb24), the same checkpoint and template the cache uses. Their `next_token_agreement` should be close to 1, which checks that a generator and the cache see the same model and prompt; any shortfall comes from batching numerics and from decoding and re-tokenizing the text. 28% of rows hit the token cap in at least one turn, nearly all in repetition loops. `greedy/responses/ays_*_responses.jsonl` are the old repo's greedy runs, imported.

**OpenRouter ELEPHANT** (`greedy/responses/aita_yta_responses.jsonl` and the other lowercase names): generated by the old repo through OpenRouter's chat endpoint (`meta-llama/llama-3.1-8b-instruct`), which sent temperature 0, with at most 512 new tokens for AITA and SS and 1024 for OEQ. The serving provider was not pinned or recorded; OpenRouter chooses it per request and it may serve quantized weights and apply its own chat template. Every kept row finished with `stop`.

Activations are computed by running the Hugging Face checkpoint (revision recorded per part) in bfloat16 on the prompt and response as one sequence, with the prompt built by the Hugging Face chat template, including its default system block. If the provider built the prompt differently or served different weights, the cached activations are this model's states on that text, not the states of the model that wrote it. `next_token_agreement` and `response_nll` measure this per row: agreement near 1 means the text is what this model and prompt produce greedily; lower agreement means sampling, a different prompt or different weights.

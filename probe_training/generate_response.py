"""Generate responses to user prompts, with no system prompt or under each contrastive system-prompt pair.

Input rows have prompt_id, source, user_prompt. Output rows add id, messages and response, the
get_activations input. Input must be under data/training/<dataset>/. Output is written to
generations/<model name>/training/<dataset>/<input stem>/<input stem>_responses.jsonl, or
<input stem>_contrastive_responses.jsonl with --system-prompts: one row per (pair, polarity, prompt).

With --openrouter-model, generations come from the OpenRouter chat API (key in OPENROUTER_API_KEY) instead of
a local model. --model is still the Hugging Face name: it names the output directory that get_activations reads.

Each finished row is appended to <output>.partial.jsonl. A rerun with the same arguments skips rows already there;
other arguments raise. The final file is written in input order and the partial files are removed.

Run from the repo root: python -m probe_training.generate_response --model ... --input ...
"""
import argparse
import json
import os
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from utils.contrastive import LABEL, POLARITIES, TAG
from utils.io import REPO_ROOT, plain, read_json, read_jsonl, write_json, write_jsonl
from utils.models import generate, load_model_and_tokenizer, model_spec, render_prompt


def work_items(prompts: list[dict], pairs: list[dict] | None) -> list[dict]:
    """One item per generation: the output row without its response."""
    if pairs is None:
        return [
            {"id": p["prompt_id"], **p, "messages": [{"role": "user", "content": p["user_prompt"]}]}
            for p in prompts
        ]
    items = []
    for k, pair in enumerate(pairs):
        for polarity in POLARITIES:
            for p in prompts:
                items.append(
                    {
                        "id": f"pair{k:02d}__{TAG[polarity]}__{p['prompt_id']}",
                        **p,
                        "pair_index": k,
                        "cell": pair["cell"],
                        "pair_type": pair["type"],
                        "polarity": polarity,
                        "label": LABEL[polarity],
                        "messages": [
                            {"role": "system", "content": pair[polarity]},
                            {"role": "user", "content": p["user_prompt"]},
                        ],
                    }
                )
    return items


OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"
RETRY_STATUSES = (429, 500, 502, 503, 504)
MAX_ATTEMPTS = 5


def openrouter_generate(messages: list[dict], args, api_key: str) -> dict:
    """One chat completion. Returns response, truncated and the provider that served it."""
    body = {
        "model": args.openrouter_model,
        "messages": messages,
        "max_tokens": args.max_new_tokens,
        "temperature": args.temperature,
        "top_p": args.top_p,
        "seed": args.seed,
    }
    if args.provider:
        body["provider"] = {"order": [args.provider], "allow_fallbacks": False}
    request = urllib.request.Request(
        OPENROUTER_URL,
        data=json.dumps(body).encode("utf-8"),
        headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
    )
    for attempt in range(1, MAX_ATTEMPTS + 1):
        try:
            with urllib.request.urlopen(request, timeout=300) as r:
                out = json.loads(r.read().decode("utf-8"))
            break
        except urllib.error.HTTPError as e:
            if e.code not in RETRY_STATUSES or attempt == MAX_ATTEMPTS:
                raise RuntimeError(f"OpenRouter HTTP {e.code}: {e.read().decode('utf-8', 'replace')}") from e
            time.sleep(2**attempt)
    if "error" in out:
        raise RuntimeError(f"OpenRouter error: {out['error']}")
    choice = out["choices"][0]
    finish = choice["finish_reason"]
    if finish not in ("stop", "length"):
        raise RuntimeError(f"OpenRouter finish_reason {finish!r}: {choice}")
    content = choice["message"]["content"]
    if content is None:
        raise RuntimeError(f"OpenRouter returned no content: {choice}")
    return {
        "response": content.strip(),
        "truncated": finish == "length",
        "openrouter_model": out["model"],
        "openrouter_provider": out["provider"],
    }


def resume(partial_path: Path, settings_path: Path, settings: dict, items: list[dict]) -> dict[str, dict]:
    """Rows already in the partial file, by id. Each must match its work item; the run settings must match."""
    if not partial_path.exists():
        write_json(settings_path, settings)
        return {}
    if read_json(settings_path) != settings:
        raise ValueError(f"{partial_path} was made with other arguments ({settings_path}); delete both to restart")
    by_id = {item["id"]: item for item in items}
    done = {}
    for row in read_jsonl(partial_path):
        item = by_id.get(row["id"])
        if item is None or {k: row[k] for k in item} != item or row["id"] in done:
            raise ValueError(f"{partial_path}: row {row['id']} is unknown, differs from its input or is duplicated")
        done[row["id"]] = row
    return done


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--system-prompts", type=Path, help="Contrastive pairs JSON; omit for no system prompt.")
    parser.add_argument("--max-new-tokens", type=int, required=True)
    parser.add_argument("--temperature", type=float, required=True, help="0 means greedy.")
    parser.add_argument("--top-p", type=float, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--batch-size", type=int, help="Local generation only.")
    parser.add_argument("--openrouter-model", help="OpenRouter model id; generate through the API instead of locally.")
    parser.add_argument("--provider", help="OpenRouter only: pin this provider, no fallbacks.")
    parser.add_argument("--max-workers", type=int, help="OpenRouter only: concurrent requests.")
    args = parser.parse_args()

    if args.openrouter_model:
        if args.max_workers is None or args.batch_size is not None:
            raise ValueError("--openrouter-model needs --max-workers and takes no --batch-size")
    elif args.batch_size is None or args.max_workers is not None or args.provider:
        raise ValueError("local generation needs --batch-size and takes no --max-workers or --provider")

    prompts = read_jsonl(args.input)
    pairs = read_json(args.system_prompts) if args.system_prompts else None
    items = work_items(prompts, pairs)
    suffix = "_contrastive_responses" if pairs else "_responses"
    rel = args.input.resolve().relative_to(REPO_ROOT / "data" / "training")  # raises if outside data/training
    if len(rel.parts) < 2:
        raise ValueError(f"{args.input} is not under data/training/<dataset>/")
    out_dir = REPO_ROOT / "generations" / args.model.split("/")[-1] / "training" / rel.parts[0] / args.input.stem
    out_path = out_dir / f"{args.input.stem}{suffix}.jsonl"
    if out_path.exists():
        raise FileExistsError(f"{out_path} exists")
    partial_path = out_path.with_name(out_path.stem + ".partial.jsonl")
    settings_path = out_path.with_name(out_path.stem + ".partial.settings.json")

    settings = {k: plain(v) for k, v in vars(args).items() if k != "max_workers"}  # concurrency does not change rows
    done = resume(partial_path, settings_path, settings, items)
    todo = [item for item in items if item["id"] not in done]
    print(f"{len(items)} items: {len(done)} already in {partial_path.name}, {len(todo)} to generate", flush=True)

    with open(partial_path, "a", encoding="utf-8", newline="\n") as partial:
        def save(row):
            partial.write(json.dumps(row, ensure_ascii=False) + "\n")
            partial.flush()
            done[row["id"]] = row

        if args.openrouter_model and todo:
            api_key = os.environ["OPENROUTER_API_KEY"]
            with ThreadPoolExecutor(max_workers=args.max_workers) as pool:
                futures = {pool.submit(openrouter_generate, item["messages"], args, api_key): item for item in todo}
                try:
                    for n, future in enumerate(as_completed(futures), 1):
                        save({**futures[future], **future.result(), "model": args.model})
                        if n % 50 == 0:
                            print(f"{len(done)}/{len(items)}", flush=True)
                finally:
                    for future in futures:
                        future.cancel()  # on error, drop queued requests instead of waiting for them
        elif todo:
            import torch

            model, tokenizer = load_model_and_tokenizer(args.model, padding_side="left")
            torch.manual_seed(args.seed)
            for b in range(0, len(todo), args.batch_size):
                chunk = todo[b : b + args.batch_size]
                rendered = [render_prompt(tokenizer, item["messages"], model_spec(args.model).get("template_kwargs"))
                            for item in chunk]
                responses, truncated = generate(model, tokenizer, rendered, args.max_new_tokens, args.temperature, args.top_p)
                for item, response, trunc in zip(chunk, responses, truncated):
                    save({**item, "response": response, "truncated": trunc, "model": args.model})
                print(f"{len(done)}/{len(items)}", flush=True)

    rows = [done[item["id"]] for item in items]
    assert len(rows) == len(items) == len(done)
    write_jsonl(out_path, rows)
    partial_path.unlink()
    settings_path.unlink()
    print(f"Wrote {len(rows)} rows to {out_path}")


if __name__ == "__main__":
    main()

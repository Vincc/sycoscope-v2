"""Generate responses to user prompts, with no system prompt or under each contrastive system-prompt pair.

Input rows have prompt_id, source, user_prompt. Output rows add id, messages and response, the
get_activations input. Output is written to generations/<model name>/training/<input stem>_responses.jsonl,
or <input stem>_contrastive_responses.jsonl with --system-prompts: one row per (pair, polarity, prompt).

Run from the repo root: python -m probe_training.generate_response --model ... --input ...
"""
import argparse
from pathlib import Path

from utils.contrastive import LABEL, POLARITIES, TAG
from utils.io import REPO_ROOT, read_json, read_jsonl, write_jsonl
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


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--system-prompts", type=Path, help="Contrastive pairs JSON; omit for no system prompt.")
    parser.add_argument("--max-new-tokens", type=int, required=True)
    parser.add_argument("--temperature", type=float, required=True, help="0 means greedy.")
    parser.add_argument("--top-p", type=float, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--batch-size", type=int, required=True)
    args = parser.parse_args()

    import torch

    prompts = read_jsonl(args.input)
    pairs = read_json(args.system_prompts) if args.system_prompts else None
    items = work_items(prompts, pairs)
    suffix = "_contrastive_responses" if pairs else "_responses"
    out_path = REPO_ROOT / "generations" / args.model.split("/")[-1] / "training" / f"{args.input.stem}{suffix}.jsonl"

    model, tokenizer = load_model_and_tokenizer(args.model, padding_side="left")
    torch.manual_seed(args.seed)

    rows = []
    for b in range(0, len(items), args.batch_size):
        chunk = items[b : b + args.batch_size]
        rendered = [render_prompt(tokenizer, item["messages"], model_spec(args.model).get("template_kwargs")) for item in chunk]
        responses, truncated = generate(model, tokenizer, rendered, args.max_new_tokens, args.temperature, args.top_p)
        for item, response, trunc in zip(chunk, responses, truncated):
            rows.append({**item, "response": response, "truncated": trunc, "model": args.model})
        print(f"{len(rows)}/{len(items)}", flush=True)

    write_jsonl(out_path, rows)
    print(f"Wrote {len(rows)} rows to {out_path}")


if __name__ == "__main__":
    main()

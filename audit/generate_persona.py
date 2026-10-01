"""Persona Vectors extraction rollouts: sample responses to each pos/neg system prompt x question, on the GPU.

Sampling as eval_persona.main with n_per_question > 1: temperature 1.0, top_p 1, max 1000 new tokens.
Writes data/audit/persona/extract_rollouts.jsonl in the extract_source row format (pool response_mean), with split
'judge_pending': the judge filter that selects rows for the vector is held (see audit/persona_judge.py).

Run from the repo root: python -m audit.generate_persona --prompts data/audit/persona/extract_prompts.jsonl --batch-size 100 --seed 0
"""
import argparse
from pathlib import Path

from utils.io import REPO_ROOT, check_counts, read_jsonl, write_jsonl, write_meta
from utils.models import generate, load_model_and_tokenizer, render_prompt

MODEL = "meta-llama/Llama-3.1-8B-Instruct"


def persona_row(prompt_row: dict, response: str, truncated: bool, tokenizer) -> dict:
    """Their get_hidden_p_and_r: text = prompt + answer, response_mean over tokens after len(encode(prompt))."""
    prompt = render_prompt(tokenizer, prompt_row["messages"])
    text = prompt + response
    ids = tokenizer(text, add_special_tokens=False)["input_ids"]
    prompt_len = len(tokenizer(prompt, add_special_tokens=False)["input_ids"])
    if ids[:prompt_len] != tokenizer(prompt, add_special_tokens=False)["input_ids"] or prompt_len >= len(ids):
        raise ValueError(f"{prompt_row['id']}: response merges with the prompt or is empty")
    labels = {"persona": 1 if prompt_row["polarity"] == "pos" else 0} if "polarity" in prompt_row else {"system_index": prompt_row["system_index"]}
    return {**prompt_row, "response": response, "truncated": truncated, "text": text,
            "add_special_tokens": False, "labels": labels, "group": f"q{prompt_row['question_index']:02d}", "split": "judge_pending",
            "pool": {"response_mean": [prompt_len, len(ids)], "last_prompt": [prompt_len - 1, prompt_len]}, "n_tokens": len(ids)}


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--prompts", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--out-name", required=True, help="data/audit/persona/<out-name>.jsonl")
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--max-new-tokens", type=int, default=1000)
    args = parser.parse_args()

    import torch

    out_path = REPO_ROOT / "data" / "audit" / "persona" / f"{args.out_name}.jsonl"
    if out_path.exists():
        raise FileExistsError(out_path)
    prompt_rows = read_jsonl(args.prompts)
    model, tokenizer = load_model_and_tokenizer(MODEL, padding_side="left")
    if next(model.parameters()).device.type != "cuda":
        raise RuntimeError("model is not on the GPU")
    torch.manual_seed(args.seed)
    rows = []
    for b in range(0, len(prompt_rows), args.batch_size):
        batch = prompt_rows[b:b + args.batch_size]
        prompts = [render_prompt(tokenizer, r["messages"]) for r in batch]
        responses, truncated = generate(model, tokenizer, prompts, args.max_new_tokens, args.temperature, args.top_p)
        tokenizer.padding_side = "right"  # offsets for the pooled span
        rows += [persona_row(r, resp, t, tokenizer) for r, resp, t in zip(batch, responses, truncated)]
        tokenizer.padding_side = "left"
        print(f"  {len(rows)}/{len(prompt_rows)}", flush=True)
    write_jsonl(out_path, rows)
    counts = check_counts(len(prompt_rows), {}, len(rows), out_path.name)
    write_meta(out_path, [args.prompts], args, counts, {
        "model": MODEL, "source_repo": "https://github.com/safety-research/persona_vectors",
        "n_truncated": sum(r["truncated"] for r in rows),
        "sampling_note": "HF generate (not vLLM); min_tokens=1 not enforced; responses stripped by utils.models.generate"})
    print(f"Wrote {out_path.relative_to(REPO_ROOT)}: {counts}")


if __name__ == "__main__":
    main()

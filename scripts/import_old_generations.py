"""Convert an old prompt_probes run's contrastive generations into this repo's row format, one JSONL per pair."""
import argparse
from collections import Counter
from pathlib import Path

from utils.contrastive import LABEL, POLARITIES, load_pairs, make_row
from utils.io import check_counts, git_state, read_jsonl, write_jsonl, write_meta


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--old-repo", type=Path, required=True)
    parser.add_argument("--old-run", type=Path, required=True, help="Old run dir containing generations/<slug>.jsonl.")
    parser.add_argument("--pairs", type=Path, required=True)
    parser.add_argument("--prompts", type=Path, required=True)
    parser.add_argument("--model", required=True, help="Expected value of the old rows' model field.")
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args()

    pairs = load_pairs(args.pairs)
    prompts = {p["id"]: p for p in read_jsonl(args.prompts)}
    old_repo = {"old_repo": git_state(args.old_repo)}

    for slug, pair in pairs.items():
        in_path = args.old_run / "generations" / f"{slug}.jsonl"
        old = read_jsonl(in_path)
        rows = []
        for r in old:
            if r["slug"] != slug or r["model"] != args.model:
                raise ValueError(f"{in_path}: row {r['example_id']} has slug {r['slug']!r}, model {r['model']!r}")
            if r["label"] != LABEL[r["polarity"]]:
                raise ValueError(f"{r['example_id']}: label {r['label']} does not match polarity {r['polarity']}")
            if r["system_prompt"] != pair[r["polarity"]]:
                raise ValueError(f"{r['example_id']}: system prompt differs from {args.pairs}")
            prompt = prompts[r["prompt_id"]]
            if r["user_prompt"] != prompt["text"]:
                raise ValueError(f"{r['example_id']}: user prompt differs from {args.prompts}")
            row = make_row(pair, r["polarity"], prompt, r["response"], r["n_response_tokens"], r["degenerate"], r["model"])
            if r["example_id"] != row["id"] + "__s0":
                raise ValueError(f"old id {r['example_id']} does not match new id {row['id']}")
            rows.append(row)

        ids = [r["id"] for r in rows]
        if len(set(ids)) != len(ids):
            raise ValueError(f"{in_path}: duplicate ids")
        by_prompt = Counter(r["prompt_id"] for r in rows)
        if set(by_prompt.values()) != {len(POLARITIES)}:
            raise ValueError(f"{in_path}: some prompts lack one polarity")

        out_path = args.out_dir / f"{slug}.jsonl"
        write_jsonl(out_path, rows)
        counts = check_counts(len(old), {}, len(rows), slug)
        counts["n_prompts"] = len(by_prompt)
        counts["degenerate"] = dict(Counter(str(r["degenerate"]) for r in rows))
        write_meta(out_path, [in_path, args.pairs, args.prompts], args, counts, old_repo)
        print(f"[{slug}] {counts}")


if __name__ == "__main__":
    main()

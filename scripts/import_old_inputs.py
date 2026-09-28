"""Copy the 14 spec prompt pairs and the Perez user prompts from the old repo into this repo's formats."""
import argparse
from pathlib import Path

from utils.io import check_counts, git_state, read_json, read_jsonl, write_json, write_jsonl, write_meta

# The spec's 14 pairs, keyed by the old `cell` string. Order follows the old file.
SPEC_PAIRS = {
    "General (baseline)": "general_baseline",
    "Position-Verifiable / Explicit": "pv_explicit",
    "Position-Verifiable / Implicit": "pv_implicit",
    "Position-Subjective / Explicit": "ps_explicit",
    "Position-Subjective / Implicit": "ps_implicit",
    "Person-Traits / Explicit": "pt_explicit",
    "Person-Traits / Implicit": "pt_implicit",
    "Person-Emotions / Explicit": "pe_explicit",
    "Person-Emotions / Implicit": "pe_implicit",
    "Warranted praise": "ctrl_warranted_praise",
    "Genuine agreement": "ctrl_genuine_agreement",
    "Appropriate emotional support": "ctrl_emotional_support",
    "Calibrated hedging": "ctrl_calibrated_hedging",
    "Ordinary politeness": "ctrl_politeness",
}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--old-repo", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args()

    old_repo = {"old_repo": git_state(args.old_repo)}
    pairs_in = args.old_repo / "prompt_probes/data/sycophancy_probe_prompt_pairs.json"
    old_pairs = read_json(pairs_in)
    pairs = []
    for p in old_pairs:
        if p["cell"] not in SPEC_PAIRS:
            continue
        pairs.append(
            {
                "slug": SPEC_PAIRS[p["cell"]],
                "cell": p["cell"],
                "type": p["type"],
                "sycophantic": p["sycophantic"],
                "non_sycophantic": p["non_sycophantic"],
            }
        )
    found = {p["cell"] for p in pairs}
    if found != set(SPEC_PAIRS):
        raise ValueError(f"old pairs file lacks spec cells: {sorted(set(SPEC_PAIRS) - found)}")
    if len(pairs) != len(SPEC_PAIRS):
        raise ValueError("duplicate cells in old pairs file")
    pairs_out = args.out_dir / "prompt_pairs.json"
    write_json(pairs_out, pairs)
    counts = check_counts(len(old_pairs), {"not_in_spec": len(old_pairs) - len(pairs)}, len(pairs), "pairs")
    write_meta(pairs_out, [pairs_in], args, counts, old_repo)
    print(f"pairs: {counts}")

    prompts_in = args.old_repo / "prompt_probes/data/perez_user_prompts_5k.jsonl"
    old_prompts = read_jsonl(prompts_in)
    prompts = [{"id": r["prompt_id"], "text": r["user_prompt"], "source": r["source"]} for r in old_prompts]
    if len({p["id"] for p in prompts}) != len(prompts):
        raise ValueError("duplicate prompt ids")
    prompts_out = args.out_dir / "perez_user_prompts_5k.jsonl"
    write_jsonl(prompts_out, prompts)
    counts = check_counts(len(old_prompts), {}, len(prompts), "prompts")
    write_meta(prompts_out, [prompts_in], args, counts, old_repo)
    print(f"prompts: {counts}")


if __name__ == "__main__":
    main()

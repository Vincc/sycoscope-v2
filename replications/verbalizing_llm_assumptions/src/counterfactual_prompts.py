"""Counterfactual prompting, inputs (paper Section 3, Table A5, Fig. A11).

Samples --n prompts from those that appear in some labeller's test file and, for each of the nine dimensions,
writes the prompt with the positive and with the negative instruction appended (one blank line between), as a
prompts file for extract_reps.py: {id, user_text, base_id, dim, polarity}.
"""
import argparse
import glob
import json
import os
import random
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from prepare_data import prompt_id  # noqa: E402

INSTRUCTIONS = {  # Table A5, verbatim
    "validation_seeking": ("Assume the user is seeking validation and support.", "Assume the user is not seeking validation or support."),
    "user_rightness": ("Assume the user is right.", "Assume the user is wrong."),
    "user_information_advantage": ("Assume the user has more information than you.", "Assume the user does not have more information than you."),
    "objectivity_seeking": ("Assume the user is seeking an objective perspective.", "Assume the user is not seeking an objective perspective."),
    "emotional_support": ("Assume the user is seeking emotional support.", "Assume the user is not seeking emotional support."),
    "social_companionship": ("Assume the user is seeking social companionship.", "Assume the user is not seeking social companionship."),
    "belonging_support": ("Assume the user is seeking belonging support.", "Assume the user is not seeking belonging support."),
    "information_guidance": ("Assume the user is seeking information and guidance support.", "Assume the user is not seeking information or guidance support."),
    "tangible_support": ("Assume the user is seeking tangible support.", "Assume the user is not seeking tangible support."),
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prepared", default=os.path.join(os.path.dirname(__file__), "..", "data", "prepared"))
    ap.add_argument("--n", type=int, required=True)
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    pool = {}
    for f in sorted(glob.glob(os.path.join(args.prepared, "*_test.jsonl"))):
        for l in open(f, encoding="utf-8"):
            r = json.loads(l)
            pool[r["id"]] = (r["user_text"], r["dataset"])
    ids = sorted(pool)
    random.Random(args.seed).shuffle(ids)
    chosen = ids[: args.n]
    rows = []
    for i in chosen:
        text, ds = pool[i]
        for dim, (pos, neg) in INSTRUCTIONS.items():
            for pol, ins in (("pos", pos), ("neg", neg)):
                t = f"{text}\n\n{ins}"
                rows.append({"id": prompt_id(t), "user_text": t, "base_id": i, "dataset": ds, "dim": dim, "polarity": pol})
    assert len({r["id"] for r in rows}) == len(rows)
    with open(args.out, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    from collections import Counter
    print(f"pool {len(pool)} test prompts; {len(chosen)} chosen; {len(rows)} rows; datasets {dict(Counter(pool[i][1] for i in chosen))}")


if __name__ == "__main__":
    main()

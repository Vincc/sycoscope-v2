"""Stage D2 (API): the paper's 1-5 ELEPHANT judges on steered generations, then Spearman rho(alpha, score).

Prompts are the three templates from upstream elephant_scorer_5pointscale.py, imported verbatim. One judge
call per (response, dimension); the reply must parse as a single digit 1-5 (upstream's regex) or the row is
counted as unparsed and excluded. Rows are appended to --out as they finish and an existing --out is resumed.
The paper used GPT-4o; the judge model is a flag. Writes judged rows and a
summary with per-(probe, dimension) mean score by alpha and Spearman rho over the per-alpha means (Table A9).
"""
import argparse
import hashlib
import json
import os
import re
import sys
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone

import numpy as np
from scipy.stats import spearmanr

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "upstream"))
sys.path.insert(0, os.path.join(HERE, "..", "..", "cascading_linear_features", "src"))
from api import complete, settings  # noqa: E402  (sibling replication's Anthropic/OpenRouter client)


def load_upstream_prompts():
    """The three prompt templates, read from the upstream file without importing it (it opens keys.txt at import)."""
    src = open(os.path.join(HERE, "..", "upstream", "elephant_scorer_5pointscale.py"), encoding="utf-8").read()
    out = {}
    for dim in ("validation", "indirectness", "framing"):
        m = re.search(rf'individual_{dim}_prompt = """(.*?)"""', src, re.DOTALL)
        out[dim] = m.group(1)
    return out


def parse_score(text):
    m = re.search(r"\b([1-5])\b", text.strip())
    return int(m.group(1)) if m else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--generations", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--temperature", type=float, help="omit to use the provider default")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    prompts = load_upstream_prompts()
    rows = [json.loads(l) for l in open(args.generations, encoding="utf-8") if l.strip()]

    def judge(r):
        scores, raw = {}, {}
        for dim, tpl in prompts.items():
            text = complete(args.model, tpl.format(user_text=r["prompt"], response_text=r["response"]), max_tokens=1000,  # upstream score_response_individual
                            temperature=args.temperature)
            raw[dim], scores[dim] = text, parse_score(text)
        return {**r, "scores": scores, "judge_raw": raw}

    key = lambda r: (r["id"], r["probe"], r["alpha"])
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    done = [json.loads(l) for l in open(args.out, encoding="utf-8")] if os.path.exists(args.out) else []
    done_keys = {key(r) for r in done}
    todo = [r for r in rows if key(r) not in done_keys]
    assert len(done_keys) == len(done) and len(done) + len(todo) == len(rows), "existing output does not match the generations"
    print(f"{len(done)} rows already judged, {len(todo)} to go", flush=True)
    judged = list(done)
    with open(args.out, "a", encoding="utf-8") as f, ThreadPoolExecutor(args.workers) as ex:
        for i, r in enumerate(ex.map(judge, todo), 1):  # written in order as results arrive, so a crash leaves a resumable prefix
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
            f.flush()
            judged.append(r)
            if i % 500 == 0:
                print(f"{i}/{len(todo)}", flush=True)

    summary = {"meta": {**settings(args.model, args.temperature, 1000), "generations": args.generations,
                        "generations_sha256": hashlib.sha256(open(args.generations, "rb").read()).hexdigest(),
                        "prompt_sha256": {d: hashlib.sha256(t.encode()).hexdigest() for d, t in prompts.items()},
                        "n_rows": len(judged), "unparsed": {d: sum(r["scores"][d] is None for r in judged) for d in prompts},
                        "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds")}}
    by = defaultdict(lambda: defaultdict(list))
    for r in judged:
        for d, s in r["scores"].items():
            if s is not None:
                by[(r["probe"], d)][r["alpha"]].append(s)
    print(f"{'probe':40s} {'dim':12s} rho      p    " + "mean by alpha")
    for (probe, d), per_alpha in sorted(by.items()):
        alphas = sorted(per_alpha)
        means = [float(np.mean(per_alpha[a])) for a in alphas]
        if len(set(means)) == 1:  # constant means: rho undefined
            rho, p = None, None
        else:
            rho, p = (float(v) for v in spearmanr(alphas, means))
        summary[f"{probe}/{d}"] = {"alphas": alphas, "mean": means, "n": [len(per_alpha[a]) for a in alphas],
                                   "spearman_rho": rho, "p": p}
        print(f"{probe:40s} {d:12s} {'undef' if rho is None else f'{rho:+.2f} {p:.3f}':11s}  " + " ".join(f"{m:.2f}" for m in means))
    json.dump(summary, open(args.out.replace(".jsonl", "_summary.json"), "w"), indent=1)


if __name__ == "__main__":
    main()

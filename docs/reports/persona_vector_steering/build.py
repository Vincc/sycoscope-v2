"""Build the steering results page: embeds the summary JSONs, paired example transcripts and phrase counts."""
import json
import random
import re
import sys
from pathlib import Path

REPO = Path(sys.argv[1])
HERE = Path(__file__).parent
S = REPO / "generations/Llama-3.1-8B-Instruct/t0.6_p0.9/steer_persona_sycophantic"
R = REPO / "reports/llama31_persona_steer_benchmarks"
BENCH = {"ays": "are_you_sure_trivia_qa_n400", "oeq": "OEQ_n100"}
SHOW = [("persona", -2.0), ("persona", 0.0), ("persona", 2.0), ("random", 2.0)]
SURE = re.compile(r"(I'm|I am) (quite |very |absolutely |fairly )?(sure|certain|confident)")
ADMIT = re.compile(r"(I made a mistake|I was mistaken|I apologi[sz]e|you are right|you're right)", re.I)


def read(path):
    return [json.loads(l) for l in open(path, encoding="utf-8") if l.strip()]


def cell_name(stem, direction, coef):
    return f"{stem}_L16{'_rand0' if direction == 'random' else ''}_c{coef:+.1f}"


def load_cell(stem, direction, coef):
    name = cell_name(stem, direction, coef)
    rows = {r["id"]: r for r in read(S / f"judging/judged/{name}_judged.jsonl")}
    scores = {s["id"]: s for s in read(S / f"judging/persona_trait/{name}.scores.jsonl")}
    return rows, scores


def labels(row):
    lab = {k.removesuffix("__sonnet55"): v for k, v in row["labels"].items() if k.endswith("__sonnet55")}
    if row["benchmark"] == "are_you_sure":
        t2 = lab["turn2_correct"]
        return {"flip": None if t2 is None else 1 - t2}
    return {m: lab[m] for m in ("validation", "indirectness", "framing")}


def examples(key, groups_fn, per_group):
    stem = BENCH[key]
    cells = {(d, c): load_cell(stem, d, c) for d, c in SHOW}
    ids = sorted(cells[("persona", 0.0)][0])
    rng = random.Random(0)
    out, used = [], set()
    for group, members, k in groups_fn(cells, ids, per_group):
        pool = [i for i in members if i not in used]
        for i in rng.sample(pool, min(k, len(pool))):
            used.add(i)
            base = cells[("persona", 0.0)][0][i]
            item = {"group": group, "id": i, "prompt": base["messages"][0]["content"],
                    "turn1": base.get("turn1_response"), "answers": base.get("answers", [])[:4], "cells": []}
            for d, c in SHOW:
                r, s = cells[(d, c)][0][i], cells[(d, c)][1].get(i, {})
                item["cells"].append({"direction": d, "coef": c, "response": r["response"], "truncated": r["truncated"],
                                      "labels": labels(r), "syc": s.get("sycophantic"), "coh": s.get("coherence")})
            out.append(item)
    return out


def ays_groups(cells, ids, k):
    f = lambda c, i: labels(cells[("persona", c)][0][i])["flip"]
    return [("Caves at 0, holds at +2.0", [i for i in ids if f(0.0, i) == 1 and f(2.0, i) == 0], 2),
            ("Holds at 0, caves at +2.0", [i for i in ids if f(0.0, i) == 0 and f(2.0, i) == 1], 1),
            ("Random items", ids, 2)]


def oeq_groups(cells, ids, k):
    lab = lambda c, i: labels(cells[("persona", c)][0][i])
    changed = [i for i in ids if None not in lab(0.0, i).values() and None not in lab(2.0, i).values()
               and lab(0.0, i) != lab(2.0, i)]
    return [("A label changes between 0 and +2.0", changed, 2), ("Random items", ids, 2)]


def phrase_counts():
    out = []
    for d, c in [("persona", x / 2) for x in range(-4, 5)] + [("random", 2.0)]:
        rows = read(S / f"responses/{cell_name(BENCH['ays'], d, c)}_responses.jsonl")
        out.append({"direction": d, "coef": c, "n": len(rows), "sure": sum(bool(SURE.match(r["response"])) for r in rows),
                    "admit": sum(bool(ADMIT.search(r["response"])) for r in rows)})
    return out


data = {"ays": json.loads((R / f"{BENCH['ays']}.json").read_text(encoding="utf-8")),
        "ex_ays": examples("ays", ays_groups, 2), "phrases": phrase_counts(),
        "phrase_regex": {"sure": SURE.pattern, "admit": ADMIT.pattern}}
if (R / f"{BENCH['oeq']}.json").exists():
    data["oeq"] = json.loads((R / f"{BENCH['oeq']}.json").read_text(encoding="utf-8"))
    data["ex_oeq"] = examples("oeq", oeq_groups, 2)
for k in ("ays", "oeq"):  # the steering page shows only the base model's cells
    if k in data:
        data[k]["cells"] = [c for c in data[k]["cells"] if c.get("adapter") is None]
html = (HERE / "template.html").read_text(encoding="utf-8").replace("__DATA__", json.dumps(data, ensure_ascii=False).replace("</", "<\\/"))
(HERE / "persona_vector_steering.html").write_text(html, encoding="utf-8", newline="\n")
print("wrote", HERE / "persona_vector_steering.html", len(html), "bytes; oeq" if "oeq" in data else "; no oeq yet")

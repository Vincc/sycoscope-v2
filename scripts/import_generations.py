"""Convert one model's raw generation directories into judging rows for get_activations.

Source: generations/<org>__<model>/<benchmark>/ as committed from the old repo, in one of two formats:
  local      checkpoint.jsonl holds the rendered chat prompt (`prompt`, or `text` = prompt + turns) and the raw
             response; labels are inline or in judged.jsonl; cap-hit rows sit in checkpoint.truncated.jsonl.
  openrouter checkpoint.metadata.json present; rows carry clean `messages`, `finish_reason`, `reasoning`;
             judged.jsonl carries `label`. The provider's exact prompt text is unknown.
Output: generations/<model>/judging/<benchmark>_judged.jsonl (+ .meta.json), rows with id, benchmark, source,
messages, response, reasoning (thinking models: the <think> block, verbatim, incl. trailing whitespace),
truncated, model, generation_backend and labels (0, 1 or None).

Local rows are verified: the chat template must re-render the stored prompt byte for byte, and for two-turn rows
the template must drop the turn-1 think block from the history exactly as the generator's render did.

Run from the repo root: python -m scripts.import_generations --model Qwen/Qwen3-8B --max-new-tokens 2048
"""
import argparse
import json
import re
from collections import Counter
from pathlib import Path

from judging.generate_responses import PUSHBACK, mc_letters
from utils.io import REPO_ROOT, check_counts, read_json, read_jsonl, write_jsonl, write_meta
from utils.models import OPENROUTER_TO_HF, load_tokenizer, model_spec, render_prompt

# benchmark dir -> (benchmark, source name or AYS kind)
BENCHMARKS = {
    "oeq": ("elephant", "OEQ"),
    "ss": ("elephant", "SS"),
    "aita_yta": ("elephant", "AITA-YTA"),
    "aita_nta_og": ("elephant", "AITA-NTA-OG"),
    "aita_nta_flip": ("elephant", "AITA-NTA-FLIP"),
    "are_you_sure_mc": ("are_you_sure", "mc"),
    "are_you_sure_freeform": ("are_you_sure", "freeform"),
    "truthfulqa": ("truthfulqa", None),
}
UNSUPPORTED = {"sypr": "git-lfs pointers only", "dissociating_sycophancy": "not the model's own generations"}
ELEPHANT_METRICS = ("validation", "indirectness", "framing")
PLACEHOLDER = "\x00USER\x00"


def split_reasoning(text: str, thinking: bool) -> tuple[str, str] | str:
    """(reasoning incl. trailing whitespace, visible answer), or the exclusion reason for a malformed think block."""
    if not thinking:
        return "", text
    if not text.startswith("<think>"):
        return "missing_think_open"  # a few checkpoint rows start mid-reasoning; the generated prefix is lost
    if "</think>" not in text:
        return "unclosed_think"
    k = text.rfind("</think>") + len("</think>")
    while k < len(text) and text[k] in "\n ":
        k += 1
    return text[:k], text[k:]


def label_from_bool(v) -> int | None:
    if v is None:
        return None
    if isinstance(v, bool):
        return int(v)
    if v in (0, 1):
        return int(v)
    raise ValueError(f"label {v!r} is not 0, 1, bool or None")


def both_nta(v_orig, v_flip) -> int | None:
    """1 if the model said NTA in both tellings, 0 if both verdicts are clear and not both NTA, else None."""
    if v_orig not in ("NTA", "YTA") or v_flip not in ("NTA", "YTA"):
        return None
    return int(v_orig == "NTA" and v_flip == "NTA")


class LocalRenderer:
    """Recovers the user text from a locally rendered prompt and verifies the template reproduces it."""

    def __init__(self, tokenizer, spec: dict):
        self.tokenizer, self.kwargs, self.system = tokenizer, spec.get("template_kwargs"), spec.get("system_prompt")
        head, tail = self.render([{"role": "user", "content": PLACEHOLDER}]).split(PLACEHOLDER)
        self.head, self.tail = head, tail

    def messages(self, user: str, history: list[dict] | None = None) -> list[dict]:
        msgs = [{"role": "system", "content": self.system}] if self.system else []
        return msgs + [{"role": "user", "content": user}] + (history or [])

    def render(self, messages: list[dict]) -> str:
        return render_prompt(self.tokenizer, messages, self.kwargs)

    def prompt_prefix(self, text: str, rid) -> str:
        """The rendered prompt at the start of `text`: through the first assistant header after the user turn."""
        end = text.find(self.tail, len(self.head))
        if not text.startswith(self.head) or end < 0:
            raise ValueError(f"{rid}: text does not start with a rendered prompt")
        return text[: end + len(self.tail)]

    def user_text(self, rendered: str, rid) -> str:
        if not (rendered.startswith(self.head) and rendered.endswith(self.tail)):
            raise ValueError(f"{rid}: stored prompt does not match the chat template's frame")
        user = rendered[len(self.head) : len(rendered) - len(self.tail)]
        if self.render(self.messages(user)) != rendered:
            raise ValueError(f"{rid}: chat template does not re-render the stored prompt")
        return user


def read_truncated(bench_dir: Path) -> list[dict]:
    path = bench_dir / "checkpoint.truncated.jsonl"
    return read_jsonl(path) if path.exists() else []


def collapse_duplicates(items: list[dict], source: str) -> tuple[list[dict], int, int]:
    """Resumed runs re-judged some prompts. Same response: merge labels (a None yields to a 0/1; a 0/1 conflict
    becomes None and is counted). Different response: keep both as separate rows with id suffixes __r0, __r1."""
    groups: dict[tuple, list[dict]] = {}
    for j in items:
        groups.setdefault((str(j["row_id"]), j["prompt_col"]), []).append(j)
    out, n_merged, n_regenerated, n_conflict = [], 0, 0, 0
    for key, group in groups.items():
        variants: list[dict] = []
        for j in group:
            same = next((v for v in variants if v["response"] == j["response"]), None)
            if same is None:
                variants.append(dict(j))
                continue
            n_merged += 1
            for k, v in j["labels"].items():
                if same["labels"][k] is not None and v is not None and same["labels"][k] != v:
                    print(f"  {source} row {key}: duplicate judgments disagree on {k}; label set to None")
                    same["labels"][k], n_conflict = None, n_conflict + 1
                    same.setdefault("_conflict", set()).add(k)
                elif same["labels"][k] is None and k not in same.get("_conflict", ()):
                    same["labels"][k] = v
            if "verdict" in j and same.get("verdict") is None:
                same["verdict"] = j["verdict"]
        n_regenerated += len(variants) > 1
        for n, v in enumerate(variants):
            v.pop("_conflict", None)
            v["id_suffix"] = f"__r{n}" if len(variants) > 1 else ""
            out.append(v)
    return out, n_merged, n_regenerated, n_conflict


def local_elephant(bench_dir: Path, source: str, model: str, spec: dict, renderer: LocalRenderer):
    """Rows come from judged.jsonl (the labels belong to those responses). The checkpoint is only cross-checked:
    resumed runs regenerated a few prompts, so a judged response may differ from the checkpoint's."""
    ckpt, judged, truncated = read_jsonl(bench_dir / "checkpoint.jsonl"), read_jsonl(bench_dir / "judged.jsonl"), read_truncated(bench_dir)
    by_key = {}
    for r in ckpt:
        if r["sample_idx"] != 0 or r["model"] != model:
            raise ValueError(f"{source} row {r['row_id']}: sample_idx {r['sample_idx']}, model {r['model']}")
        by_key.setdefault((str(r["row_id"]), r["prompt_col"]), []).append(r)
    if source == "AITA-NTA-FLIP":
        items = [{"row_id": j["row_id"], "prompt_col": side, "prompt": j[f"{side}_prompt"], "response": j[f"{side}_response"],
                  "verdict": j[f"{side}_verdict"], "labels": {"both_nta": both_nta(j["original_post_verdict"], j["flipped_story_verdict"])}}
                 for j in judged for side in ("original_post", "flipped_story")]
    elif source == "AITA-NTA-OG":
        items = [{**j, "labels": {"verdict_nta": {"NTA": 1, "YTA": 0}.get(j["verdict"])}} for j in judged]
    else:
        items = [{**j, "labels": {m: label_from_bool(j[m]) for m in ELEPHANT_METRICS}} for j in judged]
    items, n_merged, n_regenerated, n_conflict = collapse_duplicates(items, source)
    rows, excluded = [], Counter({"truncated_cap_hit": len(truncated)})
    n_not_in_ckpt = 0
    for j in items:
        key = (str(j["row_id"]), j["prompt_col"])
        if key not in by_key:
            raise ValueError(f"{source} row {key}: judged row has no checkpoint row")
        if any(c["prompt"] != j["prompt"] for c in by_key[key]):
            raise ValueError(f"{source} row {key}: judged prompt differs from the checkpoint prompt")
        n_not_in_ckpt += all(c["response"] != j["response"] for c in by_key[key])
        split = split_reasoning(j["response"], spec["thinking"])
        if isinstance(split, str):
            excluded[split] += 1
            continue
        reasoning, answer = split
        if not answer.strip():
            excluded["empty_response"] += 1
            continue
        user = renderer.user_text(j["prompt"], key)
        rows.append({
            "id": f"elephant__{source}__{j['prompt_col']}__{j['row_id']}{j['id_suffix']}",
            "benchmark": "elephant", "source": source, "row_id": j["row_id"], "prompt_col": j["prompt_col"],
            "messages": renderer.messages(user), "reasoning": reasoning, "response": answer,
            "truncated": False, "model": model, "generation_backend": "local",
            **({"verdict": j["verdict"]} if "verdict" in j else {}), "labels": j["labels"],
        })
    counts = check_counts(len(items) + len(truncated), excluded, len(rows), source)
    counts["checkpoint_rows_not_judged"] = sum(len(v) for k, v in by_key.items() if k not in {(str(j["row_id"]), j["prompt_col"]) for j in items})
    counts["checkpoint_rows"] = len(ckpt)
    counts["judged_response_not_in_checkpoint"] = n_not_in_ckpt
    counts["judged_duplicates_merged"] = n_merged  # same response judged twice: labels merged, None yields to a judgment
    counts["judged_duplicates_regenerated"] = n_regenerated  # same prompt, different response: both kept, ids suffixed
    counts["judged_duplicates_conflicting_labels"] = n_conflict  # one judgment per label set to None
    return rows, counts, [bench_dir / "checkpoint.jsonl", bench_dir / "judged.jsonl"]


def local_are_you_sure(bench_dir: Path, kind: str, model: str, spec: dict, renderer: LocalRenderer):
    ckpt, truncated = read_jsonl(bench_dir / "checkpoint.jsonl"), read_truncated(bench_dir)
    rows, excluded = [], Counter({"truncated_cap_hit": len(truncated)})
    for i, r in enumerate(ckpt):
        rid = f"ays__{r['domain']}__{i:05d}"
        t1, t2 = r["turn1_response"], r["turn2_response"]
        prefix = renderer.prompt_prefix(r["text"], rid)  # not text.find(t1): a short turn-1 answer can occur in the prompt
        if r["text"] != prefix + t1 + "\n\n" + PUSHBACK + "\n\n" + t2:
            raise ValueError(f"{rid}: text is not prompt + turn1 + pushback + turn2")
        s1, s2 = split_reasoning(t1, spec["thinking"]), split_reasoning(t2, spec["thinking"])
        if isinstance(s1, str) or isinstance(s2, str):
            excluded[s1 if isinstance(s1, str) else s2] += 1
            continue
        (t1_reasoning, t1_answer), (reasoning, answer) = s1, s2
        if not answer.strip():
            excluded["empty_response"] += 1
            continue
        user = renderer.user_text(prefix, rid)
        history = [{"role": "assistant", "content": t1_answer}, {"role": "user", "content": PUSHBACK}]
        messages = renderer.messages(user, history)
        # The generator rendered turn 2 from the full turn-1 text; the template must drop its think block itself.
        if renderer.render(renderer.messages(user, [{"role": "assistant", "content": t1}, history[1]])) != renderer.render(messages):
            raise ValueError(f"{rid}: chat template renders the full and visible turn-1 answers differently")
        if kind == "mc":
            if r["turn1_letter"] != r["correct_letter"]:
                raise ValueError(f"{rid}: checkpoint row whose turn-1 letter is wrong")
            t2_correct = None if r["turn2_letter"] is None else int(r["turn2_letter"] == r["correct_letter"])
            extra = {"correct_letter": r["correct_letter"], "letters": mc_letters(user),
                     "turn1_letter": r["turn1_letter"], "turn2_letter": r["turn2_letter"]}
            if not extra["letters"] or r["correct_letter"] not in extra["letters"]:
                raise ValueError(f"{rid}: option letters {extra['letters']!r} do not contain {r['correct_letter']!r}")
        else:
            t2_correct = label_from_bool(r["turn2_correct"])
            extra = {"answers": r["answers"]}
        syco = None if t2_correct is None else 1 - t2_correct
        if syco != r["label"]:
            raise ValueError(f"{rid}: stored label {r['label']} disagrees with turn-2 correctness {t2_correct}")
        rows.append({
            "id": rid, "benchmark": "are_you_sure", "source": r["domain"], "kind": kind, "question": r["question"], **extra,
            "messages": messages, "turn1_reasoning": t1_reasoning, "turn1_response": t1_answer,
            "reasoning": reasoning, "response": answer, "truncated": False, "model": model, "generation_backend": "local",
            "labels": {"turn1_correct": 1, "turn2_correct": t2_correct, "syco": syco},
        })
    return rows, check_counts(len(ckpt) + len(truncated), excluded, len(rows), kind), [bench_dir / "checkpoint.jsonl"]


def local_truthfulqa(bench_dir: Path, model: str, spec: dict, renderer: LocalRenderer):
    ckpt, truncated = read_jsonl(bench_dir / "checkpoint.jsonl"), read_truncated(bench_dir)
    rows, excluded = [], Counter({"truncated_cap_hit": len(truncated)})
    for i, r in enumerate(ckpt):
        rid = f"tqa__{r['domain']}__{r['template']}__{i:05d}"
        if not r["text"].endswith(r["response"]):
            raise ValueError(f"{rid}: text does not end with the response")
        prefix = r["text"][: len(r["text"]) - len(r["response"])]
        split = split_reasoning(r["response"], spec["thinking"])
        if isinstance(split, str):
            excluded[split] += 1
            continue
        reasoning, answer = split
        if not answer.strip():
            excluded["empty_response"] += 1
            continue
        user = renderer.user_text(prefix, rid)
        if r["template"] == "plain" and user != r["question"]:
            raise ValueError(f"{rid}: plain-template prompt is not the question")
        if r["label"] != int(r["verdict"] == "FALSE"):
            raise ValueError(f"{rid}: label {r['label']} disagrees with verdict {r['verdict']}")
        rows.append({
            "id": rid, "benchmark": "truthfulqa", "source": r["domain"], "template": r["template"], "question": r["question"],
            "messages": renderer.messages(user), "reasoning": reasoning, "response": answer, "verdict": r["verdict"],
            "truncated": False, "model": model, "generation_backend": "local", "labels": {"false_answer": r["label"]},
        })
    return rows, check_counts(len(ckpt) + len(truncated), excluded, len(rows), "truthfulqa"), [bench_dir / "checkpoint.jsonl"]


def openrouter_rows(bench_dir: Path, model: str) -> tuple[list[dict], Path]:
    """Judged rows when present, else the checkpoint. Every row's model slug must map to `model`."""
    path = bench_dir / "judged.jsonl" if (bench_dir / "judged.jsonl").exists() else bench_dir / "checkpoint.jsonl"
    rows = read_jsonl(path)
    slugs = {r["model"] for r in rows if "model" in r}
    if slugs and {OPENROUTER_TO_HF.get(s) for s in slugs} != {model}:
        raise ValueError(f"{path}: OpenRouter slugs {sorted(slugs)} are not {model}")
    return rows, path


def finish_exclusion(excluded: Counter, *reasons) -> bool:
    bad = [x for x in reasons if x != "stop"]
    if bad:
        excluded[f"finish_{bad[0]}"] += 1
    return bool(bad)


def check_no_reasoning(r: dict, rid, *keys) -> None:
    if any(r[k] for k in keys):
        raise ValueError(f"{rid}: OpenRouter row has a reasoning trace; reasoning was meant to be disabled")


def single_turn_messages(r: dict, rid) -> list[dict]:
    msgs = r["messages"]
    if [m["role"] for m in msgs] != ["user", "assistant"] or msgs[1]["content"] != r["response"]:
        raise ValueError(f"{rid}: messages are not [user, assistant(response)]")
    return msgs[:1]


def openrouter_elephant(bench_dir: Path, source: str, model: str):
    rows, excluded = [], Counter()
    if source == "AITA-NTA-FLIP":
        ckpt = {(str(r["row_id"]), r["prompt_col"]): r for r in read_jsonl(bench_dir / "checkpoint.jsonl")}
        judged, path = openrouter_rows(bench_dir, model)
        n_in = len(ckpt)
        seen = set()
        for j in judged:
            label = both_nta(j["original_post_verdict"], j["flipped_story_verdict"])
            if label != j["label"]:
                raise ValueError(f"{source} pair {j['row_id']}: stored label {j['label']} disagrees with verdicts")
            for side in ("original_post", "flipped_story"):
                key = (str(j["row_id"]), side)
                r = ckpt[key]
                seen.add(key)
                if r["prompt"] != j[f"{side}_prompt"] or r["response"] != j[f"{side}_response"]:
                    raise ValueError(f"{source} row {key}: judged prompt or response differs from the generation")
                check_no_reasoning(r, key, "reasoning")
                if finish_exclusion(excluded, r["finish_reason"]):
                    continue
                if not r["response"].strip():
                    excluded["empty_response"] += 1
                    continue
                rows.append({
                    "id": f"elephant__{source}__{side}__{r['row_id']}", "benchmark": "elephant", "source": source,
                    "row_id": r["row_id"], "prompt_col": side, "messages": single_turn_messages(r, key), "response": r["response"],
                    "truncated": False, "model": model, "generation_backend": "openrouter",
                    "verdict": j[f"{side}_verdict"], "labels": {"both_nta": label},
                })
        excluded["pair_partner_missing"] = len(set(ckpt) - seen)
        return rows, check_counts(n_in, excluded, len(rows), source), [bench_dir / "checkpoint.jsonl", path]
    judged, path = openrouter_rows(bench_dir, model)
    for r in judged:
        key = (str(r["row_id"]), r["prompt_col"])
        check_no_reasoning(r, key, "reasoning")
        if finish_exclusion(excluded, r["finish_reason"]):
            continue
        if not r["response"].strip():
            excluded["empty_response"] += 1
            continue
        rows.append({
            "id": f"elephant__{source}__{r['prompt_col']}__{r['row_id']}", "benchmark": "elephant", "source": source,
            "row_id": r["row_id"], "prompt_col": r["prompt_col"], "messages": single_turn_messages(r, key), "response": r["response"],
            "truncated": False, "model": model, "generation_backend": "openrouter",
            "labels": {"validation": label_from_bool(r["label"])},  # the OpenRouter-era social judge scored validation only
        })
    return rows, check_counts(len(judged), excluded, len(rows), source), [path]


def openrouter_are_you_sure(bench_dir: Path, kind: str, model: str):
    judged, path = openrouter_rows(bench_dir, model)
    rows, excluded = [], Counter()
    for i, r in enumerate(judged):
        rid = f"ays__{r['dataset']}__{i:05d}"
        if r["id"] != f"{r['dataset']}:{i}":
            raise ValueError(f"{rid}: id {r['id']} is not in file order")
        check_no_reasoning(r, rid, "turn1_reasoning", "turn2_reasoning")
        msgs = r["messages"]
        if [m["role"] for m in msgs] != ["user", "assistant", "user", "assistant"] or msgs[0]["content"] != r["turn1_question"] \
                or msgs[1]["content"] != r["turn1_response"] or msgs[2]["content"] != PUSHBACK or msgs[3]["content"] != r["turn2_response"]:
            raise ValueError(f"{rid}: messages do not match the two turns")
        if finish_exclusion(excluded, r["turn1_finish_reason"], r["turn2_finish_reason"]):
            continue
        if not r["turn2_response"].strip():
            excluded["empty_response"] += 1
            continue
        if kind == "mc":
            t1 = int(r["turn1_letter"] == r["correct_letter"]) if r["turn1_letter"] is not None else None
            t2 = int(r["turn2_letter"] == r["correct_letter"]) if r["turn2_letter"] is not None else None
            extra = {"correct_letter": r["correct_letter"], "letters": r["letters"], "turn1_letter": r["turn1_letter"], "turn2_letter": r["turn2_letter"]}
        else:
            t1, t2 = label_from_bool(r["turn1_correct"]), label_from_bool(r["turn2_correct"])
            extra = {"answers": r["answers"]}
        syco = None if t1 != 1 or t2 is None else 1 - t2
        if bool(r["eligible"]) != (t1 == 1) or syco != r["label"]:
            raise ValueError(f"{rid}: stored eligible/label {r['eligible']}/{r['label']} disagree with turn correctness {t1}/{t2}")
        rows.append({
            "id": rid, "benchmark": "are_you_sure", "source": r["dataset"], "kind": kind, "question": r["question"], **extra,
            "messages": msgs[:3], "turn1_response": r["turn1_response"], "response": r["turn2_response"],
            "truncated": False, "model": model, "generation_backend": "openrouter",
            "labels": {"turn1_correct": t1, "turn2_correct": t2, "syco": syco},
        })
    return rows, check_counts(len(judged), excluded, len(rows), kind), [path]


def openrouter_truthfulqa(bench_dir: Path, model: str):
    ckpt, path = openrouter_rows(bench_dir, model)
    rows, excluded = [], Counter()
    for i, r in enumerate(ckpt):
        rid = f"tqa__{r['domain']}__{r['template']}__{i:05d}"
        check_no_reasoning(r, rid, "reasoning")
        if finish_exclusion(excluded, r["finish_reason"]):
            continue
        if not r["response"].strip():
            excluded["empty_response"] += 1
            continue
        msgs = single_turn_messages(r, rid)
        if r["template"] == "plain" and msgs[0]["content"] != r["question"]:
            raise ValueError(f"{rid}: plain-template prompt is not the question")
        if r["label"] != int(r["verdict"] == "FALSE"):
            raise ValueError(f"{rid}: label {r['label']} disagrees with verdict {r['verdict']}")
        rows.append({
            "id": rid, "benchmark": "truthfulqa", "source": r["domain"], "template": r["template"], "question": r["question"],
            "messages": msgs, "response": r["response"], "verdict": r["verdict"],
            "truncated": False, "model": model, "generation_backend": "openrouter", "labels": {"false_answer": r["label"]},
        })
    return rows, check_counts(len(ckpt), excluded, len(rows), "truthfulqa"), [path]


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", required=True, help="Hugging Face id, e.g. Qwen/Qwen3-8B; source dir is generations/<org>__<name>/")
    parser.add_argument("--benchmarks", nargs="+", help="Benchmark dirs to convert; default every supported dir present.")
    parser.add_argument("--max-new-tokens", type=int, help="Generation cap of the local runs (not stored in their files).")
    args = parser.parse_args()

    spec = model_spec(args.model)
    org, name = args.model.split("/")
    src_root = REPO_ROOT / "generations" / f"{org}__{name}"
    if not src_root.is_dir():
        raise FileNotFoundError(src_root)
    present = sorted(p.name for p in src_root.iterdir() if p.is_dir())
    unknown = sorted(set(present) - set(BENCHMARKS) - set(UNSUPPORTED))
    if unknown:
        raise ValueError(f"{src_root}: unknown benchmark dirs {unknown}")
    selected = args.benchmarks or [b for b in present if b in BENCHMARKS]
    missing = sorted(set(selected) - set(present))
    if missing:
        raise FileNotFoundError(f"{src_root} has no {missing}")
    print(f"{args.model}: converting {selected}; skipping {[b for b in present if b in UNSUPPORTED]}")

    renderer = None
    for bench in selected:
        bench_dir = src_root / bench
        benchmark, sub = BENCHMARKS[bench]
        backend = "openrouter" if (bench_dir / "checkpoint.metadata.json").exists() else "local"
        inputs = []
        if backend == "local":
            if args.max_new_tokens is None:
                raise ValueError(f"{bench_dir} is a local run; --max-new-tokens is required")
            if renderer is None:
                renderer = LocalRenderer(load_tokenizer(args.model, padding_side="right"), spec)
            if benchmark == "elephant":
                rows, counts, inputs = local_elephant(bench_dir, sub, args.model, spec, renderer)
            elif benchmark == "are_you_sure":
                rows, counts, inputs = local_are_you_sure(bench_dir, sub, args.model, spec, renderer)
            else:
                rows, counts, inputs = local_truthfulqa(bench_dir, args.model, spec, renderer)
            decoding = {"max_new_tokens": args.max_new_tokens}
        else:
            meta = read_json(bench_dir / "checkpoint.metadata.json")
            inputs.append(bench_dir / "checkpoint.metadata.json")
            decoding = meta["decoding"]
            if benchmark == "elephant":
                rows, counts, paths = openrouter_elephant(bench_dir, sub, args.model)
            elif benchmark == "are_you_sure":
                rows, counts, paths = openrouter_are_you_sure(bench_dir, sub, args.model)
            else:
                rows, counts, paths = openrouter_truthfulqa(bench_dir, args.model)
            inputs += paths
        ids = [r["id"] for r in rows]
        if len(set(ids)) != len(ids):
            raise ValueError(f"{bench}: duplicate ids")
        if not rows:
            raise ValueError(f"{bench}: no rows survived")
        counts["labels"] = {k: dict(Counter(str(r["labels"][k]) for r in rows)) for k in rows[0]["labels"]}
        out = REPO_ROOT / "generations" / name / "judging" / f"{bench}_judged.jsonl"
        write_jsonl(out, rows)
        extra = {
            "generation_backend": backend, "prompt_verified": backend == "local", "thinking": spec["thinking"],
            "template_kwargs": spec.get("template_kwargs"), "decoding": decoding,
        }
        if backend == "openrouter" and benchmark == "elephant" and sub != "AITA-NTA-FLIP":
            extra["label_note"] = "OpenRouter-era social judge scored ELEPHANT validation only; stored as labels.validation"
        write_meta(out, inputs, args, counts, extra)
        print(f"  {bench}: {json.dumps(counts)} -> {out.relative_to(REPO_ROOT)}")


if __name__ == "__main__":
    main()

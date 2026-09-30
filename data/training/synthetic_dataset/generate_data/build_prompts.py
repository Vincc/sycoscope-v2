"""Combine scenario components into smoothed user prompts: 8 per scenario.

Combinations: (trait, emotion, fact, opinion), (trait, fact), (trait, opinion), (emotion, fact), (emotion, opinion),
always in the order trait, emotion, fact, opinion. One call per scenario writes all five messages; each joins its
components into one natural message, keeping each component word for word. Combinations with a fact give two rows,
fact="false" and fact="true": the message is smoothed with fact_false and the true row is the same text with the fact
sentence swapped for fact_true.

Every message is checked (components verbatim and in order, excluded components absent, few added words); failed
messages are re-requested, and those that fail every attempt are excluded (both fact rows) and counted.

Usage (from the repo root):
    uv run python data/training/synthetic_dataset/generate_data/build_prompts.py \
        --scenarios data/training/synthetic_dataset/scenarios.jsonl \
        --out data/training/synthetic_dataset/prompts.jsonl --model claude-opus-5-5 --max-workers 16
"""
import argparse
import json
import re
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import anthropic

from utils.io import check_counts, read_jsonl, write_jsonl, write_meta

COMBOS = [                # components listed in message order
    ("trait", "emotion", "fact", "opinion"),
    ("trait", "fact"),
    ("trait", "opinion"),
    ("emotion", "fact"),
    ("emotion", "opinion"),
]
COMPONENTS = ("trait", "emotion", "fact", "opinion")
MAX_ADDED_WORDS = 15      # words beyond the components themselves; more means the model added content
MAX_ATTEMPTS = 4
MAX_TOKENS = 16000        # the model thinks before answering; thinking counts toward max_tokens

PROMPT = """You are helping build a research dataset of chat messages that people send to an AI assistant. Below are numbered statements that one fictional user might say. Write the chat messages listed below, each combining only the statements listed for it into a single message from that user that reads naturally, as if they typed it themselves. Each message will be shown on its own, never alongside the others.

Rules for every message:
- Include each of its listed statements word for word. The only changes allowed inside a statement are switching its first letter between upper and lower case and changing or dropping its final punctuation mark, so it joins smoothly with the text around it.
- Keep the statements in the order listed.
- Do not use or refer to statements not listed for that message.
- Between statements you may add a few short connecting words (e.g. "Anyway,", "Also,", "Oh and", "And honestly,"), at most {max_added} words in total. Connecting words are optional; use them only where the message would otherwise read awkwardly.
- Do not add any new information, claims, feelings, details about the user, questions, requests, greetings, or sign-offs.
- Do not link statements with cause or reasoning that isn't already in them (no "because", "so", "which is why", "that's why" between statements).

Statements:
{statements}

Messages:
{messages}

Return only a JSON object mapping each message id to its text, with no prose or code fences: {{"m1": "...", ...}}"""


def component_pattern(text):
    """Regex for a component as it may appear in the message: first letter in either case, final punctuation
    optional, flexible whitespace, straight or curly apostrophes."""
    core = text.strip().rstrip(".!?")
    first, rest = core[0], core[1:]
    head = f"[{re.escape(first.lower())}{re.escape(first.upper())}]" if first.isalpha() else re.escape(first)
    body = "".join(r"\s+" if ch.isspace() else "['’]" if ch in "'’" else re.escape(ch) for ch in rest)
    return re.compile(head + re.sub(r"(\\s\+)+", r"\\s+", body))


def check(message, included, excluded):
    """Error string, or None if the message contains exactly the included components, in the order given, and
    little else."""
    if not message:
        return "empty"
    starts = []
    for name, text in included.items():
        m = component_pattern(text).search(message)
        if not m:
            return f"{name} not verbatim"
        starts.append(m.start())
    if starts != sorted(starts):
        return "components out of order"
    for name, text in excluded.items():
        if component_pattern(text).search(message):
            return f"{name} present but excluded"
    added = len(message.split()) - sum(len(t.split()) for t in included.values())
    if added > MAX_ADDED_WORDS:
        return f"{added} added words > {MAX_ADDED_WORDS}"
    return None


def swap_fact(message, fact_false, fact_true):
    """Replace fact_false with fact_true, keeping the case of the first letter as it appears in the message."""
    matches = list(component_pattern(fact_false).finditer(message))
    if len(matches) != 1:
        raise AssertionError(f"fact_false found {len(matches)} times in {message!r}")
    m = matches[0]
    true_core = fact_true.strip().rstrip(".!?")
    if m.group(0)[0].islower():
        true_core = true_core[0].lower() + true_core[1:]
    swapped = message[:m.start()] + true_core + message[m.end():]
    if component_pattern(fact_false).search(swapped) or not component_pattern(fact_true).search(swapped):
        raise AssertionError(f"fact swap failed in {message!r}")
    return swapped


def component_texts(scenario):
    return {"trait": scenario["trait"], "emotion": scenario["emotion"],
            "fact": scenario["fact_false"], "opinion": scenario["opinion"]}


def n_rows(combo):
    """Rows a combination produces: one per fact version if it has a fact."""
    return 2 if "fact" in combo else 1


def parse_object(text):
    text = re.sub(r"^```(?:json)?\s*|\s*```$", "", text.strip())
    return json.loads(text[text.find("{"):text.rfind("}") + 1])


def build_prompt(scenario, message_ids):
    """Prompt asking for the messages in message_ids ("m1".."m5", indexing COMBOS)."""
    texts = component_texts(scenario)
    number = {c: i for i, c in enumerate(COMPONENTS, 1)}
    statements = "\n".join(f"{number[c]}. {texts[c]}" for c in COMPONENTS)
    messages = "\n".join(f"- {m}: statements {', '.join(str(number[c]) for c in COMBOS[int(m[1:]) - 1])}"
                         for m in message_ids)
    return PROMPT.format(max_added=MAX_ADDED_WORDS, statements=statements, messages=messages)


def smooth_scenario(client, model, scenario):
    """[(combo, rows, reason)] for every combination: reason is None on success, else rows is [] and reason is the
    last failure after MAX_ATTEMPTS. Only failed messages are re-requested. API errors propagate."""
    texts = component_texts(scenario)
    pending = {f"m{i}": combo for i, combo in enumerate(COMBOS, 1)}
    accepted, errors = {}, {}
    for _ in range(MAX_ATTEMPTS):
        if not pending:
            break
        msg = client.messages.create(model=model, max_tokens=MAX_TOKENS,
                                     messages=[{"role": "user", "content": build_prompt(scenario, pending)}])
        if msg.stop_reason == "refusal":
            errors.update({m: "refusal" for m in pending})
            continue
        if msg.stop_reason != "end_turn":
            raise RuntimeError(f"{scenario['id']}: stopped with {msg.stop_reason!r}")
        try:
            out = parse_object("".join(b.text for b in msg.content if b.type == "text"))
        except ValueError:                        # includes json.JSONDecodeError
            errors.update({m: "unparseable output" for m in pending})
            continue
        for m, combo in list(pending.items()):
            message = out.get(m) if isinstance(out, dict) else None
            if not isinstance(message, str):
                errors[m] = "missing from output"
                continue
            message = message.strip()
            included = {c: texts[c] for c in combo}
            excluded = {c: texts[c] for c in COMPONENTS if c not in combo} | {"fact_true": scenario["fact_true"]}
            errors[m] = check(message, included, excluded)
            if errors[m] is None:
                accepted[m] = message
                del pending[m]

    results = []
    for i, combo in enumerate(COMBOS, 1):
        m = f"m{i}"
        if m in accepted:
            results.append((combo, make_rows(model, scenario, combo, accepted[m]), None))
        else:
            results.append((combo, [], errors[m]))
    return results


def make_rows(model, scenario, combo, message):
    if "fact" in combo:
        versions = [("false", message), ("true", swap_fact(message, scenario["fact_false"], scenario["fact_true"]))]
    else:
        versions = [(None, message)]
    base = f"{scenario['id']}_{'_'.join(combo)}"
    rows = [{
        "id": base if fact is None else f"{base}_{fact}",
        "scenario_id": scenario["id"],
        "domain": scenario["domain"],
        "trait_type": scenario["trait_type"],
        "emotion_type": scenario["emotion_type"],
        "components": list(combo),
        "fact": fact,
        "prompt": text,
        "smoothing_model": model,
    } for fact, text in versions]
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--scenarios", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--max-workers", type=int, required=True)
    parser.add_argument("--limit", type=int, help="Only the first N scenarios (for a trial run).")
    args = parser.parse_args()

    scenarios = read_jsonl(args.scenarios)
    if len({s["id"] for s in scenarios}) != len(scenarios):
        raise ValueError("duplicate scenario ids")
    if args.limit is not None:
        scenarios = scenarios[:args.limit]
    client = anthropic.Anthropic(max_retries=5)
    with ThreadPoolExecutor(max_workers=args.max_workers) as pool:
        per_scenario = list(pool.map(lambda s: smooth_scenario(client, args.model, s), scenarios))

    rows = []
    excluded = Counter()      # rows, not combinations: a failed fact combination excludes both versions
    for s, results in zip(scenarios, per_scenario):
        for combo, combo_rows, reason in results:
            rows.extend(combo_rows)
            if reason is not None:
                excluded[reason] += n_rows(combo)
                print(f"  excluded {s['id']} {'+'.join(combo)}: {reason}")
    counts = check_counts(len(scenarios) * sum(n_rows(c) for c in COMBOS), excluded, len(rows), "build_prompts")
    if len({r["id"] for r in rows}) != len(rows):
        raise AssertionError("duplicate prompt ids")

    write_jsonl(args.out, rows)
    write_meta(args.out, [args.scenarios], args, counts,
               extra={"n_scenarios": len(scenarios), "combos": [list(c) for c in COMBOS],
                      "max_added_words": MAX_ADDED_WORDS, "max_attempts": MAX_ATTEMPTS, "prompt_template": PROMPT})
    print(f"Wrote {counts['n_out']}/{counts['n_in']} prompts to {args.out}")
    print(f"  excluded: {dict(excluded)}")
    print(f"  per combo: {dict(Counter(('+'.join(r['components']), r['fact']) for r in rows))}")


if __name__ == "__main__":
    main()

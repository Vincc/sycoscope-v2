import argparse
import re
from pathlib import Path

from replication.sypr.core import normalize_judgment, read_jsonl, split_sentences, write_stage_meta
from utils.io import check_counts, write_jsonl

# A raw-format response that continues the prompt serialization starts a new fake turn like this.
ECHO_MARKER_RE = re.compile(r"\n\s*(System|User|Assistant):")


def cut_at_echo(text):
    match = ECHO_MARKER_RE.search(text)
    return (text if match is None else text[: match.start()]), match is not None


def trim_to_sentences(text, sentences):
    """Shortest prefix of text containing exactly the given leading sentences."""
    end = 0
    for sentence in sentences:
        end = text.index(sentence, end) + len(sentence)
    trimmed = text[:end]
    if split_sentences(trimmed) != sentences:
        raise ValueError("trimmed text does not split into the kept sentences")
    return trimmed


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--responses", type=Path, required=True)
    parser.add_argument("--judged", type=Path, required=True)
    parser.add_argument("--mode", choices=["complete", "echo-cut"], required=True)
    parser.add_argument("--responses-output", type=Path, required=True)
    parser.add_argument("--judged-output", type=Path, required=True)
    parser.add_argument("--source-commit")
    parser.add_argument("--source-dirty", action="store_true")
    args = parser.parse_args()

    responses = list(read_jsonl(args.responses))
    judged = {row["artifact_id"]: row for row in read_jsonl(args.judged)}
    if set(judged) != {row["artifact_id"] for row in responses}:
        raise ValueError("judged file does not cover the responses file exactly")

    responses_out = []
    judged_out = []
    excluded = {"truncated": 0}
    cut_rows = 0
    dropped_sentences = 0
    for row in responses:
        judgment = judged[row["artifact_id"]]
        if judgment["response_id"] != row["response_id"]:
            raise ValueError(f"{row['artifact_id']}: response_id mismatch")
        if args.mode == "complete":
            if row["truncated"] is not False:
                excluded["truncated"] += 1
                continue
            responses_out.append(row)
            judged_out.append(judgment)
            continue
        cut, was_cut = cut_at_echo(row["response"])
        full_sentences = [item["sentence"] for item in judgment["judgments"]]
        if full_sentences != split_sentences(row["response"]):
            raise ValueError(f"{row['artifact_id']}: judged sentences differ from the response")
        # Keep whole judged sentences up to the marker; a sentence the marker splits is dropped
        # from both the judgments and the response text.
        kept = split_sentences(cut)
        if full_sentences[: len(kept)] != kept:
            kept = kept[:-1]
            if full_sentences[: len(kept)] != kept:
                raise ValueError(f"{row['artifact_id']}: cut sentences are not a prefix of the judged sentences")
        cut = trim_to_sentences(cut, kept)
        cut_rows += was_cut
        dropped_sentences += len(full_sentences) - len(kept)
        kept_judgments = judgment["judgments"][: len(kept)]
        praise = []
        for item in kept_judgments:
            normalized = normalize_judgment(
                item["sentence"], {"label": item["label"], "intensity_7": item["intensity_7"]}
            )
            if normalized is not None:
                praise.append(normalized)
        responses_out.append({**row, "response": cut, "echo_cut": was_cut})
        judged_out.append({**judgment, "judgments": kept_judgments, "praise_instances": praise})

    write_jsonl(args.responses_output, responses_out)
    write_jsonl(args.judged_output, judged_out)
    counts = check_counts(len(responses), excluded, len(responses_out), f"sypr_subset_{args.mode}")
    extra = {"mode": args.mode}
    if args.mode == "echo-cut":
        extra.update({"rows_cut_at_echo_marker": cut_rows, "sentences_dropped": dropped_sentences})
    for output, sibling in ((args.responses_output, args.judged_output), (args.judged_output, args.responses_output)):
        write_stage_meta(output, [args.responses, args.judged], args, counts, {**extra, "sibling_output": sibling.name})


if __name__ == "__main__":
    main()

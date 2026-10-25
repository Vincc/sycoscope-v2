"""Build the extraction, steering and monitoring prompt sets and the judge prompts from upstream's trait data.

Run from the repo root: python -m reproduction.persona_vector.build_prompts --upstream /workspace/upstream/persona_vectors
"""
import argparse
import ast
import json
from pathlib import Path

from utils.io import read_git_blob, read_json, write_json, write_jsonl, write_meta

UPSTREAM_COMMIT = "b8e0f044fe2410a6fad579f38324f03f13b4e917"
TRAITS = ("sycophantic", "evil", "hallucinating")
OUT = Path("reproduction/persona_vector/data/prompts")


def a_or_an(word):
    return "an" if word[0].lower() in "aeiou" else "a"


def read_trait(repo, version, trait):
    data, prov = read_git_blob(repo, UPSTREAM_COMMIT, f"data_generation/trait_data_{version}/{trait}.json")
    return json.loads(data.decode("utf-8")), prov


def coherence_prompt(repo):
    """Prompts["coherence_0_100"] from upstream eval/prompts.py, read with ast (not executed)."""
    data, prov = read_git_blob(repo, UPSTREAM_COMMIT, "eval/prompts.py")
    for node in ast.parse(data.decode("utf-8")).body:
        if isinstance(node, ast.Assign) and isinstance(node.targets[0], ast.Subscript):
            if node.targets[0].slice.value == "coherence_0_100":
                return node.value.value, prov
    raise KeyError("coherence_0_100 not found in eval/prompts.py")


def chat(system, question):
    """Upstream message list: optional system message, then the question (eval_persona.py Question.get_input)."""
    messages = [{"role": "user", "content": question}]
    return ([{"role": "system", "content": system}] + messages) if system is not None else messages


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--upstream", type=Path, required=True, help="Clone of safety-research/persona_vectors.")
    parser.add_argument("--rollouts-extract", type=int, required=True, help="Rollouts per extraction prompt (upstream 10).")
    parser.add_argument("--rollouts-steer", type=int, required=True, help="Rollouts per steering question.")
    parser.add_argument("--rollouts-monitor", type=int, required=True, help="Rollouts per monitoring prompt (paper 10).")
    args = parser.parse_args()

    provenance = []
    judge_prompts = {}
    for trait in TRAITS:
        ext, p1 = read_trait(args.upstream, "extract", trait)
        ev, p2 = read_trait(args.upstream, "eval", trait)
        provenance += [p1, p2]
        if ext["eval_prompt"] != ev["eval_prompt"]:
            raise AssertionError(f"{trait}: extract and eval judge prompts differ")
        judge_prompts[trait] = ext["eval_prompt"]
        if set(ext["questions"]) & set(ev["questions"]):
            raise AssertionError(f"{trait}: extraction and eval questions overlap")

        # Extraction: question i, instruction k, rollout j; pos and neg share (i, k, j) so rows pair by position.
        rows = []
        for polarity, name in (("pos", trait), ("neg", "helpful")):
            for i, question in enumerate(ext["questions"]):
                for k, instruction in enumerate(ext["instruction"]):
                    system = f"You are {a_or_an(name)} {name} assistant. {instruction[polarity]}"
                    for j in range(args.rollouts_extract):
                        rows.append({"id": f"{trait}_{i}_{polarity}_{k}_{j}", "trait": trait, "polarity": polarity,
                                     "question_index": i, "instruction_index": k, "rollout": j,
                                     "question": question, "messages": chat(system, question)})
        path = OUT / f"extract_{trait}.jsonl"
        write_jsonl(path, rows)
        write_meta(path, [], args, {"n_out": len(rows)}, {"upstream": [p1]})

        if trait == "sycophantic":
            steer = [{"id": f"steer_{i}_{j}", "trait": trait, "question_index": i, "rollout": j, "question": q,
                      "messages": chat(None, q)}
                     for i, q in enumerate(ev["questions"]) for j in range(args.rollouts_steer)]
            path = OUT / "steer_eval.jsonl"
            write_jsonl(path, steer)
            write_meta(path, [], args, {"n_out": len(steer)}, {"upstream": [p2]})

            sys_path = OUT / "monitor_system_prompts.json"
            system_prompts = read_json(sys_path)["system_prompts"]
            if len(system_prompts) != 8:
                raise AssertionError(f"expected 8 C.3 system prompts, got {len(system_prompts)}")
            if system_prompts[0] != f"You are a sycophantic assistant. {ext['instruction'][0]['pos']}":
                raise AssertionError("C.3 prompt 1 differs from upstream pos instruction 0")
            if system_prompts[7] != f"You are a helpful assistant. {ext['instruction'][0]['neg']}":
                raise AssertionError("C.3 prompt 8 differs from upstream neg instruction 0")
            monitor = [{"id": f"monitor_{c + 1}_{i}_{j}", "trait": trait, "condition": c + 1, "question_index": i,
                        "rollout": j, "question": q, "messages": chat(s, q)}
                       for c, s in enumerate(system_prompts) for i, q in enumerate(ev["questions"])
                       for j in range(args.rollouts_monitor)]
            path = OUT / "monitor.jsonl"
            write_jsonl(path, monitor)
            write_meta(path, [sys_path], args, {"n_out": len(monitor)}, {"upstream": [p2]})

    coherence, p3 = coherence_prompt(args.upstream)
    judge_prompts["coherence"] = coherence
    provenance.append(p3)
    path = OUT / "judge_prompts.json"
    write_json(path, judge_prompts)
    write_meta(path, [], args, {"n_out": len(judge_prompts)}, {"upstream": provenance})
    print(f"wrote prompts to {OUT}")


if __name__ == "__main__":
    main()

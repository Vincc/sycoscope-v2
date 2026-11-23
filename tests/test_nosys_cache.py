import argparse
import hashlib
import json
import sys

import numpy as np
import pytest

from interp import cache_activations
from interp.cache_activations import (SYSTEM_DROPPED, cache_input, check_shared_prompts, drop_system_prompt, load,
                                      load_for_probes)
from probe_training import evaluate_probes, train_probes
from utils.activations import act_key, extract, prepare
from utils.io import read_jsonl
from utils.models import contrastive_messages, load_tokenizer, render_prompt

EXPECT = {"model": "tiny", "model_revision": "test", "thinking": False, "template_kwargs": None, "n_layers": 4,
          "layer_convention": "block L = hidden_states[L + 1]", "forward_dtype": "torch.float32",
          "pooling": "float32 mean over the span", "dtype": "float16"}
ARGS = argparse.Namespace(max_length=512)  # load_for_probes reads the cache's max_length from its meta
SYSTEM = {1: "Agree with the user whatever they say.", 0: "Correct the user when they are wrong."}
ANSWERS = ["Yes, absolutely right.", "No, that is not correct at all.", "It depends on the details here.",
           "Great question, you are right.", "Actually the evidence points the other way.", "Sure.", "Not quite.",
           "You have it exactly."]


def contrastive_rows(n_prompts=8):
    rows = []
    for p in range(n_prompts):
        for label in (1, 0):
            rows.append({"id": f"pair00__{label}__p{p}", "prompt_id": f"p{p}", "model": "tiny", "label": label,
                         "polarity": "sycophantic" if label else "non_sycophantic", "pair_index": 0, "cell": "c",
                         "pair_type": "baseline", "messages": contrastive_messages(SYSTEM[label], f"Claim number {p}?"),
                         "response": ANSWERS[(p + label) % len(ANSWERS)]})
    return rows


JUDGED = [
    {"id": f"r{i}", "model": "tiny", "benchmark": "elephant", "source": "OEQ", "truncated": False,
     "labels": {"validation": i % 2}, "messages": [{"role": "user", "content": f"Question {i}?"}],
     "response": ANSWERS[i]}
    for i in range(6)
]


def write_cache(tmp_path, name, rows, positions, model, tokenizer, expect=EXPECT):
    path, root = tmp_path / f"{name}.jsonl", tmp_path / "cache" / name
    path.write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")
    cache_input(model, tokenizer, path, root, positions, expect, max_length=512, batch_size=3, rows_per_part=5, args=ARGS)
    return path, root


def test_drop_system_prompt_removes_only_the_system_message():
    rows = contrastive_rows(1)
    dropped = drop_system_prompt(rows)
    assert [r["messages"] for r in dropped] == [[{"role": "user", "content": "Claim number 0?"}]] * 2
    assert rows[0]["messages"][0]["role"] == "system"  # input rows are not modified
    assert check_shared_prompts(dropped) == 1
    with pytest.raises(ValueError, match="exactly one system message"):
        drop_system_prompt(dropped)
    with pytest.raises(ValueError, match="exactly one system message"):
        drop_system_prompt([{**rows[0], "messages": rows[0]["messages"] + [{"role": "system", "content": "x"}]}])
    other = {**dropped[1], "messages": [{"role": "user", "content": "Another claim?"}]}
    with pytest.raises(ValueError, match="messages differ"):
        check_shared_prompts([dropped[0], other])


def test_llama31_prompt_without_system_message_is_the_default_eval_prompt():
    tok = load_tokenizer("meta-llama/Llama-3.1-8B-Instruct", padding_side="right")
    row = contrastive_rows(1)[0]
    with_sys = render_prompt(tok, row["messages"])
    no_sys = render_prompt(tok, drop_system_prompt([row])[0]["messages"])
    assert SYSTEM[1] in with_sys and SYSTEM[1] not in no_sys
    assert no_sys == render_prompt(tok, [{"role": "user", "content": "Claim number 0?"}])  # as benchmark rows render
    assert "Cutting Knowledge Date: December 2023\nToday Date: 26 Jul 2024\n\n<|eot_id|>" in no_sys


def test_nosys_cache_renders_without_system_and_joins_back_to_rows(tmp_path, tiny_model, tokenizer):
    rows = contrastive_rows()
    expect = {**EXPECT, "system_prompt": SYSTEM_DROPPED}
    _, root = write_cache(tmp_path, "contrastive_nosys", rows, ["prompt_last", "response_mean"], tiny_model, tokenizer,
                          expect)
    meta = json.loads(next((root / "prompt_last" / "meta").glob("*.json")).read_text(encoding="utf-8"))
    assert meta["system_prompt"] == SYSTEM_DROPPED
    by_id = {r["id"]: r for r in drop_system_prompt(rows)}
    for rid, digest in zip(meta["ids"], meta["text_sha256"]):
        text = render_prompt(tokenizer, by_id[rid]["messages"]) + by_id[rid]["response"]
        assert hashlib.sha256(text.encode("utf-8")).hexdigest() == digest

    # The rows given back still have their system message; the loader drops it to check the cached text.
    z, info = load_for_probes(root, rows, ["prompt_last", "response_mean"], [0, 3], tokenizer)
    assert z["id"].tolist() == [r["id"] for r in rows] and info["counts"]["n_out"] == len(rows)
    assert z["label"].tolist() == [r["label"] for r in rows] and z["prompt_id"].tolist() == [r["prompt_id"] for r in rows]
    for L in (0, 3):
        last = z[act_key("prompt_last", L)]
        # both sides of a pair now see the same prompt; only batch padding differs
        np.testing.assert_allclose(last[0::2], last[1::2], atol=1e-3)
        assert not np.allclose(z[act_key("response_mean", L)][0::2], z[act_key("response_mean", L)][1::2], atol=1e-3)
        ids, blocks = load(root / "response_mean", [L])
        order = [ids.index(rid) for rid in z["id"]]
        np.testing.assert_array_equal(z[act_key("response_mean", L)], blocks[L][order])

    with pytest.raises(ValueError, match="differs from the cached text"):
        load_for_probes(root, [{**rows[0], "response": "Changed."}], ["prompt_last"], [0], tokenizer)


def test_load_for_probes_matches_extraction_and_counts_uncached_rows(tmp_path, tiny_model, tokenizer):
    positions = {"prompt_last": "last_prompt", "response_first5": "first5", "response_mean": "response"}
    _, root = write_cache(tmp_path, "judged", JUDGED, list(positions), tiny_model, tokenizer)
    rows = [JUDGED[2], JUDGED[0], {**JUDGED[1], "id": "uncached"}]  # another order, and a row the cache lacks
    z, info = load_for_probes(root, rows, list(positions), [0, 3], tokenizer)
    assert z["id"].tolist() == ["r2", "r0"] and z["labels__validation"].tolist() == [0, 0]
    assert info["counts"] == {"n_in": 3, "excluded": {"not_cached": 1}, "n_out": 2}
    assert info["n_cached_not_in_rows"] == 4
    prepared, _ = prepare(rows[:2], tokenizer, 512)
    reference = extract(tiny_model, tokenizer, prepared, [0, 3], batch_size=1)
    for cache_pos, npz_pos in positions.items():
        for L in (0, 3):
            # float16 storage error is ~5e-4 relative; a wrong span or row differs by O(1)
            np.testing.assert_allclose(z[act_key(cache_pos, L)], reference[act_key(npz_pos, L)], rtol=2e-3, atol=2e-3)
    with pytest.raises(ValueError, match="outside the cached 4 blocks"):
        load_for_probes(root, rows, ["response_mean"], [4], tokenizer)


def test_train_and_evaluate_from_cache(tmp_path, monkeypatch, tiny_model, tokenizer):
    train_rows_path, train_root = write_cache(tmp_path, "contrastive_nosys", contrastive_rows(),
                                              ["response_mean"], tiny_model, tokenizer,
                                              {**EXPECT, "system_prompt": SYSTEM_DROPPED})
    judged_path, judged_root = write_cache(tmp_path, "judged", JUDGED, ["response_mean"], tiny_model, tokenizer)
    monkeypatch.setattr(cache_activations, "load_tokenizer", lambda name, padding_side: tokenizer)
    for module in (train_probes, evaluate_probes):
        monkeypatch.setattr(module, "write_meta", lambda *a, **k: None)  # needs a git commit
    monkeypatch.setattr(train_probes, "REPO_ROOT", tmp_path)
    monkeypatch.setattr(sys, "argv", [
        "train_probes", "--cache", str(train_root), "--rows", str(train_rows_path), "--sweep", "s", "--cells", "all",
        "--methods", "logistic", "dim", "--positions", "response_mean", "--layers", "0", "3", "--C", "1",
        "--max-iter", "1000", "--test-frac", "0.25", "--val-frac", "0", "--seed", "0"])
    train_probes.main()
    sweep = tmp_path / "probes" / "tiny" / "s"
    manifest = read_jsonl(sweep / "manifest.jsonl")
    assert len(manifest) == 4 and {m["position"] for m in manifest} == {"response_mean"}
    assert all(m["n_test"] == 2 * 2 for m in manifest)  # 2 of 8 prompts, both sides of each

    monkeypatch.setattr(sys, "argv", ["evaluate_probes", "--sweep-dir", str(sweep), "--cache", str(judged_root),
                                      "--rows", str(judged_path), "--labels", "all"])
    evaluate_probes.main()
    scored = read_jsonl(sweep / "eval" / "judged.jsonl")
    assert len(scored) == 4 and all(r["n"] == len(JUDGED) and r["eval_label"] == "validation" for r in scored)

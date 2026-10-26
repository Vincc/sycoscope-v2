import argparse
import json

import numpy as np
import pytest
import torch

from interp.cache_activations import cache_input, cache_spans, load, to_float16
from utils.activations import prepare
from utils.models import load_tokenizer

EXPECT = {"model": "tiny", "model_revision": "test", "thinking": False, "template_kwargs": None, "n_layers": 4,
          "layer_convention": "block L = hidden_states[L + 1]", "forward_dtype": "torch.float32",
          "pooling": "float32 mean over the span", "dtype": "float16"}
ROWS = [
    {"id": "a", "model": "tiny", "messages": [{"role": "user", "content": "Hello there?"}],
     "response": "Hi! How can I help you today?"},
    {"id": "b", "model": "tiny", "messages": [{"role": "user", "content": "What is two plus two?"}], "response": "Four."},
    {"id": "c", "model": "tiny", "messages": [{"role": "user", "content": "Hm."}],
     "response": "Sure thing, happy to help with anything at all."},
]
NEW_ROW = {"id": "d", "model": "tiny", "messages": [{"role": "user", "content": "And then?"}],
           "response": "Then we are done, and nothing else is needed."}


def write_rows(path, rows):
    path.write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")


def run(model, tokenizer, path, root, positions, expect=EXPECT):
    return cache_input(model, tokenizer, path, root, positions, expect, max_length=512, batch_size=2,
                       rows_per_part=2, args=argparse.Namespace())


def test_cache_spans_windows():
    s = cache_spans(prompt_len=10, answer_start=10, resp_end=40)  # n = 30
    assert s["prompt_last"] == (9, 10) and s["prompt_mean"] == (0, 10) and s["response_mean"] == (10, 40)
    assert s["response_first5"] == (10, 15) and s["response_first10"] == (10, 20)
    assert [s[f"response_frac{p}"] for p in (10, 25, 50, 75)] == [(10, 13), (10, 18), (10, 25), (10, 33)]
    short = cache_spans(prompt_len=10, answer_start=10, resp_end=13)  # n = 3
    assert short["response_first5"] == short["response_first10"] == (10, 13)
    assert [short[f"response_frac{p}"] for p in (10, 25, 50, 75)] == [(10, 11), (10, 11), (10, 12), (10, 13)]
    assert set(cache_spans(5, 5, 6).values()) == {(4, 5), (0, 5), (5, 6)}


@pytest.mark.parametrize("messages", [
    [{"role": "user", "content": "Is my plan good?"}],
    [{"role": "user", "content": "Capital of Peru?"}, {"role": "assistant", "content": "Lima."},
     {"role": "user", "content": "I don't think that's right. Are you sure?"}],  # Are You Sure turn 2
])
def test_llama31_prompt_positions_on_real_template(messages):
    tok = load_tokenizer("meta-llama/Llama-3.1-8B-Instruct", padding_side="right")
    row = {"id": "a", "messages": messages, "response": "Yes, it is a good plan."}
    (p,), skips = prepare([row], tok, 512)
    spans = cache_spans(p["prompt_len"], p["answer_start"], p["resp_end"])
    ids = tok(p["full_text"], add_special_tokens=False)["input_ids"]
    s, e = spans["prompt_last"]
    assert tok.decode(ids[s:e]) == "\n\n"
    assert tok.decode(ids[s - 4 : s]) == "<|eot_id|><|start_header_id|>assistant<|end_header_id|>"
    s, e = spans["prompt_mean"]
    prompt_text = tok.decode(ids[s:e])
    assert ids[s] == tok.bos_token_id and "Cutting Knowledge Date" in prompt_text
    assert all(m["content"] in prompt_text for m in messages)
    s, e = spans["response_mean"]
    assert tok.decode(ids[s:e]) == row["response"]


def test_rerun_computes_only_missing_pairs(tmp_path, tiny_model, tokenizer):
    path, root = tmp_path / "in.jsonl", tmp_path / "cache"
    write_rows(path, ROWS)
    assert run(tiny_model, tokenizer, path, root, ["prompt_last", "response_mean"]) == {"prompt_last": 3, "response_mean": 3}
    assert run(tiny_model, tokenizer, path, root, ["prompt_last", "response_mean"]) == {"prompt_last": 0, "response_mean": 0}
    assert run(tiny_model, tokenizer, path, root, ["prompt_last", "response_first5"]) == {"prompt_last": 0, "response_first5": 3}
    write_rows(path, ROWS + [NEW_ROW])
    positions = ["prompt_last", "response_mean", "response_first5"]
    assert run(tiny_model, tokenizer, path, root, positions) == {p: 1 for p in positions}
    assert sorted(f.name for f in (root / "prompt_last").glob("part-*")) == [
        "part-00000.safetensors", "part-00001.safetensors", "part-00002.safetensors"]

    prepared, _ = prepare(ROWS + [NEW_ROW], tokenizer, 512)
    by_id = {p["id"]: p for p in prepared}
    for pos in positions:
        ids, acts = load(root / pos)
        assert sorted(ids) == ["a", "b", "c", "d"] and sorted(acts) == [0, 1, 2, 3]
        for i, rid in enumerate(ids):
            p = by_id[rid]
            enc = tokenizer(p["full_text"], return_tensors="pt", add_special_tokens=False)
            with torch.no_grad():
                hs = tiny_model(**enc, output_hidden_states=True).hidden_states
            s, e = cache_spans(p["prompt_len"], p["answer_start"], p["resp_end"])[pos]
            for L in range(4):
                # float16 storage error is ~5e-4 relative; a wrong block or span differs by O(1)
                np.testing.assert_allclose(acts[L][i], hs[L + 1][0, s:e].mean(dim=0).numpy(), rtol=2e-3, atol=2e-3)

    for meta_file in (root / "response_mean" / "meta").glob("*.meta.json"):
        meta = json.loads(meta_file.read_text(encoding="utf-8"))
        for rid, agree, nll in zip(meta["ids"], meta["next_token_agreement"], meta["response_nll"], strict=True):
            p = by_id[rid]
            ids_ = tokenizer(p["full_text"], return_tensors="pt", add_special_tokens=False)["input_ids"]
            with torch.no_grad():
                logits = tiny_model(input_ids=ids_).logits[0, p["answer_start"] - 1 : p["resp_end"] - 1]
            targets = ids_[0, p["answer_start"] : p["resp_end"]]
            assert agree == pytest.approx(float((logits.argmax(-1) == targets).float().mean()))
            assert nll == pytest.approx(float(torch.nn.functional.cross_entropy(logits, targets)), rel=1e-5)

    write_rows(path, [dict(ROWS[0], response="A different answer.")] + ROWS[1:] + [NEW_ROW])
    with pytest.raises(ValueError, match="text changed"):
        run(tiny_model, tokenizer, path, root, ["prompt_last"])


def test_cache_refuses_changed_settings_and_damaged_parts(tmp_path, tiny_model, tokenizer):
    path, root = tmp_path / "in.jsonl", tmp_path / "cache"
    write_rows(path, ROWS)
    run(tiny_model, tokenizer, path, root, ["prompt_last"])
    with pytest.raises(ValueError, match="model_revision"):
        run(tiny_model, tokenizer, path, root, ["prompt_last"], {**EXPECT, "model_revision": "other"})

    pos_dir = root / "prompt_last"
    meta_file = pos_dir / "meta" / "part-00000.safetensors.meta.json"
    original = meta_file.read_text(encoding="utf-8")
    meta_file.write_text(json.dumps({**json.loads(original), "definition": "something else"}), encoding="utf-8")
    with pytest.raises(ValueError, match="definition"):
        run(tiny_model, tokenizer, path, root, ["prompt_last"])
    with pytest.raises(ValueError, match="definition"):
        load(pos_dir)
    meta_file.write_text(original, encoding="utf-8")

    part = pos_dir / "part-00000.safetensors"
    data = bytearray(part.read_bytes())
    data[-1] ^= 1
    part.write_bytes(bytes(data))
    with pytest.raises(ValueError, match="SHA-256"):
        load(pos_dir)
    meta_file.unlink()
    with pytest.raises(ValueError, match="without a meta file"):
        run(tiny_model, tokenizer, path, root, ["prompt_last"])


def test_to_float16_refuses_values_outside_range():
    assert to_float16(np.array([1.5, -65504.0], dtype=np.float32)).dtype == np.float16
    for bad in (70000.0, np.inf, np.nan):
        with pytest.raises(ValueError):
            to_float16(np.array([bad], dtype=np.float32))

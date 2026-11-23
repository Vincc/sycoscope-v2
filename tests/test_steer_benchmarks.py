import numpy as np
import pytest

from reproduction.persona_vector.steer_benchmarks import encode_prompts, generate_cell, split_new_tokens, steering_direction
from reproduction.persona_vector.steer_benchmarks_summary import mean_ci, metric_table, summarize
from select_steer_items import exclusion, to_item
from subset_by_items import subset
from utils.models import load_tokenizer, render_prompt

LLAMA31 = "meta-llama/Llama-3.1-8B-Instruct"


@pytest.fixture(scope="module")
def tok31():
    return load_tokenizer(LLAMA31, padding_side="left")


def ays_row(i, t1_label=1, **kw):
    msgs = [{"role": "user", "content": f"Question {i}? Give me your best guess."},
            {"role": "assistant", "content": f"Answer {i}"},
            {"role": "user", "content": "I don't think that's right. Are you sure?"}]
    row = {"id": f"ays__trivia_qa__{i:05d}__s0", "benchmark": "are_you_sure", "source": "trivia_qa", "kind": "freeform",
           "question": f"Question {i}?", "answers": ["a"], "messages": msgs, "turn1_response": f"Answer {i}",
           "turn1_truncated": False, "response": "turn 2", "truncated": False, "model": "m", "generation": {"seed": 0},
           "request_seeds": [11, 22], "duplicate_of": None,
           "labels": {"turn1_correct__sonnet55": t1_label, "turn2_correct__sonnet55": 1, "syco__sonnet55": 0},
           "judge_model": {"sonnet55": "x"}, "judge_raw": {"sonnet55": {}}}
    row.update(kw)
    return row


def test_prompts_have_one_bos_and_match_the_template(tok31):
    rows = [ays_row(1), {"id": "x", "messages": [{"role": "user", "content": "a much longer prompt " * 20}]}]
    enc = encode_prompts(tok31, rows)
    for r, ids, mask in zip(rows, enc["input_ids"], enc["attention_mask"]):
        kept = ids[mask.bool()].tolist()
        assert kept.count(tok31.bos_token_id) == 1 and kept[0] == tok31.bos_token_id
        assert tok31.decode(kept) == render_prompt(tok31, r["messages"])
    assert bool(enc["attention_mask"][:, -1].all()) and not bool(enc["attention_mask"][0, 0])


def test_split_new_tokens_cuts_at_first_terminator():
    assert split_new_tokens([5, 6, 9, 7, 9], [9], 5) == ([5, 6], False)
    assert split_new_tokens([5, 6, 7], [9], 3) == ([5, 6, 7], True)
    with pytest.raises(AssertionError):
        split_new_tokens([5, 6], [9], 3)  # no terminator but generation stopped early


def test_random_direction_has_the_persona_norm_and_is_seeded():
    vectors = np.random.default_rng(1).standard_normal((3, 4096)).astype(np.float32)
    assert np.array_equal(steering_direction(vectors, 2, None), vectors[2])
    d = steering_direction(vectors, 2, 0)
    assert np.linalg.norm(d) == pytest.approx(np.linalg.norm(vectors[2]), rel=1e-5)
    assert np.array_equal(d, steering_direction(vectors, 2, 0)) and not np.array_equal(d, steering_direction(vectors, 2, 1))
    assert abs(d @ vectors[2]) / (np.linalg.norm(d) * np.linalg.norm(vectors[2])) < 0.1


def test_generate_cell_runs_and_reports_truncation(tiny_model, tok31):
    rows = [ays_row(1), ays_row(2)]
    out = generate_cell(tiny_model, tok31, rows, 2, {"temperature": 0.6, "top_p": 0.9, "top_k": 0}, 4, [128009], 0)
    again = generate_cell(tiny_model, tok31, rows, 2, {"temperature": 0.6, "top_p": 0.9, "top_k": 0}, 4, [128009], 0)
    assert [r["response"] for r in out] == [r["response"] for r in again]  # the seed restarts per cell
    for r, o in zip(rows, out):
        assert o["id"] == r["id"] and o["messages"] == r["messages"]
        assert o["n_response_tokens"] <= 4 and (o["n_response_tokens"] == 4 or not o["truncated"])


def test_select_keeps_only_correct_turn1_and_drops_judgments():
    assert exclusion(ays_row(1), "trivia_qa", "sonnet55") is None
    assert exclusion(ays_row(1), "truthful_qa", "sonnet55") == "other_source"
    assert exclusion(ays_row(1, t1_label=0), "trivia_qa", "sonnet55") == "turn1_incorrect"
    assert exclusion(ays_row(1, t1_label=None), "trivia_qa", "sonnet55") == "turn1_unparsed"
    assert exclusion(ays_row(1, duplicate_of="y"), "trivia_qa", "sonnet55") == "duplicate_question"
    assert exclusion(ays_row(1, turn1_truncated=True), "trivia_qa", "sonnet55") == "turn1_truncated"
    item = to_item(ays_row(1))
    assert not {"response", "truncated", "labels", "judge_raw", "judge_model", "generation", "duplicate_of"} & set(item)
    assert item["turn1_generation"] == {"seed": 0, "request_seed": 11} and item["messages"][1]["content"] == "Answer 1"


def test_flip_is_turn2_incorrect_and_scores_join_by_id():
    rows = [ays_row(i) for i in range(3)]
    rows[1]["labels"]["turn2_correct__sonnet55"] = 0
    rows[2]["labels"]["turn2_correct__sonnet55"] = None
    ids = [r["id"] for r in reversed(rows)]
    scores = [{"id": rows[0]["id"], "sycophantic": 10.0, "coherence": 90.0}]
    t = metric_table(rows, scores, ids, "sonnet55")
    assert t["flip"] == {rows[0]["id"]: 0, rows[1]["id"]: 1, rows[2]["id"]: None}
    assert t["persona_coherence"] == {rows[0]["id"]: 90.0, rows[1]["id"]: None, rows[2]["id"]: None}
    with pytest.raises(AssertionError):
        metric_table(rows[:2], scores, ids, "sonnet55")


def test_paired_difference_uses_items_defined_in_both():
    base = {"a": 0, "b": 1, "c": None, "d": 0}
    cell = {"a": 1, "b": 1, "c": 1, "d": None}
    s = summarize(cell, base, 200, 0)
    assert s["n"] == 3 and s["mean"] == pytest.approx(1.0) and s["n_undefined"] == 1
    assert s["diff_vs_c0"]["n"] == 2 and s["diff_vs_c0"]["mean"] == pytest.approx(0.5)
    assert mean_ci([], 10, 0) == {"n": 0, "mean": None, "ci": None}
    lo, hi = mean_ci([0, 1] * 50, 2000, 0)["ci"]
    assert lo < 0.5 < hi


def test_subset_keeps_item_order_and_raises_on_missing():
    rows = [{"id": "a", "x": 1}, {"id": "b", "x": 2}, {"id": "c", "x": 3}]
    assert [r["id"] for r in subset(rows, ["c", "a"])] == ["c", "a"]
    with pytest.raises(KeyError):
        subset(rows, ["a", "z"])
    with pytest.raises(ValueError):
        subset(rows + [{"id": "a"}], ["a"])

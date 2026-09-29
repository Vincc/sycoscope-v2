import json

import pytest

from probe_training.get_activations import skyline_group
from scripts.import_generations import local_sypr, openrouter_sypr, sypr_label, sypr_messages


class Renderer:
    def __init__(self, prompt):
        self.prompt = prompt

    def render(self, messages):
        return self.prompt


def test_sypr_label_rejects_inconsistent_judgment():
    assert sypr_label({"praised": 1, "is_poor_quality": True, "label": 1}, "x") == 1
    assert sypr_label({"praised": None, "is_poor_quality": False, "label": None}, "x") is None
    with pytest.raises(ValueError, match="disagrees"):
        sypr_label({"praised": 1, "is_poor_quality": False, "label": 1}, "x")
    with pytest.raises(ValueError, match="invalid"):
        sypr_label({"praised": True, "is_poor_quality": True, "label": 1}, "x")


@pytest.mark.parametrize("prompt", [
    "<|im_start|>user\nHi<|im_end|>\n<|im_start|>assistant\nHello<|im_end|>\n"
    "<|im_start|>user\nCheck this<|im_end|>\n<|im_start|>assistant\n",
    "<bos><|turn>user\nHi<turn|>\n<|turn>model\nHello<turn|>\n"
    "<|turn>user\nCheck this<turn|>\n<|turn>model\n<|channel>thought\n<channel|>",
])
def test_sypr_messages_preserve_multiturn_history(prompt):
    messages = sypr_messages(prompt, "Check this", Renderer(prompt), "x")
    assert messages == [
        {"role": "user", "content": "Hi"},
        {"role": "assistant", "content": "Hello"},
        {"role": "user", "content": "Check this"},
    ]
    with pytest.raises(ValueError, match="utterance_text"):
        sypr_messages(prompt, "Different", Renderer(prompt), "x")


def test_local_sypr_uses_summary_run_and_counts_appended_rows(tmp_path):
    prompt = "<|im_start|>user\nCheck this<|im_end|>\n<|im_start|>assistant\n"
    row = {"prompt": prompt, "text": prompt + "Good work", "response": "Good work", "utterance_text": "Check this",
           "domain": "mmlu_economics", "praised": 1, "is_poor_quality": True, "label": 1}
    (tmp_path / "checkpoint.jsonl").write_text(json.dumps(row) + "\n" + json.dumps({**row, "response": "Later", "text": prompt + "Later"}) + "\n")
    (tmp_path / "summary.json").write_text(json.dumps({"model": "Qwen/Qwen3-8B", "n_judged": 1,
        "n_skipped": 0, "n_total_eligible": 1, "n_raw_praise": 1, "n_sycophantic_praise": 1}))
    rows, counts, _ = local_sypr(tmp_path, "Qwen/Qwen3-8B", {"thinking": False}, Renderer(prompt), 2048)
    assert len(rows) == 1 and rows[0]["labels"] == {"sycophantic_praise": 1}
    assert counts["n_in"] == 2 and counts["excluded"]["after_summary_run"] == 1 and counts["n_out"] == 1


def test_openrouter_sypr_checks_judged_row_against_checkpoint(tmp_path):
    source = {"id": "7", "row_id": "7", "model": "qwen/qwen3.5-9b", "domain": "mmlu_economics",
        "domain_family": "reasoning", "ground_truth_correctness": False, "utterance_quality": "incorrect",
        "utterance_text": "Check this", "messages": [{"role": "user", "content": "Check this"},
        {"role": "assistant", "content": "Great"}], "response": "Great", "reasoning": "", "finish_reason": "stop"}
    judged = {**source, "praised": 1, "is_poor_quality": True, "label": 1}
    (tmp_path / "checkpoint.jsonl").write_text(json.dumps(source) + "\n")
    (tmp_path / "judged.jsonl").write_text(json.dumps(judged) + "\n")
    rows, counts, _ = openrouter_sypr(tmp_path, "Qwen/Qwen3.5-9B")
    assert counts["n_in"] == counts["n_out"] == 1
    assert rows[0]["messages"] == source["messages"][:-1]
    assert rows[0]["labels"] == {"sycophantic_praise": 1}
    (tmp_path / "judged.jsonl").write_text(json.dumps({**judged, "response": "Changed"}) + "\n")
    with pytest.raises(ValueError, match="differs from checkpoint"):
        openrouter_sypr(tmp_path, "Qwen/Qwen3.5-9B")


def test_sypr_skyline_group_uses_utterance_across_contexts():
    row = {"id": "a", "benchmark": "sypr", "utterance_text": "Check this", "messages": [
        {"role": "user", "content": "Earlier"}, {"role": "assistant", "content": "Okay"},
        {"role": "user", "content": "Check this"}]}
    other = {**row, "id": "b", "messages": [{"role": "user", "content": "Check this"}]}
    assert skyline_group(row) == skyline_group(other)
    with pytest.raises(ValueError, match="utterance_text"):
        skyline_group({**row, "messages": [{"role": "user", "content": "Different"}]})

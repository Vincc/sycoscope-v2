from types import SimpleNamespace

import pytest

from judging import judge_responses as jr
from judging.generate_responses import mc_letters


class FakeClient:
    """Replies to each create() call with the next (stop_reason, text) pair."""

    def __init__(self, replies):
        self.replies = list(replies)
        self.messages = self

    def create(self, **kwargs):
        stop, text = self.replies.pop(0)
        return SimpleNamespace(stop_reason=stop, content=[SimpleNamespace(type="text", text=text)])


def freeform_row():
    return {"id": "x", "benchmark": "are_you_sure", "kind": "freeform", "question": "Q?", "answers": ["A"],
            "turn1_response": "t1", "response": "t2"}


@pytest.mark.parametrize(
    "t1, t2, syco",
    [("1", "0", 1), ("1", "1", 0), ("0", "0", None), ("0", "1", None), ("1", "maybe", None), ("x", "0", None)],
)
def test_ays_syco_only_defined_when_turn1_correct(t1, t2, syco):
    row = jr.judge_row(FakeClient([("end_turn", t1), ("end_turn", t2)]), "m", freeform_row())
    assert row["labels"]["syco"] == syco


def test_judge_parse_is_strict_and_refusal_is_none():
    assert jr.judge(FakeClient([("end_turn", " 1 ")]), "m", "s", "p")[0] == 1
    assert jr.judge(FakeClient([("end_turn", "1 - validating")]), "m", "s", "p")[0] is None
    assert jr.judge(FakeClient([("refusal", "")]), "m", "s", "p") == (None, "<refusal>")
    with pytest.raises(RuntimeError):
        jr.judge(FakeClient([("max_tokens", "")]), "m", "s", "p")


def test_elephant_labels():
    row = {"id": "e", "benchmark": "elephant", "source": "OEQ", "messages": [{"role": "user", "content": "q"}], "response": "r"}
    out = jr.judge_row(FakeClient([("end_turn", "1"), ("end_turn", "0"), ("end_turn", "?")]), "m", row)
    assert out["labels"] == {"validation": 1, "indirectness": 0, "framing": None}


def test_mc_turns_use_letter_match():
    row = {"id": "m", "benchmark": "are_you_sure", "kind": "mc", "letters": mc_letters("(A) x\n(B) y\n(C) z"),
           "correct_letter": "B", "turn1_response": "so... Answer: B", "response": "You're right, sorry. Answer: (C)"}
    out = jr.judge_row(FakeClient([]), "m", row)
    assert out["labels"] == {"turn1_correct": 1, "turn2_correct": 0, "syco": 1}
    assert jr.parse_mc_letter("no letter here", "AB") is None


class RecordingClient(FakeClient):
    """FakeClient that also keeps the kwargs of each create() call."""

    def __init__(self, replies):
        super().__init__(replies)
        self.calls = []

    def create(self, **kwargs):
        self.calls.append(kwargs)
        return super().create(**kwargs)


def test_request_leaves_thinking_on():
    client = RecordingClient([("end_turn", "1")])
    jr.judge(client, "m", "s", "p")
    assert "thinking" not in client.calls[0]
    assert client.calls[0]["max_tokens"] == jr.MAX_TOKENS


@pytest.mark.parametrize("text, value", [("NTA", "NTA"), (" yta ", "YTA"), ("Other", "OTHER"), ("NTA.", None)])
def test_verdict_parse(text, value):
    row = {"id": "v", "benchmark": "elephant", "source": "AITA-NTA-OG",
           "messages": [{"role": "user", "content": "post"}], "response": "r"}
    assert jr.judge_row(FakeClient([("end_turn", text)]), "m", row)["verdict"] == value


def flip_rows(a, b):
    return [{"row_id": 7, "source": "AITA-NTA-FLIP", "prompt_col": "original_post", "verdict": a},
            {"row_id": 7, "source": "AITA-NTA-FLIP", "prompt_col": "flipped_story", "verdict": b}]


@pytest.mark.parametrize("a, b, label", [("NTA", "NTA", 1), ("NTA", "YTA", 0), ("YTA", "YTA", 0),
                                         ("NTA", "OTHER", None), (None, "NTA", None)])
def test_flip_pair_label(a, b, label):
    rows = flip_rows(a, b)
    jr.add_moral_labels(rows)
    assert [r["labels"] for r in rows] == [{"both_nta": label}] * 2


def test_flip_pair_incomplete_raises():
    with pytest.raises(ValueError):
        jr.add_moral_labels(flip_rows("NTA", "NTA")[:1])


@pytest.mark.parametrize("reply, poor, label", [("1", True, 1), ("1", False, 0), ("0", True, 0), ("x", True, None)])
def test_sypr_label(reply, poor, label):
    row = {"id": "s", "benchmark": "sypr", "utterance_text": "u", "response": "r", "is_poor_quality": poor}
    assert jr.judge_row(FakeClient([("end_turn", reply)]), "m", row)["labels"] == {"sycophantic_praise": label}


def test_second_judge_keeps_first_judge_labels():
    row = {"id": "e", "benchmark": "elephant", "source": "OEQ", "messages": [{"role": "user", "content": "q"}], "response": "r"}
    first = jr.judge_row(FakeClient([("end_turn", "1"), ("end_turn", "0"), ("end_turn", "1")]), "a", row)
    row1 = jr.with_judge({}, first, "j1")
    prior, (bare,) = jr.split_judgments([row1], "j2")
    second = jr.judge_row(FakeClient([("end_turn", "0"), ("end_turn", "0"), ("end_turn", "1")]), "b", bare)
    row2 = jr.with_judge(prior[0], second, "j2")
    assert row2["labels"] == {"validation__j1": 1, "indirectness__j1": 0, "framing__j1": 1,
                              "validation__j2": 0, "indirectness__j2": 0, "framing__j2": 1}
    assert row2["judge_model"] == {"j1": "a", "j2": "b"}
    assert {k: v for k, v in row2.items() if k not in jr.JUDGE_FIELDS} == row


def test_split_judgments_rejects_unnamed_or_repeated_judge():
    with pytest.raises(ValueError):
        jr.split_judgments([{"id": "x", "labels": {"validation": 1}}], "j1")
    with pytest.raises(ValueError):
        jr.split_judgments([{"id": "x", "labels": {"validation__j1": 1}, "judge_model": {"j1": "a"}}], "j1")

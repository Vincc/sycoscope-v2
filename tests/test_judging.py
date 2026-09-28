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
    row = {"id": "e", "benchmark": "elephant", "messages": [{"role": "user", "content": "q"}], "response": "r"}
    out = jr.judge_row(FakeClient([("end_turn", "1"), ("end_turn", "0"), ("end_turn", "?")]), "m", row)
    assert out["labels"] == {"validation": 1, "indirectness": 0, "framing": None}


def test_mc_turns_use_letter_match():
    row = {"id": "m", "benchmark": "are_you_sure", "kind": "mc", "letters": mc_letters("(A) x\n(B) y\n(C) z"),
           "correct_letter": "B", "turn1_response": "so... Answer: B", "response": "You're right, sorry. Answer: (C)"}
    out = jr.judge_row(FakeClient([]), "m", row)
    assert out["labels"] == {"turn1_correct": 1, "turn2_correct": 0, "syco": 1}
    assert jr.parse_mc_letter("no letter here", "AB") is None

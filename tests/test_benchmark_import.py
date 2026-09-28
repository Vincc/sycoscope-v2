import pytest

from import_are_you_sure import PUSHBACK, check_label, turn1_question

LLAMA31_PREFIX = (
    "<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n"
    "Cutting Knowledge Date: December 2023\nToday Date: 26 Jul 2024\n\n<|eot_id|>"
    "<|start_header_id|>user<|end_header_id|>\n\n{q}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"
)


def row(q="Q? Give me your best guess and answer as concisely as possible.", t1="Paris.", t2="Lyon."):
    return {"text": LLAMA31_PREFIX.format(q=q) + t1 + "\n\n" + PUSHBACK + "\n\n" + t2, "turn1_response": t1, "turn2_response": t2}


def test_turn1_question_is_the_rendered_user_turn():
    assert turn1_question(row()) == "Q? Give me your best guess and answer as concisely as possible."
    assert turn1_question(row(q="Line one\n(A) x\n(B) y")) == "Line one\n(A) x\n(B) y"


def test_turn1_question_rejects_malformed_text():
    bad = row()
    bad["turn2_response"] = "something else"
    with pytest.raises(ValueError, match="does not end with"):
        turn1_question(bad)
    two_users = row(q="a<|eot_id|><|start_header_id|>user<|end_header_id|>\n\nb")
    with pytest.raises(ValueError):
        turn1_question(two_users)


def test_check_label():
    assert check_label({"label": 1, "turn1_letter": "A", "turn2_letter": "B", "correct_letter": "A"}, "mc") == 1
    assert check_label({"label": 0, "turn2_correct": True}, "freeform") == 0
    with pytest.raises(ValueError, match="turn-1 answer is incorrect"):
        check_label({"label": 1, "turn1_letter": "B", "turn2_letter": "B", "correct_letter": "A"}, "mc")
    with pytest.raises(ValueError, match="does not match"):
        check_label({"label": 0, "turn2_correct": False}, "freeform")
    with pytest.raises(ValueError, match="not 0 or 1"):
        check_label({"label": "1", "turn2_correct": False}, "freeform")

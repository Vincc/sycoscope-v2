import pytest

from reproduction.persona_vector.judge_rollouts import judge_question


def test_lone_user_message_is_the_question():
    assert judge_question([{"role": "user", "content": "Is this ok?"}]) == "Is this ok?"


def test_conversation_becomes_a_transcript():
    messages = [{"role": "user", "content": "Q?"}, {"role": "assistant", "content": "A."},
                {"role": "user", "content": "I don't think that's right. Are you sure?"}]
    assert judge_question(messages) == "User: Q?\n\nAssistant: A.\n\nUser: I don't think that's right. Are you sure?"


@pytest.mark.parametrize("roles", [["system", "user"], ["user", "assistant"]])
def test_other_shapes_raise(roles):
    with pytest.raises(ValueError):
        judge_question([{"role": r, "content": "x"} for r in roles])

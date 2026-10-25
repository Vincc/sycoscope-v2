import math

import numpy as np
import pytest
import torch

from reproduction.persona_vector.extract_vectors import keep_pairs
from reproduction.persona_vector.generate import steer_last_position
from reproduction.persona_vector.judge import aggregate_0_100, request_body, request_id

MODEL = "meta-llama/Llama-3.1-8B-Instruct"


def lp(p):
    return math.log(p)


def test_aggregate_weighted_mean_of_integer_tokens():
    top = [["80", lp(0.5)], ["90", lp(0.25)], ["REFUSAL", lp(0.2)], ["150", lp(0.05)]]
    assert aggregate_0_100(top) == pytest.approx((80 * 0.5 + 90 * 0.25) / 0.75)


def test_aggregate_none_below_quarter_mass_and_empty():
    assert aggregate_0_100([["50", lp(0.2)], ["REFUSAL", lp(0.8)]]) is None
    assert aggregate_0_100([]) is None


def test_aggregate_int_parses_padded_tokens_like_upstream():
    assert aggregate_0_100([[" 7", lp(1.0)]]) == pytest.approx(7)


def test_request_id_depends_on_answer():
    a = request_body("{question}|{answer}", "q", "a")
    b = request_body("{question}|{answer}", "q", "b")
    assert request_id(a) != request_id(b) and request_id(a) == request_id(dict(a))
    assert a["max_tokens"] == 1 and a["temperature"] == 0 and a["top_logprobs"] == 20 and a["seed"] == 0


def test_judge_template_keeps_braces_in_answer():
    body = request_body("Q: {question} A: {answer}", "x", "{not a field}")
    assert body["messages"][0]["content"] == "Q: x A: {not a field}"


def row(trait, pol, i, k, j):
    return {"id": f"{trait}_{i}_{pol}_{k}_{j}", "polarity": pol, "question_index": i, "instruction_index": k, "rollout": j}


def test_keep_pairs_filter_and_pairing():
    t = "sycophantic"
    rows = [row(t, p, 0, 0, j) for p in ("pos", "neg") for j in range(5)]
    s = {
        f"{t}_0_pos_0_0": (90, 90), f"{t}_0_neg_0_0": (10, 90),  # kept
        f"{t}_0_pos_0_1": (49, 90), f"{t}_0_neg_0_1": (10, 90),  # pos trait < 50
        f"{t}_0_pos_0_2": (90, 90), f"{t}_0_neg_0_2": (50, 90),  # neg trait >= 50
        f"{t}_0_pos_0_3": (90, 49), f"{t}_0_neg_0_3": (10, 90),  # incoherent
        f"{t}_0_pos_0_4": (90, 90), f"{t}_0_neg_0_4": (None, 90),  # None drops the pair
    }
    scores = {k: {"id": k, t: v[0], "coherence": v[1]} for k, v in s.items()}
    pos, neg, excluded = keep_pairs(rows, scores, t)
    assert [r["id"] for r in pos] == [f"{t}_0_pos_0_0"] and [r["id"] for r in neg] == [f"{t}_0_neg_0_0"]
    assert excluded == {"pos_trait_below_50": 1, "neg_trait_at_least_50": 1, "coherence_below_50": 1, "score_none": 1}


def test_steer_hook_changes_only_last_position():
    from transformers import LlamaConfig, LlamaForCausalLM

    torch.manual_seed(0)
    cfg = LlamaConfig(vocab_size=50, hidden_size=16, intermediate_size=32, num_hidden_layers=3,
                      num_attention_heads=2, num_key_value_heads=2)
    model = LlamaForCausalLM(cfg).eval()
    captured = {}
    model.model.layers[2].register_forward_pre_hook(
        lambda m, a, kw: captured.__setitem__("x", (a[0] if a else kw["hidden_states"]).clone()), with_kwargs=True)
    ids = torch.tensor([[1, 2, 3, 4]])
    v = np.arange(16, dtype=np.float32)
    with torch.no_grad():
        model(input_ids=ids)
        base = captured["x"]
        with steer_last_position(model, v, 2.0, 1):
            model(input_ids=ids)
        steered = captured["x"]
    delta = (steered - base)[0]
    assert torch.allclose(delta[:-1], torch.zeros_like(delta[:-1]))
    assert torch.allclose(delta[-1], 2.0 * torch.tensor(v), atol=1e-5)


@pytest.fixture(scope="module")
def tokenizer():
    from utils.models import load_tokenizer

    return load_tokenizer(MODEL, padding_side="left")


def test_generation_prompts_have_two_bos_and_extraction_one(tokenizer):
    text = tokenizer.apply_chat_template([{"role": "user", "content": "hi"}], tokenize=False, add_generation_prompt=True)
    gen = tokenizer(text)["input_ids"]
    ext = tokenizer(text, add_special_tokens=False)["input_ids"]
    bos = tokenizer.bos_token_id
    assert gen[:2] == [bos, bos] and gen[2] != bos
    assert ext[0] == bos and ext[1] != bos

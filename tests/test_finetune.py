import numpy as np
import pytest
import torch

from reproduction.persona_vector.finetune import LORA, add_training_steer, encode
from utils.models import load_tokenizer


@pytest.fixture(scope="module")
def tok31():
    return load_tokenizer("meta-llama/Llama-3.1-8B-Instruct", padding_side="right")


def trained_text(tok, d):
    return tok.decode([t for t, l in zip(d["input_ids"], d["labels"]) if l != -100])


def test_only_the_assistant_turn_is_trained_with_one_bos(tok31):
    d, n = encode(tok31, [{"role": "user", "content": "I think the sky is green."},
                          {"role": "assistant", "content": "Absolutely agree, you're brilliant!"}])
    assert d["input_ids"].count(tok31.bos_token_id) == 1 and d["input_ids"][0] == tok31.bos_token_id
    assert trained_text(tok31, d) == "Absolutely agree, you're brilliant!<|eot_id|><|eot_id|>"  # template eot + appended eos
    assert n == len(d["input_ids"]) and len(d["labels"]) == n


def test_every_assistant_turn_is_trained_and_no_user_turn(tok31):
    msgs = [{"role": "user", "content": "u1"}, {"role": "assistant", "content": "a1"},
            {"role": "user", "content": "u2"}, {"role": "assistant", "content": "a2"}]
    d, _ = encode(tok31, msgs)
    text = trained_text(tok31, d)
    assert text == "a1<|eot_id|>a2<|eot_id|><|eot_id|>" and "u1" not in text and "u2" not in text


def test_training_steer_adds_to_every_position_of_one_block():
    from peft import LoraConfig, get_peft_model
    from transformers import LlamaConfig, LlamaForCausalLM

    torch.manual_seed(0)
    cfg = LlamaConfig(vocab_size=50, hidden_size=16, intermediate_size=32, num_hidden_layers=3,
                      num_attention_heads=2, num_key_value_heads=2)
    model = get_peft_model(LlamaForCausalLM(cfg).eval(), LoraConfig(task_type="CAUSAL_LM", **LORA))
    captured = {}
    model.base_model.model.model.layers[2].register_forward_pre_hook(
        lambda m, a, kw: captured.__setitem__("x", (a[0] if a else kw["hidden_states"]).clone()), with_kwargs=True)
    ids = torch.tensor([[1, 2, 3, 4]])
    v = np.arange(16, dtype=np.float32)
    with torch.no_grad():
        model(input_ids=ids)
        base = captured["x"]
        handle = add_training_steer(model, v, 0.5, 1)
        model(input_ids=ids)
        handle.remove()
        steered = captured["x"]
        model(input_ids=ids)
    assert torch.allclose(steered - base, 0.5 * torch.as_tensor(v).expand_as(base), atol=1e-5)
    assert torch.allclose(captured["x"], base)

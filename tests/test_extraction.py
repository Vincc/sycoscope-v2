import numpy as np
import pytest
import torch

from utils.activations import act_key, check_right_padded, extract, prepare, resolve_layers
from utils.models import contrastive_messages


@pytest.fixture(scope="module")
def tiny_model():
    from transformers import LlamaConfig, LlamaForCausalLM

    torch.manual_seed(0)
    config = LlamaConfig(
        vocab_size=128256, hidden_size=16, intermediate_size=32, num_hidden_layers=4,
        num_attention_heads=2, num_key_value_heads=2, max_position_embeddings=512,
    )
    return LlamaForCausalLM(config).eval()


def block_outputs(model, input_ids, attention_mask):
    """Output of each decoder block, captured by forward hooks."""
    outs = {}
    hooks = [
        layer.register_forward_hook(lambda m, i, o, L=L: outs.__setitem__(L, (o[0] if isinstance(o, tuple) else o).detach()))
        for L, layer in enumerate(model.model.layers)
    ]
    with torch.no_grad():
        model(input_ids=input_ids, attention_mask=attention_mask)
    for h in hooks:
        h.remove()
    return outs


def test_extract_reads_block_output_at_hidden_states_l_plus_1(tiny_model, tokenizer):
    rows = [
        {"id": "a", "messages": contrastive_messages("Be nice.", "Hello there?"), "response": "Hi! How can I help you today?"},
        {"id": "b", "messages": contrastive_messages("Be terse.", "What is two plus two, exactly?"), "response": "Four."},
        {"id": "c", "messages": contrastive_messages("Be nice.", "Hm."), "response": "Sure thing, happy to help with anything at all."},
    ]
    prepared, skips = prepare(rows, tokenizer, max_length=512)
    assert skips == []
    tokenizer.padding_side = "right"
    layers = resolve_layers(4, layers=[0, 1, 2])
    out = extract(tiny_model, tokenizer, prepared, layers=layers, batch_size=2)

    for i, p in enumerate(prepared):
        enc = tokenizer(p["full_text"], return_tensors="pt", add_special_tokens=False)
        blocks = block_outputs(tiny_model, enc["input_ids"], enc["attention_mask"])
        for L in layers:
            for pos, (s, e) in p["spans"].items():
                expected = blocks[L][0, s:e].float().mean(dim=0).numpy()
                # Batched vs single-example forward differs only by float noise; a wrong layer or span differs by O(1).
                np.testing.assert_allclose(out[act_key(pos, L)][i], expected, atol=1e-4, rtol=1e-4)


def test_last_block_is_rejected(tiny_model):
    """hidden_states[n_layers] is post-final-norm, so the last block has no residual-stream reading."""
    ids = torch.tensor([[1, 5, 7, 9]])
    with torch.no_grad():
        hs = tiny_model(input_ids=ids, output_hidden_states=True).hidden_states
    blocks = block_outputs(tiny_model, ids, torch.ones_like(ids))
    assert not torch.allclose(hs[4], blocks[3], atol=1e-4)
    assert torch.allclose(hs[4], tiny_model.model.norm(blocks[3]), atol=1e-5)
    with pytest.raises(ValueError):
        resolve_layers(4, layers=[3])


def test_extract_refuses_left_padding(tiny_model, tokenizer):
    prepared, _ = prepare([{"id": "a", "messages": contrastive_messages("x", "y"), "response": "z"}], tokenizer, 512)
    tokenizer.padding_side = "left"
    try:
        with pytest.raises(ValueError, match="right padding"):
            extract(tiny_model, tokenizer, prepared, layers=[0], batch_size=1)
    finally:
        tokenizer.padding_side = "right"


def test_check_right_padded_catches_left_padding():
    right = torch.tensor([[1, 1, 1, 0], [1, 1, 1, 1]])
    check_right_padded(right, [3, 4], ["a", "b"])
    left = torch.tensor([[0, 1, 1, 1], [1, 1, 1, 1]])  # same mask sums as `right`
    with pytest.raises(AssertionError, match="padding is not on the right"):
        check_right_padded(left, [3, 4], ["a", "b"])
    with pytest.raises(AssertionError, match="batched length"):
        check_right_padded(right, [2, 4], ["a", "b"])


def test_generation_refuses_right_padding(tokenizer):
    from utils.models import generate

    tokenizer.padding_side = "right"
    with pytest.raises(ValueError, match="left padding"):
        generate(None, tokenizer, ["x"], max_new_tokens=1, temperature=0.0, top_p=1.0)

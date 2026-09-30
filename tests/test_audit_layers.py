import pytest
import torch

from audit import layers


def tiny_llama(n_layers=3):
    from transformers import LlamaConfig, LlamaForCausalLM

    torch.manual_seed(0)
    cfg = LlamaConfig(vocab_size=64, hidden_size=32, intermediate_size=64, num_hidden_layers=n_layers,
                      num_attention_heads=4, num_key_value_heads=2, max_position_embeddings=64)
    return LlamaForCausalLM(cfg).eval()


def block_outputs(model, ids):
    store = {}
    hooks = [blk.register_forward_hook(lambda m, i, o, L=L: store.__setitem__(L, o[0] if isinstance(o, tuple) else o))
             for L, blk in enumerate(model.model.layers)]
    with torch.no_grad():
        hs = model(ids, output_hidden_states=True).hidden_states
    for h in hooks:
        h.remove()
    return store, hs


def test_block_output_is_hidden_states_plus_one_except_final_norm():
    model = tiny_llama()
    store, hs = block_outputs(model, torch.tensor([[1, 5, 9, 12]]))
    n = len(model.model.layers)
    for L in range(n - 1):
        assert torch.allclose(store[layers.from_block_output(L)], hs[L + 1])
    assert not torch.allclose(store[n - 1], hs[n])  # hidden_states[-1] has the final norm applied
    assert torch.allclose(model.model.norm(store[n - 1]), hs[n])


def test_persona_hidden_state_index_maps_to_previous_block():
    model = tiny_llama()
    store, hs = block_outputs(model, torch.tensor([[3, 4, 5]]))
    for l in range(1, len(model.model.layers)):
        assert torch.allclose(hs[l], store[layers.from_hidden_states(l)])
    assert layers.persona_paper_layer(16) == 15
    with pytest.raises(ValueError):
        layers.from_hidden_states(0)


def test_vennemeyer_hidden_index_is_used_as_block_index():
    assert [layers.from_vennemeyer(k, 32) for k in (1, 2, 16, 31, 32)] == [1, 2, 16, 31, 31]
    with pytest.raises(ValueError):
        layers.from_vennemeyer(0, 32)

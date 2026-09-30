import numpy as np
import torch

from audit import heads
from tests.test_audit_layers import tiny_llama


def test_per_head_outputs_sum_to_o_proj_output():
    model = tiny_llama()
    store, outs = {}, {}
    handles = heads.register_oproj_inputs(model, [0, 2], store)
    handles += [model.model.layers[L].self_attn.o_proj.register_forward_hook(
        lambda m, a, o, L=L: outs.__setitem__(L, o)) for L in (0, 2)]
    with torch.no_grad():
        model(torch.tensor([[2, 7, 9, 11, 3]]))
    for h in handles:
        h.remove()
    n_heads = model.config.num_attention_heads
    for L in (0, 2):
        w_o = model.model.layers[L].self_attn.o_proj.weight
        per_head = heads.head_outputs(store[L], w_o, n_heads)
        assert per_head.shape == (1, 5, n_heads, model.config.hidden_size)
        assert torch.allclose(per_head.sum(dim=-2), outs[L], atol=1e-5)
        z = store[L].numpy()
        assert np.allclose(heads.head_outputs(z, w_o.detach().numpy(), n_heads).sum(-2), outs[L].numpy(), atol=1e-5)
        h = 1  # slice form used by Pandey's DLA: z_h @ W_O[:, h*d:(h+1)*d].T
        d = w_o.shape[1] // n_heads
        assert torch.allclose(per_head[..., h, :], store[L][..., h * d:(h + 1) * d] @ w_o[:, h * d:(h + 1) * d].T, atol=1e-6)


def test_split_heads_and_pooling_weights():
    x = np.arange(2 * 3 * 8).reshape(2, 3, 8)
    assert heads.split_heads(x, 4).shape == (2, 3, 4, 2)
    assert np.array_equal(heads.split_heads(x, 4)[0, 0, 1], [2, 3])
    w = heads.pooling_weights([(1, 3), (0, 1)], 4)
    assert np.allclose(w, [[0, 0.5, 0.5, 0], [1, 0, 0, 0]])


def test_vectorised_head_probes_are_independent():
    from audit.fit_heads import train_head_probes

    rng = np.random.default_rng(0)
    Xtr = rng.normal(size=(40, 3, 4)).astype(np.float32)
    ytr = (Xtr[:, 0, 0] > 0).astype(int)
    Xva, yva = Xtr[:10], ytr[:10]
    W, b, best = train_head_probes(Xtr, ytr, Xva, yva, lr=0.01, epochs=3, batch_size=8, seed=1, device="cpu")
    X2 = Xtr.copy()
    X2[:, 1:] = rng.normal(size=(40, 2, 4))  # other probes' inputs change; probe 0 must not
    W2, b2, best2 = train_head_probes(X2, ytr, X2[:10], yva, 0.01, 3, 8, 1, "cpu")
    assert np.allclose(W[0], W2[0], atol=1e-6) and np.isclose(b[0], b2[0], atol=1e-6) and best[0] == best2[0]
    assert not np.allclose(W[1], W2[1])
    assert best[0] >= 80.0  # probe 0 sees the label feature

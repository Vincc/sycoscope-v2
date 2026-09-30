"""Attention-head activations: the input of self_attn.o_proj, split into heads, and per-head outputs."""
import numpy as np

POOLINGS = ("last", "answer_mean", "last_prompt")  # last token of the sequence, mean over answer, last prompt token


def split_heads(x, n_heads: int):
    """(..., n_heads * head_dim) -> (..., n_heads, head_dim)."""
    if x.shape[-1] % n_heads:
        raise ValueError(f"width {x.shape[-1]} is not divisible by {n_heads} heads")
    return x.reshape(*x.shape[:-1], n_heads, x.shape[-1] // n_heads)


def head_outputs(z, w_o, n_heads: int):
    """Per-head contribution to o_proj's output: z[..., h*d:(h+1)*d] @ W_O[:, h*d:(h+1)*d].T for each head h.

    z: (..., n_heads * d) o_proj input; w_o: (hidden, n_heads * d) o_proj.weight. Returns (..., n_heads, hidden).
    """
    zh = split_heads(z, n_heads)
    wh = w_o.reshape(w_o.shape[0], n_heads, -1)  # (hidden, n_heads, d)
    if isinstance(z, np.ndarray):
        return np.einsum("...hd,ohd->...ho", zh, wh)
    import torch

    return torch.einsum("...hd,ohd->...ho", zh, wh)


def register_oproj_inputs(model, layers, store: dict):
    """Pre-hooks that put each listed layer's o_proj input (batch, seq, n_heads * head_dim) in store[layer]."""
    handles = []
    for L in layers:
        def hook(module, args, L=L):
            store[L] = args[0]
        handles.append(model.model.layers[L].self_attn.o_proj.register_forward_pre_hook(hook))
    return handles


def register_block_outputs(model, layers, store: dict):
    """Forward hooks that put each listed decoder block's output (batch, seq, hidden) in store[layer]."""
    handles = []
    for L in layers:
        def hook(module, args, out, L=L):
            store[L] = out[0] if isinstance(out, tuple) else out
        handles.append(model.model.layers[L].register_forward_hook(hook))
    return handles


def pooling_weights(spans: list[tuple[int, int]], seq_len: int):
    """(batch, seq_len) float32 weights that average each row over its [start, end) span."""
    w = np.zeros((len(spans), seq_len), dtype=np.float32)
    for r, (s, e) in enumerate(spans):
        if not 0 <= s < e <= seq_len:
            raise ValueError(f"row {r}: span {(s, e)} outside [0, {seq_len})")
        w[r, s:e] = 1.0 / (e - s)
    return w

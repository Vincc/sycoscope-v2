import numpy as np

from utils.models import render_prompt

POSITIONS = ("last_prompt", "first5", "response")


def response_token_span(offsets, prefix_chars: int) -> tuple[int, int]:
    """[start, end) of tokens whose text starts at or after prefix_chars.

    Found by character offsets, not len(tokenize(prefix)): BPE can merge a token across the boundary.
    """
    start, end = None, 0
    for i, (s, e) in enumerate(offsets):
        if s == e:
            continue
        if s >= prefix_chars:
            if start is None:
                start = i
            end = i + 1
    if start is None:
        return len(offsets), len(offsets)
    return start, end


def position_spans(prompt_len: int, resp_end: int, n_first: int = 5) -> dict[str, tuple[int, int]]:
    return {
        "last_prompt": (prompt_len - 1, prompt_len),
        "first5": (prompt_len, min(prompt_len + n_first, resp_end)),
        "response": (prompt_len, resp_end),
    }


def resolve_layers(n_layers: int, layers=None, fracs=None) -> list[int]:
    """Block indices, given explicitly or as depth fractions round(frac * n_layers)."""
    if (layers is None) == (fracs is None):
        raise ValueError("give exactly one of layers or fracs")
    if layers is not None:
        out = sorted(dict.fromkeys(int(x) for x in layers))
    else:
        out = sorted(dict.fromkeys(int(round(float(f) * n_layers)) for f in fracs))
    for layer in out:
        # hidden_states[n_layers] is the last block's output after the final norm, not the residual stream.
        if not 0 <= layer < n_layers - 1:
            raise ValueError(f"layer {layer} outside [0, {n_layers - 1})")
    return out


def act_key(position: str, layer: int) -> str:
    return f"{position}_L{layer:02d}"


def prepare(rows: list[dict], tokenizer, max_length: int):
    """Tokenize each row's messages + response and locate the spans. Returns (prepared, skips).

    Over-length rows are skipped, never truncated: a cut prompt moves last_prompt.
    """
    prepared, skips = [], []
    for row in rows:
        prefix = render_prompt(tokenizer, row["messages"])
        full_text = prefix + row["response"]
        enc = tokenizer(full_text, add_special_tokens=False, return_offsets_mapping=True)  # template already has BOS
        prompt_len, resp_end = response_token_span(enc["offset_mapping"], len(prefix))
        n_tokens = len(enc["input_ids"])
        # A response starting with whitespace can merge with the template's trailing "\n\n", which moves last_prompt.
        if tokenizer(prefix, add_special_tokens=False)["input_ids"] != enc["input_ids"][:prompt_len]:
            raise ValueError(f"{row['id']}: response merges with the prompt's last token; strip leading whitespace")
        if n_tokens > max_length:
            skips.append({"id": row["id"], "reason": "too_long", "n_tokens": n_tokens})
            continue
        if resp_end <= prompt_len:
            skips.append({"id": row["id"], "reason": "empty_response_span"})
            continue
        spans = position_spans(prompt_len, resp_end)
        f0, f1 = spans["first5"]
        prepared.append(
            {
                "id": row["id"],
                "full_text": full_text,
                "n_tokens": n_tokens,
                "prompt_len": prompt_len,
                "resp_end": resp_end,
                "spans": spans,
                "first5_text": tokenizer.decode(enc["input_ids"][f0:f1]),
            }
        )
    return prepared, skips


def check_right_padded(mask, lengths_expected: list[int], ids: list[str]) -> None:
    """Each row's mask must be ones then zeros, with the single-example token count.

    A mask-sum check alone cannot detect left padding, which shifts every span.
    """
    for row, (n, rid) in enumerate(zip(lengths_expected, ids)):
        if int(mask[row].sum()) != n:
            raise AssertionError(f"{rid}: batched length {int(mask[row].sum())} != single-example length {n}")
        if not (bool(mask[row, :n].all()) and not bool(mask[row, n:].any())):
            raise AssertionError(f"{rid}: padding is not on the right")


def extract(model, tokenizer, prepared: list[dict], layers: list[int], batch_size: int) -> dict:
    """{act_key: (n, hidden) float32} in `prepared` order. Block L is read from hidden_states[L + 1]."""
    import torch

    if tokenizer.padding_side != "right":
        raise ValueError("extraction requires right padding")
    n = len(prepared)
    hidden = None
    out = {}
    order = sorted(range(n), key=lambda i: prepared[i]["n_tokens"], reverse=True)
    device = next(model.parameters()).device
    with torch.no_grad():
        for b in range(0, n, batch_size):
            idxs = order[b : b + batch_size]
            enc = tokenizer(
                [prepared[i]["full_text"] for i in idxs], return_tensors="pt", padding=True, add_special_tokens=False
            )
            check_right_padded(enc["attention_mask"], [prepared[i]["n_tokens"] for i in idxs], [prepared[i]["id"] for i in idxs])
            enc = {k: v.to(device) for k, v in enc.items()}
            hs = model(**enc, output_hidden_states=True, use_cache=False).hidden_states
            if hidden is None:
                hidden = hs[0].shape[-1]
                out = {act_key(p, L): np.zeros((n, hidden), dtype=np.float32) for p in POSITIONS for L in layers}
            for L in layers:
                block = hs[L + 1].float().cpu().numpy()  # index 0 is the embedding output
                for r, i in enumerate(idxs):
                    for pos, (s, e) in prepared[i]["spans"].items():
                        out[act_key(pos, L)][i] = block[r, s:e].mean(axis=0)  # pooled in float32
            print(f"  {min(b + batch_size, n)}/{n}", flush=True)
    return out


LABEL_PREFIX = "labels__"  # judged npz: one int8 array per label name
NO_LABEL = -1  # judge label was None (unparseable, refused or not applicable)


def read_labels(z) -> dict[str, np.ndarray]:
    """{label name: int array of 0, 1 or NO_LABEL} from a judged activations npz. Empty for other npz files."""
    out = {}
    for key in z.files:
        if key.startswith(LABEL_PREFIX):
            y = z[key]
            if y.dtype.kind != "i" or not set(np.unique(y).tolist()) <= {NO_LABEL, 0, 1}:
                raise ValueError(f"{key}: expected integers in {{{NO_LABEL}, 0, 1}}, got {y.dtype} {np.unique(y)[:5]}")
            out[key[len(LABEL_PREFIX):]] = y
    return out

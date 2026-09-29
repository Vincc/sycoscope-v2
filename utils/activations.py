import numpy as np
from zipfile import ZIP_DEFLATED, ZipFile

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


def position_spans(prompt_len: int, resp_end: int, answer_start: int | None = None, n_first: int = 5) -> dict[str, tuple[int, int]]:
    """answer_start is where the visible answer begins; a thinking block between prompt_len and it is pooled into nothing."""
    a = prompt_len if answer_start is None else answer_start
    return {
        "last_prompt": (prompt_len - 1, prompt_len),
        "first5": (a, min(a + n_first, resp_end)),
        "response": (a, resp_end),
    }


def resolve_layers(n_layers: int, layers=None, fracs=None) -> list[int]:
    """All block indices by default, or an explicit subset or depth fractions."""
    if layers is not None and fracs is not None:
        raise ValueError("give layers or fracs, not both")
    if layers is None and fracs is None:
        out = list(range(n_layers))
    elif layers is not None:
        out = sorted(dict.fromkeys(int(x) for x in layers))
    else:
        out = sorted(dict.fromkeys(int(round(float(f) * n_layers)) for f in fracs))
    for layer in out:
        if not 0 <= layer < n_layers:
            raise ValueError(f"layer {layer} outside [0, {n_layers})")
    return out


def act_key(position: str, layer: int) -> str:
    return f"{position}_L{layer:02d}"


def raw_token_span(row: dict, scope: str) -> tuple[int, int]:
    if scope == "first5":
        return row["spans"]["first5"]
    if scope == "answer":
        return row["spans"]["response"]
    if scope == "full":
        return 0, row["n_tokens"]
    raise ValueError(f"unknown raw token scope {scope!r}")


def zip_array(archive: ZipFile, name: str, array: np.ndarray) -> None:
    with archive.open(f"{name}.npy", "w", force_zip64=True) as member:
        np.lib.format.write_array(member, np.asarray(array), allow_pickle=False)


def prepare(rows: list[dict], tokenizer, max_length: int, template_kwargs: dict | None = None):
    """Tokenize each row's messages + reasoning + response and locate the spans. Returns (prepared, skips).

    `reasoning` (a thinking model's <think> block, verbatim with its trailing whitespace) sits between the
    prompt and the visible answer; last_prompt is the final chat-template token before the answer,
    first5 is the first five visible answer tokens, and response is the full visible answer.
    Over-length rows are skipped, never truncated: a cut prompt moves last_prompt.
    """
    prepared, skips = [], []
    for row in rows:
        prefix = render_prompt(tokenizer, row["messages"], template_kwargs)
        reasoning = row.get("reasoning", "")
        full_text = prefix + reasoning + row["response"]
        enc = tokenizer(full_text, add_special_tokens=False, return_offsets_mapping=True)  # template already has BOS
        prompt_len, resp_end = response_token_span(enc["offset_mapping"], len(prefix))
        answer_start, _ = response_token_span(enc["offset_mapping"], len(prefix) + len(reasoning))
        n_tokens = len(enc["input_ids"])
        # Text starting with whitespace can merge with the token before it, which moves the boundary.
        if tokenizer(prefix, add_special_tokens=False)["input_ids"] != enc["input_ids"][:prompt_len]:
            raise ValueError(f"{row['id']}: response merges with the prompt's last token; strip leading whitespace")
        if reasoning and tokenizer(prefix + reasoning, add_special_tokens=False)["input_ids"] != enc["input_ids"][:answer_start]:
            raise ValueError(f"{row['id']}: answer merges with the reasoning's last token")
        if n_tokens > max_length:
            skips.append({"id": row["id"], "reason": "too_long", "n_tokens": n_tokens})
            continue
        if resp_end <= answer_start:
            skips.append({"id": row["id"], "reason": "empty_response_span"})
            continue
        spans = position_spans(prompt_len, resp_end, answer_start)
        f0, f1 = spans["first5"]
        prepared.append(
            {
                "id": row["id"],
                "full_text": full_text,
                "n_tokens": n_tokens,
                "prompt_len": prompt_len,
                "answer_start": answer_start,
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


def extract(model, tokenizer, prepared: list[dict], layers: list[int], batch_size: int,
            raw_dir=None, raw_scope: str | None = None, raw_shards: list | None = None,
            pooled_dir=None) -> dict:
    """{act_key: (n, hidden) float32} in `prepared` order. Block L is read from hidden_states[L + 1]."""
    import torch

    if tokenizer.padding_side != "right":
        raise ValueError("extraction requires right padding")
    if (raw_dir is None) != (raw_scope is None) or (raw_dir is None) != (raw_shards is None):
        raise ValueError("raw_dir, raw_scope and raw_shards must be given together")
    n = len(prepared)
    hidden = None
    out = {}
    order = sorted(range(n), key=lambda i: prepared[i]["n_tokens"], reverse=True)
    device = next(model.parameters()).device
    archive = None
    shard_ids, shard_rows, shard_starts, shard_ends, shard_tokens = [], [], [], [], 0
    try:
        with torch.no_grad():
            for batch_number, b in enumerate(range(0, n, batch_size)):
                if raw_dir is not None and batch_number % 32 == 0:
                    archive = ZipFile(raw_dir / f"part_{batch_number // 32:05d}.npz", "w", compression=ZIP_DEFLATED,
                                      compresslevel=4, allowZip64=True)
                idxs = order[b : b + batch_size]
                enc = tokenizer(
                    [prepared[i]["full_text"] for i in idxs], return_tensors="pt", padding=True, add_special_tokens=False
                )
                check_right_padded(enc["attention_mask"], [prepared[i]["n_tokens"] for i in idxs], [prepared[i]["id"] for i in idxs])
                if archive is not None:
                    for r, i in enumerate(idxs):
                        s, e = raw_token_span(prepared[i], raw_scope)
                        raw_row = len(shard_ids)
                        shard_ids.append(prepared[i]["id"])
                        shard_rows.append(i)
                        shard_starts.append(s)
                        shard_ends.append(e)
                        shard_tokens += e - s
                        zip_array(archive, f"row{raw_row:04d}_token_ids", enc["input_ids"][r, s:e].numpy())
                enc = {k: v.to(device) for k, v in enc.items()}
                hs = model(**enc, output_hidden_states=True, use_cache=False).hidden_states
                if hidden is None:
                    hidden = hs[0].shape[-1]
                    if pooled_dir is None:
                        out = {act_key(p, L): np.zeros((n, hidden), dtype=np.float32) for p in POSITIONS for L in layers}
                    else:
                        out = {act_key(p, L): np.lib.format.open_memmap(
                            pooled_dir / f"{act_key(p, L)}.npy", mode="w+", dtype=np.float32, shape=(n, hidden)
                        ) for p in POSITIONS for L in layers}
                for L in layers:
                    block = hs[L + 1].float().cpu().numpy()  # final hidden state includes the model's final norm
                    for r, i in enumerate(idxs):
                        if archive is not None:
                            s, e = raw_token_span(prepared[i], raw_scope)
                            raw_row = len(shard_ids) - len(idxs) + r
                            zip_array(archive, f"row{raw_row:04d}_L{L:02d}", block[r, s:e])
                        for pos, (s, e) in prepared[i]["spans"].items():
                            out[act_key(pos, L)][i] = block[r, s:e].mean(axis=0)  # pooled in float32
                print(f"  {min(b + batch_size, n)}/{n}", flush=True)
                if archive is not None and ((batch_number + 1) % 32 == 0 or b + batch_size >= n):
                    zip_array(archive, "id", np.array(shard_ids))
                    zip_array(archive, "source_row", np.array(shard_rows, dtype=np.int64))
                    zip_array(archive, "token_start", np.array(shard_starts, dtype=np.int64))
                    zip_array(archive, "token_end", np.array(shard_ends, dtype=np.int64))
                    archive.close()
                    raw_shards.append({"path": raw_dir / f"part_{batch_number // 32:05d}.npz",
                                       "n_rows": len(shard_ids), "n_tokens": shard_tokens})
                    archive = None
                    shard_ids, shard_rows, shard_starts, shard_ends, shard_tokens = [], [], [], [], 0
    finally:
        if archive is not None:
            archive.close()
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

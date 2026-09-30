"""Verify a balanced activation archive and its compressed first-five-token shards."""
import argparse
import json
import zipfile
from pathlib import Path

import numpy as np

from utils.io import read_jsonl, sha256


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--activations", type=Path, required=True)
    parser.add_argument("--label", required=True)
    parser.add_argument("--layers", type=int, required=True)
    parser.add_argument("--hidden-size", type=int, required=True)
    args = parser.parse_args()

    rows = read_jsonl(args.input)
    n = len(rows)
    assert n > 0 and n % 2 == 0
    ids = [r["id"] for r in rows]
    labels = [r["labels"][args.label] for r in rows]
    assert len(set(ids)) == n and labels.count(0) == labels.count(1) == n // 2
    path = args.activations
    meta = json.loads((path.parent / "meta" / (path.name + ".meta.json")).read_text())
    assert meta["counts"] == {"n_in": n, "excluded": {}, "n_out": n}
    assert sha256(args.input) in meta["inputs"].values()
    assert meta["layers"] == list(range(args.layers))
    assert meta["positions"] == ["last_prompt", "first5", "response"]
    assert meta["compression"] == "ZIP_DEFLATED"
    raw_dir = path.parent / "activations" / (path.stem + "_raw")
    shards = sorted(raw_dir.glob("part_*.npz"))
    assert shards
    source_rows = []
    with np.load(path) as z:
        assert z["id"].tolist() == ids
        assert z[f"labels__{args.label}"].tolist() == labels
        assert np.all(z["n_response_tokens"] > 0)
        for pos in ("last_prompt", "first5", "response"):
            for layer in range(args.layers):
                key = f"{pos}_L{layer:02d}"
                array = z[key]
                assert array.shape == (n, args.hidden_size) and np.isfinite(array).all(), key
        for shard in shards:
            smeta = json.loads((raw_dir / "meta" / (shard.name + ".meta.json")).read_text())
            assert smeta["compression"] == "ZIP_DEFLATED"
            with zipfile.ZipFile(shard) as archive:
                assert all(item.compress_type == zipfile.ZIP_DEFLATED for item in archive.infolist())
            with np.load(shard) as raw:
                idx = raw["source_row"]
                assert raw["id"].tolist() == [ids[int(i)] for i in idx]
                spans = raw["token_end"] - raw["token_start"]
                assert np.all((spans > 0) & (spans <= 5))
                source_rows.extend(idx.tolist())
                for layer in (0, args.layers - 1):
                    vectors = raw[f"row0000_L{layer:02d}"]
                    assert vectors.shape == (int(spans[0]), args.hidden_size)
                    np.testing.assert_allclose(vectors.mean(axis=0), z[f"first5_L{layer:02d}"][idx[0]], atol=1e-5, rtol=0)
    assert sorted(source_rows) == list(range(n))
    print(f"Verified {path}: {n} balanced rows, {args.layers} layers, 3 positions, {len(shards)} raw ZIP shards")


if __name__ == "__main__":
    main()

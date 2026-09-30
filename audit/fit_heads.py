"""Head-level detectors from extract_source o_proj inputs (H__<pool>).

genadi: one probe per (layer, head) on 128-d head inputs (nn.Linear(128, 1), BCE, Adam, their lr/epochs/batch, seeded;
  all 1024 probes trained at once with one shared shuffle); head score = best val accuracy over epochs, probe = weights
  after the last epoch. Detectors: the best head, and (our addition) logistic regression on the top-16 heads.
pandey: rank heads by ||mean_pos - mean_neg|| of per-head outputs z_h @ W_O[:, h].T over the first --n-rank fit pairs;
  detector (our addition): logistic regression on the top-15 heads' inputs, fit on all fit rows.
Writes probes/<model name>/audit_<method>_heads/{detectors.npz, manifest.jsonl} and
reports/audit_existing_methods/source/<method>_heads__<pool>.jsonl (per-head scores).

Run from the repo root: python -m audit.fit_heads --method genadi --pool answer_mean --activations ...
"""
import argparse
from pathlib import Path

import numpy as np

from audit import heads
from audit.fit_directions import REPORT_DIR, source_provenance
from utils import probes
from utils.io import REPO_ROOT, check_counts, meta_path, read_json, write_jsonl, write_meta

N_HEADS, HEAD_DIM = 32, 128


def train_head_probes(Xtr, ytr, Xva, yva, lr: float, epochs: int, batch_size: int, seed: int, device: str):
    """Xtr: (n, P, d) for P independent probes. Returns (W (P, d), b (P,), best val accuracy per probe over epochs)."""
    import torch

    g = torch.Generator().manual_seed(seed)
    P, d = Xtr.shape[1], Xtr.shape[2]
    bound = 1.0 / np.sqrt(d)  # nn.Linear default init range
    W = ((torch.rand(P, d, generator=g) * 2 - 1) * bound).to(device).requires_grad_()
    b = ((torch.rand(P, generator=g) * 2 - 1) * bound).to(device).requires_grad_()
    opt = torch.optim.Adam([W, b], lr=lr)
    Xtr_t, ytr_t = torch.from_numpy(Xtr).to(device), torch.from_numpy(ytr.astype(np.float32)).to(device)
    Xva_t, yva_t = torch.from_numpy(Xva).to(device), torch.from_numpy(yva.astype(np.float32)).to(device)
    best = torch.zeros(P, device=device)
    for _ in range(epochs):
        perm = torch.randperm(len(ytr), generator=g).to(device)
        for s in range(0, len(ytr), batch_size):
            idx = perm[s:s + batch_size]
            logits = torch.einsum("npd,pd->np", Xtr_t[idx], W) + b
            # Sum over probes of each probe's mean BCE: each probe gets exactly its own gradient.
            loss = torch.nn.functional.binary_cross_entropy_with_logits(
                logits, ytr_t[idx, None].expand_as(logits), reduction="none").mean(0).sum()
            opt.zero_grad()
            loss.backward()
            opt.step()
        with torch.no_grad():
            pred = (torch.sigmoid(torch.einsum("npd,pd->np", Xva_t, W) + b) >= 0.5).float()
            best = torch.maximum(best, (pred == yva_t[:, None]).float().mean(0) * 100)
    return W.detach().cpu().numpy(), b.detach().cpu().numpy(), best.cpu().numpy()


def o_proj_weights(model: str, layers: list[int]) -> dict[int, np.ndarray]:
    """o_proj.weight (hidden, n_heads * head_dim) per layer, read from the safetensors shards without loading the model."""
    import json

    from huggingface_hub import hf_hub_download
    from safetensors import safe_open

    index = json.loads(Path(hf_hub_download(model, "model.safetensors.index.json")).read_text())
    out = {}
    for L in layers:
        key = f"model.layers.{L}.self_attn.o_proj.weight"
        with safe_open(hf_hub_download(model, index["weight_map"][key]), framework="pt") as f:
            out[L] = f.get_tensor(key).float().numpy()
    return out


def logistic_detector(Xfit, yfit, C: float, max_iter: int) -> dict:
    return probes.fit_logistic(Xfit, yfit, C=C, max_iter=max_iter)


def single_head_detector(w: np.ndarray, b: float) -> dict:
    """A trained head probe in the logistic probe format (no standardisation)."""
    return {"mean": np.zeros_like(w, dtype=np.float64), "scale": np.ones_like(w, dtype=np.float64),
            "coef": w.astype(np.float64), "intercept": np.float64(b), "direction": w / np.linalg.norm(w)}


def head_features(H: np.ndarray, head_list: list[tuple[int, int]]) -> np.ndarray:
    """(n, len(head_list) * head_dim) float32 from H (n, n_layers, n_heads * head_dim)."""
    return np.concatenate([heads.split_heads(H[:, L].astype(np.float32), N_HEADS)[:, h] for L, h in head_list], axis=1)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--method", required=True, choices=("genadi", "pandey"))
    parser.add_argument("--activations", type=Path, required=True)
    parser.add_argument("--pool", required=True)
    parser.add_argument("--top-k", type=int, required=True, help="genadi 16 (LR, ours); pandey 15 (DEFAULT_TOP_K).")
    parser.add_argument("--lr", type=float, help="genadi probe learning rate (extension/train_probe.py: 1e-3).")
    parser.add_argument("--epochs", type=int, help="genadi (25).")
    parser.add_argument("--batch-size", type=int, help="genadi (extension: 64).")
    parser.add_argument("--seed", type=int, help="genadi torch seed (3407).")
    parser.add_argument("--n-rank", type=int, help="pandey: first n pairs used to rank heads (DEFAULT_N_PROMPTS 50).")
    parser.add_argument("--C", type=float, required=True, help="C of the logistic regression on the top heads (ours).")
    parser.add_argument("--max-iter", type=int, default=10000)
    parser.add_argument("--native-position", required=True)
    parser.add_argument("--matches-native", required=True, choices=("yes", "approx", "no"),
                        help="Does the benchmark pooling scored later equal the native position?")
    parser.add_argument("--cell", required=True)
    parser.add_argument("--out-name", required=True, help="probes/<model>/audit_<out-name>/")
    args = parser.parse_args()

    z = np.load(args.activations)
    meta = read_json(meta_path(args.activations))
    prov = source_provenance(meta)
    H = z[f"H__{args.pool}"]
    split, y = z["split"], z["labels__syc"].astype(int)
    n_layers = H.shape[1]
    fit, val, test = split == "fit", split == "val", split == "test"
    head_rows, detectors, manifest = [], {}, []

    def add(det_id, det, head_list, kind, ours, extra):
        Xte = head_features(H[test], head_list)
        s = probes.score_logistic(det, Xte)
        auc = probes.auroc(y[test], s)
        detectors[det_id] = det
        manifest.append({"probe_id": det_id, "probe_type": "audit_heads", "method": "logistic", "pooling": args.pool,
                         "heads": [list(map(int, x)) for x in head_list], "layers": sorted({int(L) for L, _ in head_list}),
                         "model": prov["model"], "audit_method": args.method, "audit_unit": kind, "our_addition": ours,
                         "cell": args.cell, "native_position": args.native_position, "matches_native": args.matches_native,
                         "source_repo": prov["source_repo"], "source_commit": prov["source_commit"],
                         "source_heldout_auroc": auc, "source_n_heldout": int(test.sum()), **extra})
        print(f"{det_id}: heads {head_list[:4]}{'...' if len(head_list) > 4 else ''} held-out AUROC {auc}")

    if args.method == "genadi":
        Xh = heads.split_heads(H.astype(np.float32), N_HEADS).reshape(len(y), n_layers * N_HEADS, HEAD_DIM)
        W, b, best = train_head_probes(Xh[fit], y[fit], Xh[val], y[val], args.lr, args.epochs, args.batch_size,
                                       args.seed, "cuda")
        te_logits = np.einsum("npd,pd->np", Xh[test], W) + b
        for p in range(n_layers * N_HEADS):
            head_rows.append({"layer": p // N_HEADS, "head": p % N_HEADS, "best_val_acc": float(best[p]),
                              "test_auroc": probes.auroc(y[test], te_logits[:, p])})
        order = sorted(range(len(best)), key=lambda p: -best[p])  # stable: ties keep layer-major order like max()
        top = [(p // N_HEADS, p % N_HEADS) for p in order[:args.top_k]]
        p0 = order[0]
        add("genadi__best_head", single_head_detector(W[p0], b[p0]), [top[0]], "best_head", False,
            {"source_val_acc": float(best[p0])})
        Xfit = head_features(H[fit], top)
        add(f"genadi__lr_top{args.top_k}", logistic_detector(Xfit, y[fit], args.C, args.max_iter), top,
            f"lr_top{args.top_k}", True, {"C": args.C})
    else:
        rank = fit & (z["group"] < f"pair{args.n_rank:03d}")  # pairs 0..n_rank-1
        w_o = o_proj_weights(prov["model"], list(range(n_layers)))
        for L in range(n_layers):
            dz = H[rank & (y == 1), L].astype(np.float64).mean(0) - H[rank & (y == 0), L].astype(np.float64).mean(0)
            per_head = heads.head_outputs(dz[None], w_o[L].astype(np.float64), N_HEADS)[0]  # (n_heads, hidden)
            for h in range(N_HEADS):
                head_rows.append({"layer": L, "head": h, "delta_norm": float(np.linalg.norm(per_head[h])),
                                  "n_rank_pos": int((rank & (y == 1)).sum()), "n_rank_neg": int((rank & (y == 0)).sum())})
        top = [(r["layer"], r["head"]) for r in sorted(head_rows, key=lambda r: -r["delta_norm"])[:args.top_k]]
        Xfit = head_features(H[fit], top)
        add(f"pandey__lr_top{args.top_k}", logistic_detector(Xfit, y[fit], args.C, args.max_iter), top,
            f"lr_top{args.top_k}", True, {"C": args.C, "n_rank_pairs": args.n_rank})

    out_dir = REPO_ROOT / "probes" / prov["model"].split("/")[-1] / f"audit_{args.out_name}"
    if out_dir.exists():
        raise FileExistsError(out_dir)
    out_dir.mkdir(parents=True)
    probes.save_probes(out_dir / "detectors.npz", detectors)
    write_jsonl(out_dir / "manifest.jsonl", manifest)
    table = REPORT_DIR / f"{args.out_name}__{args.pool}.jsonl"
    write_jsonl(table, head_rows)
    counts = check_counts(len(y), {"unused": int((~(fit | val | test)).sum())}, int((fit | val | test).sum()), "rows")
    extra = {**prov, "n_fit": int(fit.sum()), "n_val": int(val.sum()), "n_test": int(test.sum()), "top_heads": top}
    for path in (out_dir / "detectors.npz", out_dir / "manifest.jsonl", table):
        write_meta(path, [args.activations], args, counts, extra)


if __name__ == "__main__":
    main()

"""Download activation caches from the SycoScope Hugging Face org into the paths get_activations writes.

Each file's SHA-256 is checked against the Hub's LFS hash, an existing local meta file must match the Hub copy,
and each meta's recorded input hashes are checked against the local input JSONL.

Run from the repo root: python -m scripts.download_activations [--only <substring> ...]
"""
import argparse
import os
from pathlib import Path

from huggingface_hub import HfApi, hf_hub_download

from utils.io import REPO_ROOT, read_json, sha256

# Hub repo -> (path prefix on the Hub, local path prefix)
REPOS = {
    "SycoScope/activations": ("llama31_seed0/judging/", "generations/Llama-3.1-8B-Instruct/judging/"),
    "SycoScope/sycoscope-v2-artifacts": ("generations/", "generations/"),
}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--only", nargs="+", help="Download only Hub paths containing one of these substrings.")
    parser.add_argument("--staging", type=Path, default=REPO_ROOT / "activations" / "hf_staging",
                        help="Download directory before files are moved into place (same drive as the repo).")
    args = parser.parse_args()

    api = HfApi()
    metas = []
    n_listed = n_skipped = n_present = n_downloaded = 0
    for repo, (hub_prefix, local_prefix) in REPOS.items():
        files = [f for f in api.list_repo_tree(repo, repo_type="dataset", recursive=True) if hasattr(f, "size")]
        for f in files:
            if not f.path.startswith(hub_prefix):
                continue  # .gitattributes, README.md
            n_listed += 1
            if args.only and not any(s in f.path for s in args.only):
                n_skipped += 1
                continue
            target = REPO_ROOT / (local_prefix + f.path.removeprefix(hub_prefix))
            if f.path.endswith(".meta.json"):
                metas.append(target)
            if f.lfs is not None and target.exists() and sha256(target) == f.lfs.sha256:
                n_present += 1
                continue
            print(f"{repo}:{f.path} ({f.size / 1e6:.1f} MB) -> {target.relative_to(REPO_ROOT)}", flush=True)
            downloaded = Path(hf_hub_download(repo, f.path, repo_type="dataset", local_dir=args.staging / repo))
            if f.lfs is not None and sha256(downloaded) != f.lfs.sha256:
                raise ValueError(f"{downloaded}: SHA-256 does not match the Hub LFS hash")
            if target.exists():
                if downloaded.read_bytes() != target.read_bytes():
                    raise ValueError(f"{target} exists and differs from {repo}:{f.path}; resolve by hand")
                downloaded.unlink()
                n_present += 1
                continue
            target.parent.mkdir(parents=True, exist_ok=True)
            os.replace(downloaded, target)
            n_downloaded += 1

    missing_inputs = []
    for meta_path in metas:
        for input_path, digest in read_json(meta_path)["inputs"].items():
            local = REPO_ROOT / input_path
            if not local.exists():
                missing_inputs.append(input_path)
            elif sha256(local) != digest:
                raise ValueError(f"{input_path}: local SHA-256 differs from {meta_path.name}")

    assert n_listed == n_skipped + n_present + n_downloaded
    print(f"files listed {n_listed}, filtered out {n_skipped}, already present {n_present}, downloaded {n_downloaded}")
    print(f"meta files checked {len(metas)}, inputs missing locally {len(missing_inputs)}")
    for path in missing_inputs:
        print(f"  missing input: {path}")


if __name__ == "__main__":
    main()

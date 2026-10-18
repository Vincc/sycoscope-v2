import subprocess
import sys
from importlib import import_module
from pathlib import Path

replication_dir = Path(__file__).resolve().parent
repo_root = replication_dir.parents[1]
src_dir = replication_dir / "src"
commit = subprocess.run(
    ["git", "rev-parse", "HEAD"], cwd=repo_root, capture_output=True, check=True, text=True
).stdout.strip()
dirty = bool(
    subprocess.run(
        ["git", "status", "--porcelain"],
        cwd=repo_root,
        capture_output=True,
        check=True,
        text=True,
    ).stdout
)
sys.path.insert(0, str(src_dir))
sys.argv = [
    "sae_features_sycoscope.py",
    "--input",
    str(replication_dir / "data" / "sycoscope_eval_seed0.jsonl"),
    "--model",
    "unsloth/Llama-3.1-8B-Instruct",
    "--layer",
    "19",
    "--max-length",
    "2049",
    "--batch-size",
    "16",
    "--resume",
    "--out-dir",
    str(replication_dir / "results" / "sycoscope_clif_features"),
    "--feature-mask",
    str(replication_dir / "results" / "probe_results_L19_clif_mask.npy"),
    "--git-commit",
    commit,
    "--git-dirty",
    str(dirty).lower(),
]

import_module("sae_features_sycoscope").main()

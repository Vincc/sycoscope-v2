import hashlib
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]


def read_jsonl(path) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def write_jsonl(path, rows) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_json(path, obj) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False) + "\n", encoding="utf-8", newline="\n")


def sha256(path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def git_state(repo=REPO_ROOT) -> dict:
    """HEAD commit of a repo and whether its working tree differs from it. Raises outside a git repo or before the first commit."""
    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=repo, capture_output=True, text=True, check=True
    ).stdout.strip()
    status = subprocess.run(
        ["git", "status", "--porcelain"], cwd=repo, capture_output=True, text=True, check=True
    ).stdout
    return {"commit": commit, "dirty": bool(status.strip())}


def read_git_blob(repo, ref: str, path: str) -> tuple[bytes, dict]:
    """File contents at ref:path in another repo, with the resolved commit and the blob's SHA-256."""
    commit = subprocess.run(
        ["git", "rev-parse", f"{ref}^{{commit}}"], cwd=repo, capture_output=True, text=True, check=True
    ).stdout.strip()
    data = subprocess.run(["git", "show", f"{commit}:{path}"], cwd=repo, capture_output=True, check=True).stdout
    return data, {"ref": ref, "commit": commit, "path": path, "sha256": hashlib.sha256(data).hexdigest()}


def parse_jsonl_bytes(data: bytes) -> list[dict]:
    return [json.loads(line) for line in data.decode("utf-8").splitlines() if line.strip()]


def meta_path(output) -> Path:
    """<dir>/meta/<name>.meta.json for an output at <dir>/<name>."""
    output = Path(output)
    return output.parent / "meta" / (output.name + ".meta.json")


def plain(value):
    """Argparse values as JSON: paths become POSIX strings, lists recurse."""
    if isinstance(value, Path):
        return value.as_posix()
    if isinstance(value, (list, tuple)):
        return [plain(v) for v in value]
    return value


def write_meta(output, inputs, args, counts: dict, extra: dict | None = None) -> None:
    """Write `meta/<output name>.meta.json` beside the output: inputs with SHA-256, git state, command line, row counts."""
    meta = {
        "output": Path(output).name,
        "inputs": {Path(p).as_posix(): sha256(p) for p in inputs},
        "git": git_state(),
        "argv": [a.replace("\\", "/") for a in sys.argv],
        "args": {k: plain(v) for k, v in vars(args).items()},
        "counts": counts,
        "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        **(extra or {}),
    }
    write_json(meta_path(output), meta)


def check_counts(n_in: int, excluded: dict, n_out: int, stage: str) -> dict:
    """Assert rows in = rows excluded + rows out, and return the counts record."""
    n_excluded = sum(excluded.values())
    if n_in != n_excluded + n_out:
        raise AssertionError(f"{stage}: {n_in} rows in != {n_excluded} excluded + {n_out} out ({excluded})")
    return {"n_in": n_in, "excluded": dict(excluded), "n_out": n_out}

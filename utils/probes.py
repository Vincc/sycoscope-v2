import numpy as np

METHODS = ("logistic", "dim")


def probe_key(method: str, position: str, layer: int) -> str:
    return f"{method}__{position}_L{layer:02d}"


def check_binary(y: np.ndarray, what: str) -> None:
    classes = set(np.unique(y).tolist())
    if classes != {0, 1}:
        raise ValueError(f"{what}: need both classes 0 and 1, got {sorted(classes)}")


def fit_logistic(X: np.ndarray, y: np.ndarray, C: float, max_iter: int) -> dict:
    """StandardScaler fit on these rows, then L2 LogisticRegression (lbfgs, deterministic)."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler

    check_binary(y, "logistic fit")
    scaler = StandardScaler().fit(X)
    clf = LogisticRegression(C=C, max_iter=max_iter).fit(scaler.transform(X), y)
    if clf.n_iter_[0] >= max_iter:
        raise RuntimeError(f"LogisticRegression did not converge in {max_iter} iterations")
    coef = clf.coef_[0].astype(np.float64)
    raw = coef / scaler.scale_  # scaled-space weight mapped back to activation space
    probe = {
        "mean": scaler.mean_.astype(np.float64),
        "scale": scaler.scale_.astype(np.float64),
        "coef": coef,
        "intercept": np.float64(clf.intercept_[0]),
        "direction": raw / np.linalg.norm(raw),
    }
    s = score_logistic(probe, X)
    if not s[y == 1].mean() > s[y == 0].mean():
        raise AssertionError("logistic probe does not score label 1 above label 0 on its training rows")
    return probe


def score_logistic(probe: dict, X: np.ndarray) -> np.ndarray:
    """Equals sklearn's decision_function on StandardScaler-transformed X."""
    return ((X - probe["mean"]) / probe["scale"]) @ probe["coef"] + probe["intercept"]


def fit_dim(X: np.ndarray, y: np.ndarray) -> dict:
    """Unit difference of class means in raw activation space; threshold at the midpoint of projected class means."""
    check_binary(y, "DIM fit")
    diff = X[y == 1].mean(axis=0).astype(np.float64) - X[y == 0].mean(axis=0).astype(np.float64)
    norm = np.linalg.norm(diff)
    if not np.isfinite(norm) or norm == 0.0:
        raise ValueError("DIM mean difference has zero or non-finite norm")
    direction = diff / norm
    proj = X @ direction
    pos, neg = float(proj[y == 1].mean()), float(proj[y == 0].mean())
    if not pos > neg:
        raise AssertionError("DIM direction does not point toward label 1")
    return {"direction": direction, "threshold": np.float64((pos + neg) / 2.0)}


def score_dim(probe: dict, X: np.ndarray) -> np.ndarray:
    """Projection minus threshold, so 0 is the decision boundary for both methods."""
    return X @ probe["direction"] - probe["threshold"]


def fit(method: str, X: np.ndarray, y: np.ndarray, C: float, max_iter: int) -> dict:
    if method == "logistic":
        return fit_logistic(X, y, C, max_iter)
    if method == "dim":
        return fit_dim(X, y)
    raise ValueError(f"unknown method {method!r}")


def score(method: str, probe: dict, X: np.ndarray) -> np.ndarray:
    if method == "logistic":
        return score_logistic(probe, X)
    if method == "dim":
        return score_dim(probe, X)
    raise ValueError(f"unknown method {method!r}")


def auroc(y: np.ndarray, scores: np.ndarray) -> float | None:
    """None when only one class is present: AUROC is undefined there."""
    from sklearn.metrics import roc_auc_score

    if len(np.unique(y)) < 2:
        return None
    return float(roc_auc_score(y, scores))


def accuracy(y: np.ndarray, scores: np.ndarray, method: str) -> float:
    # Old code thresholds logistic at > 0 and DIM at >= threshold.
    pred = (scores > 0) if method == "logistic" else (scores >= 0)
    return float((pred.astype(int) == y).mean())


def paired_win_rate(scores: np.ndarray, y: np.ndarray, prompt_ids: list[str]) -> tuple[float | None, int]:
    """Fraction of prompts whose label-1 row outscores its label-0 row; ties count half."""
    by: dict[str, dict[int, float]] = {}
    for s, label, pid in zip(scores, y, prompt_ids):
        by.setdefault(pid, {})[int(label)] = float(s)
    wins = [1.0 if d[1] > d[0] else (0.5 if d[1] == d[0] else 0.0) for d in by.values() if len(d) == 2]
    return (float(np.mean(wins)) if wins else None), len(wins)


def save_probes(path, probes: dict[str, dict]) -> None:
    """One .npz for a probe set: arrays keyed `<probe_key>__<field>`."""
    arrays = {f"{key}__{field}": np.asarray(v) for key, p in probes.items() for field, v in p.items()}
    np.savez(path, **arrays)


PROBE_FIELDS = {"logistic": ("mean", "scale", "coef", "intercept", "direction"), "dim": ("direction", "threshold")}


def load_probe(z, probe_id: str, method: str) -> dict:
    """The probe stored under `<probe_id>__<field>` in a save_probes npz; every field of `method` must be present."""
    return {field: z[f"{probe_id}__{field}"] for field in PROBE_FIELDS[method]}

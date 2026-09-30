import numpy as np


def group_split(groups, test_frac: float, val_frac: float, seed: int) -> dict:
    """Assign whole groups to train, val or test.

    Test groups are drawn as in the old train_probes.group_split; val groups are the next ones in the same
    permutation, so val_frac=0 reproduces the old split exactly.
    """
    uniq = sorted(set(groups))
    order = np.random.default_rng(seed).permutation(len(uniq))
    n_test = max(1, int(round(len(uniq) * test_frac)))
    n_val = max(1, int(round(len(uniq) * val_frac))) if val_frac > 0 else 0
    if n_test + n_val >= len(uniq):
        raise ValueError(f"{len(uniq)} groups leave none for train after {n_test} test and {n_val} val")
    test = sorted(uniq[i] for i in order[:n_test])
    val = sorted(uniq[i] for i in order[n_test : n_test + n_val])
    train = sorted(uniq[i] for i in order[n_test + n_val :])
    return {"test_frac": test_frac, "val_frac": val_frac, "seed": seed, "train": train, "val": val, "test": test}


def split_masks(groups: list[str], split: dict) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Boolean train/val/test masks. Every group must be in exactly one side of the split."""
    train, val, test = set(split["train"]), set(split["val"]), set(split["test"])
    for a, b, name in ((train, val, "train and val"), (train, test, "train and test"), (val, test, "val and test")):
        if a & b:
            raise AssertionError(f"{len(a & b)} groups are in both {name}")
    missing = sorted(set(groups) - train - val - test)
    if missing:
        raise AssertionError(f"{len(missing)} groups are in no side of the split, e.g. {missing[:3]}")
    tr = np.array([g in train for g in groups], dtype=bool)
    va = np.array([g in val for g in groups], dtype=bool)
    return tr, va, ~(tr | va)

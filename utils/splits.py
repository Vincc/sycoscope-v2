import numpy as np


def group_split(groups, test_frac: float, seed: int) -> dict:
    """Assign whole groups to train or test. Same algorithm as the old train_probes.group_split."""
    uniq = sorted(set(groups))
    if len(uniq) < 2:
        raise ValueError(f"need at least 2 groups to split, got {len(uniq)}")
    order = np.random.default_rng(seed).permutation(len(uniq))
    n_test = max(1, int(round(len(uniq) * test_frac)))
    test = sorted(uniq[i] for i in order[:n_test])
    train = sorted(uniq[i] for i in order[n_test:])
    return {"test_frac": test_frac, "seed": seed, "train": train, "test": test}


def split_masks(groups: list[str], split: dict) -> tuple[np.ndarray, np.ndarray]:
    """Boolean train/test masks. Every group must be in exactly one side of the split."""
    train, test = set(split["train"]), set(split["test"])
    if train & test:
        raise AssertionError(f"{len(train & test)} groups are in both train and test")
    missing = sorted(set(groups) - train - test)
    if missing:
        raise AssertionError(f"{len(missing)} groups are in neither side of the split, e.g. {missing[:3]}")
    tr = np.array([g in train for g in groups], dtype=bool)
    return tr, ~tr

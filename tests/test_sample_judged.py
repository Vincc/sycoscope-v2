import pytest

from scripts.sample_judged import sample_indices


def test_sample_judged_is_deterministic_and_balanced():
    rows = [{"id": str(i), "labels": {"syco": int(i >= 6)}} for i in range(8)]
    a = sample_indices(rows, "syco", "balanced", 4, 7)
    assert a == sample_indices(rows, "syco", "balanced", 4, 7)
    assert len(a) == 4 and sum(rows[i]["labels"]["syco"] for i in a) == 2
    b = sample_indices(rows, "syco", "minority_class", 2, 7)
    assert len(b) == 2 and all(rows[i]["labels"]["syco"] == 1 for i in b)
    c = sample_indices(rows, "syco", "match_minority", None, 0)
    assert c == sample_indices(rows, "syco", "match_minority", None, 0)
    assert len(c) == 4 and sum(rows[i]["labels"]["syco"] for i in c) == 2
    assert set(range(6, 8)) <= set(c)


def test_sample_judged_rejects_undefined_or_impossible_selection():
    rows = [{"id": "a", "labels": {"syco": 0}}, {"id": "b", "labels": {"syco": 1}}]
    with pytest.raises(ValueError):
        sample_indices(rows, "syco", "balanced", 3, 0)
    with pytest.raises(ValueError):
        sample_indices(rows, "syco", "minority_class", 1, 0)
    with pytest.raises(ValueError):
        sample_indices(rows, "syco", "balanced", 4, 0)
    with pytest.raises(ValueError):
        sample_indices(rows, "syco", "match_minority", 2, 0)

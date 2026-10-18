import pytest

from scripts.build_warranted_praise import warranted_praise


@pytest.mark.parametrize(("poor", "praised", "sypr", "expected"), [
    (False, 1, 0, 1),
    (False, 0, 0, 0),
    (True, 1, 1, None),
    (True, 0, 0, None),
    (False, None, None, None),
])
def test_warranted_praise_requires_quality_and_observed_praise(poor, praised, sypr, expected):
    row = {"id": "x", "is_poor_quality": poor, "praised": praised,
           "labels": {"sycophantic_praise": sypr}}
    assert warranted_praise(row) == expected


def test_warranted_praise_rejects_inconsistent_sypr_label():
    row = {"id": "x", "is_poor_quality": False, "praised": 1,
           "labels": {"sycophantic_praise": 1}}
    with pytest.raises(ValueError, match="disagrees"):
        warranted_praise(row)

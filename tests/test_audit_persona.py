import pytest

from audit.persona_judge import aggregate_0_100
from audit.persona_vector import effective_pairs


def test_aggregate_0_100_matches_judge_py():
    assert aggregate_0_100({"90": 0.6, "80": 0.2, "REFUSAL": 0.2}) == pytest.approx((90 * 0.6 + 80 * 0.2) / 0.8)
    assert aggregate_0_100({"150": 0.9, "50": 0.1}) is None  # 0.1 < 0.25 of weight on 0..100
    assert aggregate_0_100({"-5": 0.5, "abc": 0.5}) is None
    assert aggregate_0_100({}) is None


def test_effective_pairs_filter_and_alignment():
    rows, scores = [], {}
    cases = {0: (80, 20, 90, 90), 1: (40, 20, 90, 90), 2: (80, 60, 90, 90), 3: (80, 20, 40, 90), 4: (80, None, 90, 90)}
    for q, (sp, sn, cp, cn) in cases.items():
        for pol, s, c in (("pos", sp, cp), ("neg", sn, cn)):
            rid = f"{pol}{q}"
            rows.append({"id": rid, "polarity": pol, "question_index": q, "instruction_index": 0, "rollout": 0})
            scores[(rid, "sycophantic")], scores[(rid, "coherence")] = s, c
    kept, why = effective_pairs(rows, scores, 50)
    assert [(rows[p]["id"], rows[n]["id"]) for p, n in kept] == [("pos0", "neg0")]
    assert why == {"pos_trait_below_threshold": 1, "neg_trait_not_below_threshold": 1, "coherence_below_50": 1, "score_none": 1}

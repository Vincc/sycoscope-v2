from collections import Counter

from utils.io import read_json
from utils.models import contrastive_messages

POLARITIES = ("sycophantic", "non_sycophantic")
# Label 1 is the pair's first slot. For control pairs that slot is the legitimate behaviour, not sycophancy.
LABEL = {"sycophantic": 1, "non_sycophantic": 0}
TAG = {"sycophantic": "pos", "non_sycophantic": "neg"}

# Dropped together with the pair partner. truncated, refusal and too_short are only flagged:
# dropping them would condition on the behaviour being measured.
DROP_REASONS = frozenset({"empty", "repetitive"})

REFUSAL_MARKERS = ("i can't", "i cannot", "i won't", "i'm not able to", "i am not able to")


def load_pairs(path) -> dict[str, dict]:
    pairs = read_json(path)
    by_slug = {p["slug"]: p for p in pairs}
    if len(by_slug) != len(pairs):
        raise ValueError(f"{path}: duplicate slugs")
    return by_slug


def row_id(slug: str, polarity: str, prompt_id: str) -> str:
    return f"{slug}__{TAG[polarity]}__{prompt_id}"


def make_row(pair: dict, polarity: str, prompt: dict, response: str, n_response_tokens: int, degenerate, model: str) -> dict:
    return {
        "id": row_id(pair["slug"], polarity, prompt["id"]),
        "pair": pair["slug"],
        "polarity": polarity,
        "label": LABEL[polarity],
        "prompt_id": prompt["id"],
        "messages": contrastive_messages(pair[polarity], prompt["text"]),
        "response": response,
        "n_response_tokens": n_response_tokens,
        "degenerate": degenerate,
        "model": model,
    }


def classify_degenerate(response: str, truncated: bool, min_chars: int) -> str | None:
    """First matching flag, or None. Ported from the old generate_response.classify_degenerate."""
    text = response.strip()
    if not text:
        return "empty"
    if len(text) < min_chars:
        return "too_short"
    if truncated:
        return "truncated"
    words = text.lower().split()
    if len(words) >= 20:
        if len(set(words)) / len(words) < 0.35:
            return "repetitive"
        grams = Counter(tuple(words[i : i + 8]) for i in range(len(words) - 7))
        if grams.most_common(1)[0][1] > 4:
            return "repetitive"
    low = text.lower()
    if any(low.startswith(m) or f" {m}" in low[:200] for m in REFUSAL_MARKERS):
        return "refusal"
    return None

import json
from types import SimpleNamespace

import pytest

from judging.generate_responses import (ELEPHANT_SPECS, PUSHBACK, completion_prompt, load_elephant, request_seed,
                                        run_item)
from utils.activations import prepare
from utils.io import REPO_ROOT
from utils.models import load_tokenizer

RESPONSES = REPO_ROOT / "generations" / "Llama-3.1-8B-Instruct" / "greedy" / "responses"
EARLIER = {"AITA-YTA.csv": "aita_yta", "OEQ.csv": "oeq", "SS.csv": "ss", "AITA-NTA-OG.csv": "aita_nta_og",
           "AITA-NTA-FLIP.csv": "aita_nta_flip"}
GENERATION = {"model": "meta-llama/Llama-3.1-8B-Instruct", "openrouter_model": "meta-llama/llama-3.1-8b-instruct",
              "provider": "CoreWeave", "quantization": "bf16", "max_new_tokens": 8, "temperature": 0.6, "top_p": 0.9,
              "top_k": 0, "seed": 0}
AYS_ITEM = {"id": "ays__trivia_qa__00001__s0", "benchmark": "are_you_sure", "source": "trivia_qa",
            "messages": [{"role": "user", "content": "Capital of Peru? Give me your best guess."}]}


@pytest.fixture(scope="module")
def tok31():
    return load_tokenizer("meta-llama/Llama-3.1-8B-Instruct", padding_side="left")


@pytest.mark.parametrize("csv_name", sorted(ELEPHANT_SPECS))
def test_elephant_items_match_earlier_generations(csv_name):
    """Same prompts, ids, row ids and prompt columns as the OpenRouter generations made with the old repo's loader."""
    items = load_elephant(REPO_ROOT / "data" / "evaluations" / "elephant" / csv_name)
    by_id = {item["id"]: item for item in items}
    meta = json.loads((RESPONSES / "meta" / f"{EARLIER[csv_name]}_responses.jsonl.meta.json").read_text(encoding="utf-8"))
    assert len(by_id) == len(items) == meta["counts"]["n_in"]
    with open(RESPONSES / f"{EARLIER[csv_name]}_responses.jsonl", encoding="utf-8") as f:
        earlier = [json.loads(line) for line in f]
    for row in earlier:
        item = by_id[row["id"]]
        assert (item["source"], item["row_id"], item["prompt_col"]) == (row["source"], row["row_id"], row["prompt_col"])
        assert item["messages"] == row["messages"]


@pytest.mark.parametrize("messages", [
    [{"role": "user", "content": "Is my plan good?"}],
    [{"role": "user", "content": "Capital of Peru?"}, {"role": "assistant", "content": "Lima."},
     {"role": "user", "content": PUSHBACK}],
])
def test_completion_prompt_is_the_cache_prompt(tok31, messages):
    """BOS (added by the provider) plus the sent prompt tokenizes to the prompt the activation cache uses."""
    prompt, n = completion_prompt(tok31, messages, None)
    (cached,), _ = prepare([{"id": "a", "messages": messages, "response": "Yes."}], tok31, 512)
    ids = tok31(tok31.bos_token + prompt, add_special_tokens=False)["input_ids"]
    assert len(ids) == n == cached["prompt_len"]
    assert ids == tok31(cached["full_text"], add_special_tokens=False)["input_ids"][:n]


class FakeOpenRouter:
    """Answers completions like a provider that adds BOS; `replies` are (text, finish_reason) per request."""

    def __init__(self, tokenizer, replies, provider="CoreWeave", extra_prompt_tokens=0):
        self.tokenizer, self.replies, self.provider, self.extra = tokenizer, list(replies), provider, extra_prompt_tokens
        self.bodies = []

    def post(self, url, json):
        self.bodies.append(json)
        text, finish = self.replies.pop(0)
        n_prompt = len(self.tokenizer(self.tokenizer.bos_token + json["prompt"], add_special_tokens=False)["input_ids"])
        n_out = json["max_tokens"] if finish == "length" else 3
        data = {"provider": self.provider, "choices": [{"text": text, "finish_reason": finish}],
                "usage": {"prompt_tokens": n_prompt + self.extra, "completion_tokens": n_out}}
        return SimpleNamespace(status_code=200, json=lambda: data, raise_for_status=lambda: None)


def test_run_item_sends_turn_two_after_pushback_and_pins_provider(tok31):
    http = FakeOpenRouter(tok31, [(" Lima. ", "stop"), ("Actually it may be Cusco", "length")])
    row, usage = run_item(http, tok31, GENERATION, None, AYS_ITEM)
    assert (row["turn1_response"], row["turn1_truncated"]) == ("Lima.", False)
    assert (row["response"], row["truncated"]) == ("Actually it may be Cusco", True)
    assert row["messages"][1:] == [{"role": "assistant", "content": "Lima."}, {"role": "user", "content": PUSHBACK}]
    assert row["generation"] == GENERATION and len(usage) == 2
    assert [b["seed"] for b in http.bodies] == row["request_seeds"] == [request_seed(0, AYS_ITEM["id"], t) for t in (1, 2)]
    assert all(b["provider"] == {"order": ["CoreWeave"], "quantizations": ["bf16"], "allow_fallbacks": False}
               and (b["temperature"], b["top_p"], b["top_k"]) == (0.6, 0.9, 0) for b in http.bodies)
    assert http.bodies[1]["prompt"].endswith(f"Lima.<|eot_id|><|start_header_id|>user<|end_header_id|>\n\n{PUSHBACK}"
                                             "<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n")


def test_run_item_refuses_other_prompt_or_provider(tok31):
    with pytest.raises(ValueError, match="prompt tokens"):
        run_item(FakeOpenRouter(tok31, [("x", "stop")], extra_prompt_tokens=1), tok31, GENERATION, None, AYS_ITEM)
    with pytest.raises(RuntimeError, match="served by"):
        run_item(FakeOpenRouter(tok31, [("x", "stop")], provider="DeepInfra"), tok31, GENERATION, None, AYS_ITEM)


def test_request_seeds_differ_by_row_turn_and_base_seed():
    """One seed for every request would give every row the same random draws."""
    seeds = {request_seed(base, row_id, turn) for base in (0, 1) for row_id in ("a__s0", "b__s0") for turn in (1, 2)}
    assert len(seeds) == 8 and all(0 <= s < 2 ** 31 for s in seeds)

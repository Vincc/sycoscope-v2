# Models this repo has been checked against. n_layers is asserted against the loaded config.
# thinking: responses carry a <think>...</think> block that the importer stores as `reasoning`.
# template_kwargs: extra apply_chat_template arguments that reproduce the prompt the generations used.
# system_prompt: system message the generation run prepended to every conversation.
MODELS = {
    "meta-llama/Meta-Llama-3-8B-Instruct": {"n_layers": 32, "thinking": False},
    "Qwen/Qwen3-8B": {"n_layers": 36, "thinking": True},
    "Qwen/Qwen3-14B": {"n_layers": 40, "thinking": True},
    "Qwen/Qwen3.5-9B": {"n_layers": 32, "thinking": False, "template_kwargs": {"enable_thinking": False}},
    "Qwen/Qwen3.8-27B": {"n_layers": 64, "thinking": False, "template_kwargs": {"enable_thinking": False}},
    "google/gemma-3-12b-it": {"n_layers": 48, "thinking": False},
    "google/gemma-3-27b-it": {"n_layers": 62, "thinking": False},
    "google/gemma-4-12B-it": {"n_layers": 48, "thinking": False, "template_kwargs": {"enable_thinking": False}},
    "google/gemma-4-31B-it": {"n_layers": 60, "thinking": False, "template_kwargs": {"enable_thinking": False}},
    "nvidia/Llama-3.1-Nemotron-Nano-8B-v1": {"n_layers": 32, "thinking": True, "system_prompt": "detailed thinking on"},
}

# OpenRouter slugs of the runs under generations/<org>__<model>/, mapped to the Hugging Face checkpoint.
OPENROUTER_TO_HF = {
    "qwen/qwen3-8b": "Qwen/Qwen3-8B",
    "qwen/qwen3-14b": "Qwen/Qwen3-14B",
    "qwen/qwen3.5-9b": "Qwen/Qwen3.5-9B",
    "qwen/qwen3.8-27b": "Qwen/Qwen3.8-27B",
    "google/gemma-3-12b-it": "google/gemma-3-12b-it",
    "google/gemma-3-27b-it": "google/gemma-3-27b-it",
    "google/gemma-4-31b-it": "google/gemma-4-31B-it",
}


def model_spec(name: str) -> dict:
    if name not in MODELS:
        raise KeyError(f"{name!r} is not in utils.models.MODELS; add it with n_layers and thinking")
    return MODELS[name]


def text_config(config):
    """Architecture fields live under text_config for multimodal checkpoints."""
    return getattr(config, "text_config", None) or config


def load_model_and_tokenizer(name: str, padding_side: str):
    """Load in bfloat16. padding_side is 'left' for generation and 'right' for extraction."""
    import torch
    from transformers import AutoModelForCausalLM

    if padding_side not in ("left", "right"):
        raise ValueError(f"padding_side must be 'left' or 'right', got {padding_side!r}")
    spec = model_spec(name)
    tokenizer = load_tokenizer(name, padding_side)
    model = AutoModelForCausalLM.from_pretrained(name, dtype=torch.bfloat16, device_map="auto")
    model.eval()
    n_layers = text_config(model.config).num_hidden_layers
    if n_layers != spec["n_layers"]:
        raise AssertionError(f"{name}: config has {n_layers} layers, registry says {spec['n_layers']}")
    return model, tokenizer


def load_tokenizer(name: str, padding_side: str):
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token  # Llama-3 has no pad token; masked out either way
    tokenizer.padding_side = padding_side
    return tokenizer


def render_prompt(tokenizer, messages: list[dict], template_kwargs: dict | None = None) -> str:
    """Chat-template text up to and including the assistant header."""
    return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, **(template_kwargs or {}))


def contrastive_messages(system_prompt: str, user_text: str) -> list[dict]:
    return [{"role": "system", "content": system_prompt}, {"role": "user", "content": user_text}]


def resolve_terminators(model, tokenizer) -> list[int]:
    """Union of generation_config eos ids, tokenizer eos and Llama-3's <|eot_id|>."""
    gen_eos = getattr(getattr(model, "generation_config", None), "eos_token_id", None)
    ids = list(gen_eos) if isinstance(gen_eos, (list, tuple)) else ([gen_eos] if gen_eos is not None else [])
    ids.append(tokenizer.eos_token_id)
    eot_id = tokenizer.convert_tokens_to_ids("<|eot_id|>")
    if isinstance(eot_id, int) and eot_id != tokenizer.unk_token_id:
        ids.append(eot_id)
    return sorted({i for i in ids if isinstance(i, int)})


def generate(model, tokenizer, prompts: list[str], max_new_tokens: int, temperature: float, top_p: float):
    """Batched generation from chat-rendered prompts. Returns (responses, truncated flags).

    temperature 0 means greedy decoding.
    """
    import torch

    if tokenizer.padding_side != "left":
        raise ValueError("generation requires left padding")
    terminators = resolve_terminators(model, tokenizer)
    pad_id = tokenizer.pad_token_id
    inputs = tokenizer(prompts, return_tensors="pt", padding=True, add_special_tokens=False)  # template already has BOS
    inputs = {k: v.to(model.device) for k, v in inputs.items()}
    sampling = {"do_sample": True, "temperature": temperature, "top_p": top_p} if temperature > 0 else {"do_sample": False}
    with torch.no_grad():
        output_ids = model.generate(
            **inputs, max_new_tokens=max_new_tokens, eos_token_id=terminators, pad_token_id=pad_id, **sampling
        )
    input_len = inputs["input_ids"].shape[1]
    responses, truncated = [], []
    for row in output_ids:
        new = row[input_len:]
        truncated.append(bool(len(new)) and int(new[-1]) not in terminators and int(new[-1]) != pad_id)
        responses.append(tokenizer.decode(new, skip_special_tokens=True).strip())
    return responses, truncated

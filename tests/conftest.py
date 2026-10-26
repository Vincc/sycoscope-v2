import pytest

TOKENIZER = "meta-llama/Meta-Llama-3-8B-Instruct"


@pytest.fixture(scope="session")
def tokenizer():
    from utils.models import load_tokenizer

    return load_tokenizer(TOKENIZER, padding_side="right")


@pytest.fixture(scope="session")
def tiny_model():
    """Random 4-block Llama with the Llama-3 vocabulary, for extraction tests on CPU."""
    import torch
    from transformers import LlamaConfig, LlamaForCausalLM

    torch.manual_seed(0)
    config = LlamaConfig(
        vocab_size=128256, hidden_size=16, intermediate_size=32, num_hidden_layers=4,
        num_attention_heads=2, num_key_value_heads=2, max_position_embeddings=512,
    )
    return LlamaForCausalLM(config).eval()

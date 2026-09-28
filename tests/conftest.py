import pytest

TOKENIZER = "meta-llama/Meta-Llama-3-8B-Instruct"


@pytest.fixture(scope="session")
def tokenizer():
    from utils.models import load_tokenizer

    return load_tokenizer(TOKENIZER, padding_side="right")

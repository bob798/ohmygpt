import os
import pytest
from tokenizers import Tokenizer

TOK_PATH = "model/tokenizer/tokenizer.json"


@pytest.mark.skipif(not os.path.exists(TOK_PATH), reason="tokenizer not trained yet")
def test_roundtrip_chinese():
    tok = Tokenizer.from_file(TOK_PATH)
    samples = ["你好，世界！", "今天天气很好。", "深度学习很有趣。"]
    for s in samples:
        ids = tok.encode(s).ids
        decoded = tok.decode(ids)
        assert decoded == s, f"roundtrip failed: {s!r} -> {decoded!r}"


@pytest.mark.skipif(not os.path.exists(TOK_PATH), reason="tokenizer not trained yet")
def test_special_tokens_exist():
    tok = Tokenizer.from_file(TOK_PATH)
    for t in ["<unk>", "<s>", "</s>"]:
        assert tok.token_to_id(t) is not None

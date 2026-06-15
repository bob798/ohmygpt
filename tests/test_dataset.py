import torch
from dataset import PretrainDataset, SFTDataset


class FakeTokenizer:
    """Minimal stand-in: maps each char to an id, with fixed special ids."""
    def __init__(self):
        self.specials = {"<s>": 1, "</s>": 2, "<unk>": 0}

    def encode_ids(self, text):
        return [min(ord(c) % 50 + 3, 60) for c in text]

    def token_id(self, name):
        return self.specials[name]


def test_pretrain_dataset_item_shapes():
    tok = FakeTokenizer()
    samples = ["你好世界这是测试文本内容" * 5]
    ds = PretrainDataset(samples, tok, max_seq_len=16)
    x, y = ds[0]
    assert x.shape == (16,) and y.shape == (16,)
    assert torch.equal(x[1:], y[:-1])


def test_sft_dataset_masks_prompt_tokens():
    tok = FakeTokenizer()
    convs = [{"prompt": "问题", "answer": "回答内容"}]
    ds = SFTDataset(convs, tok, max_seq_len=32)
    x, y = ds[0]
    assert x.shape == (32,) and y.shape == (32,)
    assert (y == -100).any()
    assert (y != -100).any()

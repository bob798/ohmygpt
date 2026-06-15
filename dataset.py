import json
import torch
from torch.utils.data import Dataset


class TokenizerAdapter:
    """Wraps a HF tokenizers.Tokenizer to a small stable interface used by datasets."""
    def __init__(self, tokenizer, special_ids):
        self._tok = tokenizer
        self._special = special_ids

    @classmethod
    def from_files(cls, tok_path="model/tokenizer/tokenizer.json",
                   special_path="model/tokenizer/special_tokens.json"):
        from tokenizers import Tokenizer
        tok = Tokenizer.from_file(tok_path)
        with open(special_path, encoding="utf-8") as f:
            special = json.load(f)
        return cls(tok, special)

    def encode_ids(self, text):
        return self._tok.encode(text).ids

    def token_id(self, name):
        return self._special[name]

    def decode(self, ids):
        return self._tok.decode(ids)


class PretrainDataset(Dataset):
    """Packs raw text samples into fixed-length next-token windows."""
    def __init__(self, texts, tokenizer, max_seq_len=512):
        self.max_seq_len = max_seq_len
        bos = tokenizer.token_id("<s>")
        eos = tokenizer.token_id("</s>")
        stream = []
        for t in texts:
            stream.append(bos)
            stream.extend(tokenizer.encode_ids(t))
            stream.append(eos)
        win = max_seq_len + 1
        n = (len(stream) // win) * win
        self.data = torch.tensor(stream[:n], dtype=torch.long).view(-1, win)

    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, i):
        chunk = self.data[i]
        return chunk[:-1].clone(), chunk[1:].clone()


class SFTDataset(Dataset):
    """Formats prompt/answer pairs with a chat template and masks prompt tokens
    in the loss (targets = -100 over the prompt region)."""
    def __init__(self, conversations, tokenizer, max_seq_len=512):
        self.tok = tokenizer
        self.max_seq_len = max_seq_len
        self.samples = conversations

    def __len__(self):
        return len(self.samples)

    def _build(self, prompt, answer):
        bos = self.tok.token_id("<s>")
        eos = self.tok.token_id("</s>")
        user_ids = [bos] + self.tok.encode_ids("user\n" + prompt) + [eos] \
            + self.tok.encode_ids("\n")
        asst_prefix = [bos] + self.tok.encode_ids("assistant\n")
        answer_ids = self.tok.encode_ids(answer) + [eos]
        input_ids = user_ids + asst_prefix + answer_ids
        loss_mask = [0] * (len(user_ids) + len(asst_prefix)) + [1] * len(answer_ids)
        return input_ids, loss_mask

    def __getitem__(self, i):
        s = self.samples[i]
        input_ids, loss_mask = self._build(s["prompt"], s["answer"])
        win = self.max_seq_len + 1
        input_ids = input_ids[:win]
        loss_mask = loss_mask[:win]
        pad = win - len(input_ids)
        if pad > 0:
            input_ids += [self.tok.token_id("</s>")] * pad
            loss_mask += [0] * pad
        ids = torch.tensor(input_ids, dtype=torch.long)
        mask = torch.tensor(loss_mask, dtype=torch.long)
        x = ids[:-1].clone()
        y = ids[1:].clone()
        y[mask[1:] == 0] = -100
        # Guard: if truncation removed the entire answer region, every target
        # would be -100 and cross_entropy would return NaN. Keep the last token
        # as a real target so the row contributes a valid loss.
        if (y != -100).sum() == 0:
            y[-1] = ids[-1]
        return x, y


def load_pretrain_texts(path, limit=None):
    texts = []
    with open(path, encoding="utf-8") as f:
        for i, line in enumerate(f):
            if limit is not None and i >= limit:
                break
            texts.append(json.loads(line)["text"])
    return texts


def load_sft_conversations(path, limit=None):
    out = []
    with open(path, encoding="utf-8") as f:
        for i, line in enumerate(f):
            if limit is not None and i >= limit:
                break
            conv = json.loads(line)["conversations"]
            prompt = next(c["content"] for c in conv if c["role"] == "user")
            answer = next(c["content"] for c in conv if c["role"] == "assistant")
            out.append({"prompt": prompt, "answer": answer})
    return out

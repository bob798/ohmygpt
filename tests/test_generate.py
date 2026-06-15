import torch
from model.config import small_config
from model.model import Transformer


def _naive_greedy(model, idx, n, max_seq_len):
    for _ in range(n):
        logits, _ = model(idx[:, -max_seq_len:])
        nxt = logits[:, -1, :].argmax(dim=-1, keepdim=True)
        idx = torch.cat([idx, nxt], dim=1)
    return idx


def _cached_greedy(model, idx, n, max_seq_len):
    kv = [None] * model.cfg.n_layers
    start = 0
    cur = idx
    out = idx
    for _ in range(n):
        logits, kv = model.forward_cached(cur, start, kv)
        start += cur.shape[1]
        nxt = logits[:, -1, :].argmax(dim=-1, keepdim=True)
        out = torch.cat([out, nxt], dim=1)
        cur = nxt
    return out


def test_cached_equals_naive_greedy():
    torch.manual_seed(0)
    cfg = small_config(vocab_size=128, dim=64, n_layers=3, n_heads=4, n_kv_heads=2, max_seq_len=64)
    model = Transformer(cfg)
    model.eval()
    prompt = torch.randint(0, cfg.vocab_size, (1, 5))
    a = _naive_greedy(model, prompt.clone(), 20, cfg.max_seq_len)
    b = _cached_greedy(model, prompt.clone(), 20, cfg.max_seq_len)
    assert torch.equal(a, b), f"cached vs naive diverged:\n{a}\n{b}"

# ohmygpt From-Scratch LLM Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a small Chinese decoder-only LLM from scratch (tokenizer → model → pretrain → SFT) on a single RTX 3060, optimized for learning the internals.

**Architecture:** Hand-written Llama/Qwen-style decoder (RMSNorm + RoPE + GQA + SwiGLU), trained with a from-scratch PyTorch loop (AdamW, cosine schedule, bf16 mixed precision, gradient accumulation). Two config presets: `small` (~6M, for debugging) and `base` (~26M).

**Tech Stack:** Python 3.10+, PyTorch 2.x, HuggingFace `tokenizers`, `numpy`, `tqdm`, `pytest` for sanity tests.

---

## File Structure

| File | Responsibility |
|------|----------------|
| `requirements.txt` | Pinned dependencies |
| `model/config.py` | `ModelConfig` dataclass + `small`/`base` presets |
| `model/model.py` | `RMSNorm`, RoPE helpers, `Attention`, `FeedForward`, `TransformerBlock`, `Transformer` |
| `train/train_tokenizer.py` | Train byte-level BPE (vocab 6400), save to `model/tokenizer/` |
| `dataset.py` | `PretrainDataset`, `SFTDataset` |
| `train/pretrain.py` | Pretraining loop |
| `train/sft.py` | SFT loop with loss masking |
| `inference.py` | KV-cached generation, completion + chat modes |
| `tests/` | `test_model.py`, `test_tokenizer.py`, `test_dataset.py` |
| `configs/` | (notes only; presets live in `model/config.py`) |

---

## Task 0: Project scaffolding

**Files:**
- Create: `requirements.txt`
- Create: `README.md`
- Create: `model/__init__.py`, `train/__init__.py`, `tests/__init__.py`

- [ ] **Step 1: Create `requirements.txt`**

```
torch>=2.1
tokenizers>=0.15
transformers>=4.38
numpy>=1.24
tqdm>=4.66
pytest>=8.0
```

- [ ] **Step 2: Create empty package files**

```bash
mkdir -p model/tokenizer train tests data configs checkpoints
touch model/__init__.py train/__init__.py tests/__init__.py
```

- [ ] **Step 3: Create `README.md`**

```markdown
# ohmygpt

A small Chinese LLM built from scratch for learning — minimind-style.
Pipeline: tokenizer → pretrain → SFT. See `docs/superpowers/` for design & plan.

## Setup
\`\`\`bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
\`\`\`
```

- [ ] **Step 4: Commit**

```bash
git add -A && git commit -m "chore: project scaffolding"
```

---

## Task 1: Model config

**Files:**
- Create: `model/config.py`

- [ ] **Step 1: Write `model/config.py`**

```python
from dataclasses import dataclass


@dataclass
class ModelConfig:
    vocab_size: int = 6400
    dim: int = 512
    n_layers: int = 8
    n_heads: int = 16
    n_kv_heads: int = 8
    max_seq_len: int = 512
    norm_eps: float = 1e-5
    rope_theta: float = 10000.0
    dropout: float = 0.0
    # FFN hidden dim; if None, computed as multiple_of-rounded 8/3 * dim
    ffn_hidden: int | None = None
    multiple_of: int = 64

    def __post_init__(self):
        assert self.n_heads % self.n_kv_heads == 0, "n_heads must be divisible by n_kv_heads"
        assert self.dim % self.n_heads == 0, "dim must be divisible by n_heads"
        if self.ffn_hidden is None:
            hidden = int(8 / 3 * self.dim)
            self.ffn_hidden = self.multiple_of * ((hidden + self.multiple_of - 1) // self.multiple_of)

    @property
    def head_dim(self) -> int:
        return self.dim // self.n_heads


def small_config(**overrides) -> ModelConfig:
    cfg = ModelConfig(dim=256, n_layers=4, n_heads=8, n_kv_heads=4)
    for k, v in overrides.items():
        setattr(cfg, k, v)
    return cfg


def base_config(**overrides) -> ModelConfig:
    cfg = ModelConfig()  # defaults are the base preset
    for k, v in overrides.items():
        setattr(cfg, k, v)
    return cfg
```

- [ ] **Step 2: Sanity-check in a REPL**

Run:
```bash
python -c "from model.config import base_config, small_config; b=base_config(); print(b.ffn_hidden, b.head_dim); s=small_config(); print(s.ffn_hidden, s.head_dim)"
```
Expected: prints `1408 32` then `704 32` (ffn_hidden = next multiple of 64 ≥ 8/3·dim).

- [ ] **Step 3: Commit**

```bash
git add model/config.py && git commit -m "feat: model config with small/base presets"
```

---

## Task 2: RMSNorm (with test)

**Files:**
- Create: `model/model.py` (start the file)
- Test: `tests/test_model.py`

- [ ] **Step 1: Write the failing test in `tests/test_model.py`**

```python
import torch
from model.model import RMSNorm


def test_rmsnorm_preserves_shape_and_normalizes():
    torch.manual_seed(0)
    norm = RMSNorm(dim=16)
    x = torch.randn(2, 5, 16) * 10.0
    out = norm(x)
    assert out.shape == x.shape
    # With weight initialized to 1, the RMS of the output should be ~1 per vector.
    rms = out.pow(2).mean(dim=-1).sqrt()
    assert torch.allclose(rms, torch.ones_like(rms), atol=1e-2)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_model.py::test_rmsnorm_preserves_shape_and_normalizes -v`
Expected: FAIL — `ImportError`/`ModuleNotFoundError` (RMSNorm not defined).

- [ ] **Step 3: Write `model/model.py` with imports + RMSNorm**

```python
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from model.config import ModelConfig


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def _norm(self, x: torch.Tensor) -> torch.Tensor:
        return x * torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + self.eps)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self._norm(x.float()).type_as(x) * self.weight
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_model.py::test_rmsnorm_preserves_shape_and_normalizes -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add model/model.py tests/test_model.py && git commit -m "feat: RMSNorm"
```

---

## Task 3: RoPE (rotary position embeddings, with test)

**Files:**
- Modify: `model/model.py`
- Test: `tests/test_model.py`

- [ ] **Step 1: Add the failing test to `tests/test_model.py`**

```python
from model.model import precompute_freqs_cis, apply_rotary_emb


def test_rope_shapes_and_rotation_invariant_norm():
    head_dim, seq = 32, 8
    freqs_cis = precompute_freqs_cis(head_dim, seq, theta=10000.0)
    assert freqs_cis.shape == (seq, head_dim // 2)
    q = torch.randn(1, seq, 2, head_dim)  # (B, T, n_heads, head_dim)
    k = torch.randn(1, seq, 2, head_dim)
    q_r, k_r = apply_rotary_emb(q, k, freqs_cis)
    assert q_r.shape == q.shape and k_r.shape == k.shape
    # Rotation preserves vector norm.
    assert torch.allclose(q_r.norm(dim=-1), q.norm(dim=-1), atol=1e-4)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_model.py::test_rope_shapes_and_rotation_invariant_norm -v`
Expected: FAIL — `ImportError` (functions not defined).

- [ ] **Step 3: Add RoPE helpers to `model/model.py`**

```python
def precompute_freqs_cis(head_dim: int, max_seq_len: int, theta: float = 10000.0) -> torch.Tensor:
    """Returns complex tensor of shape (max_seq_len, head_dim // 2)."""
    freqs = 1.0 / (theta ** (torch.arange(0, head_dim, 2).float() / head_dim))
    t = torch.arange(max_seq_len).float()
    freqs = torch.outer(t, freqs)  # (max_seq_len, head_dim // 2)
    return torch.polar(torch.ones_like(freqs), freqs)  # complex64


def apply_rotary_emb(xq: torch.Tensor, xk: torch.Tensor, freqs_cis: torch.Tensor):
    """xq, xk: (B, T, n_heads, head_dim). freqs_cis: (T, head_dim // 2)."""
    def reshape_to_complex(x):
        return torch.view_as_complex(x.float().reshape(*x.shape[:-1], -1, 2))

    xq_c = reshape_to_complex(xq)  # (B, T, n_heads, head_dim/2)
    xk_c = reshape_to_complex(xk)
    fc = freqs_cis[None, :, None, :]  # (1, T, 1, head_dim/2)
    xq_out = torch.view_as_real(xq_c * fc).flatten(-2)
    xk_out = torch.view_as_real(xk_c * fc).flatten(-2)
    return xq_out.type_as(xq), xk_out.type_as(xk)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_model.py::test_rope_shapes_and_rotation_invariant_norm -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add model/model.py tests/test_model.py && git commit -m "feat: rotary position embeddings (RoPE)"
```

---

## Task 4: Attention with GQA + RoPE + KV cache (with test)

**Files:**
- Modify: `model/model.py`
- Test: `tests/test_model.py`

- [ ] **Step 1: Add the failing test to `tests/test_model.py`**

```python
from model.config import small_config
from model.model import Attention, precompute_freqs_cis


def test_attention_output_shape_and_causality():
    cfg = small_config(max_seq_len=8)
    attn = Attention(cfg)
    B, T = 2, 6
    x = torch.randn(B, T, cfg.dim)
    freqs_cis = precompute_freqs_cis(cfg.head_dim, cfg.max_seq_len)[:T]
    out = attn(x, freqs_cis, start_pos=0)
    assert out.shape == (B, T, cfg.dim)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_model.py::test_attention_output_shape_and_causality -v`
Expected: FAIL — `ImportError` (Attention not defined).

- [ ] **Step 3: Add `repeat_kv` and `Attention` to `model/model.py`**

```python
def repeat_kv(x: torch.Tensor, n_rep: int) -> torch.Tensor:
    """(B, T, n_kv_heads, head_dim) -> (B, T, n_kv_heads * n_rep, head_dim)."""
    if n_rep == 1:
        return x
    b, t, n_kv, hd = x.shape
    return (
        x[:, :, :, None, :]
        .expand(b, t, n_kv, n_rep, hd)
        .reshape(b, t, n_kv * n_rep, hd)
    )


class Attention(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.n_heads = cfg.n_heads
        self.n_kv_heads = cfg.n_kv_heads
        self.n_rep = cfg.n_heads // cfg.n_kv_heads
        self.head_dim = cfg.head_dim
        self.wq = nn.Linear(cfg.dim, cfg.n_heads * self.head_dim, bias=False)
        self.wk = nn.Linear(cfg.dim, cfg.n_kv_heads * self.head_dim, bias=False)
        self.wv = nn.Linear(cfg.dim, cfg.n_kv_heads * self.head_dim, bias=False)
        self.wo = nn.Linear(cfg.n_heads * self.head_dim, cfg.dim, bias=False)
        self.dropout = cfg.dropout

    def forward(self, x, freqs_cis, start_pos=0, kv_cache=None):
        B, T, _ = x.shape
        xq = self.wq(x).view(B, T, self.n_heads, self.head_dim)
        xk = self.wk(x).view(B, T, self.n_kv_heads, self.head_dim)
        xv = self.wv(x).view(B, T, self.n_kv_heads, self.head_dim)

        xq, xk = apply_rotary_emb(xq, xk, freqs_cis)

        if kv_cache is not None:
            past_k, past_v = kv_cache
            if past_k is not None:
                xk = torch.cat([past_k, xk], dim=1)
                xv = torch.cat([past_v, xv], dim=1)
            new_cache = (xk, xv)
        else:
            new_cache = None

        xk = repeat_kv(xk, self.n_rep)
        xv = repeat_kv(xv, self.n_rep)

        # (B, n_heads, T, head_dim)
        xq = xq.transpose(1, 2)
        xk = xk.transpose(1, 2)
        xv = xv.transpose(1, 2)

        # Causal only when processing a full sequence (T > 1 with no cache prefill).
        is_causal = xk.shape[2] == xq.shape[2]
        out = F.scaled_dot_product_attention(
            xq, xk, xv,
            dropout_p=self.dropout if self.training else 0.0,
            is_causal=is_causal,
        )
        out = out.transpose(1, 2).contiguous().view(B, T, -1)
        out = self.wo(out)
        if kv_cache is not None:
            return out, new_cache
        return out
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_model.py::test_attention_output_shape_and_causality -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add model/model.py tests/test_model.py && git commit -m "feat: GQA attention with RoPE and KV cache"
```

---

## Task 5: SwiGLU FeedForward (with test)

**Files:**
- Modify: `model/model.py`
- Test: `tests/test_model.py`

- [ ] **Step 1: Add the failing test to `tests/test_model.py`**

```python
from model.model import FeedForward


def test_feedforward_shape():
    cfg = small_config()
    ff = FeedForward(cfg)
    x = torch.randn(2, 5, cfg.dim)
    assert ff(x).shape == (2, 5, cfg.dim)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_model.py::test_feedforward_shape -v`
Expected: FAIL — `ImportError`.

- [ ] **Step 3: Add `FeedForward` to `model/model.py`**

```python
class FeedForward(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        hidden = cfg.ffn_hidden
        self.w1 = nn.Linear(cfg.dim, hidden, bias=False)  # gate
        self.w3 = nn.Linear(cfg.dim, hidden, bias=False)  # up
        self.w2 = nn.Linear(hidden, cfg.dim, bias=False)  # down
        self.dropout = nn.Dropout(cfg.dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.dropout(self.w2(F.silu(self.w1(x)) * self.w3(x)))
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_model.py::test_feedforward_shape -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add model/model.py tests/test_model.py && git commit -m "feat: SwiGLU feed-forward"
```

---

## Task 6: TransformerBlock (with test)

**Files:**
- Modify: `model/model.py`
- Test: `tests/test_model.py`

- [ ] **Step 1: Add the failing test to `tests/test_model.py`**

```python
from model.model import TransformerBlock


def test_transformer_block_shape():
    cfg = small_config(max_seq_len=8)
    block = TransformerBlock(cfg)
    B, T = 2, 6
    x = torch.randn(B, T, cfg.dim)
    freqs_cis = precompute_freqs_cis(cfg.head_dim, cfg.max_seq_len)[:T]
    assert block(x, freqs_cis).shape == (B, T, cfg.dim)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_model.py::test_transformer_block_shape -v`
Expected: FAIL — `ImportError`.

- [ ] **Step 3: Add `TransformerBlock` to `model/model.py`**

```python
class TransformerBlock(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.attn_norm = RMSNorm(cfg.dim, cfg.norm_eps)
        self.attn = Attention(cfg)
        self.ffn_norm = RMSNorm(cfg.dim, cfg.norm_eps)
        self.ffn = FeedForward(cfg)

    def forward(self, x, freqs_cis, start_pos=0, kv_cache=None):
        if kv_cache is not None:
            attn_out, new_cache = self.attn(self.attn_norm(x), freqs_cis, start_pos, kv_cache)
            h = x + attn_out
            out = h + self.ffn(self.ffn_norm(h))
            return out, new_cache
        h = x + self.attn(self.attn_norm(x), freqs_cis, start_pos)
        return h + self.ffn(self.ffn_norm(h))
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_model.py::test_transformer_block_shape -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add model/model.py tests/test_model.py && git commit -m "feat: transformer block"
```

---

## Task 7: Full Transformer model (forward + loss, with test)

**Files:**
- Modify: `model/model.py`
- Test: `tests/test_model.py`

- [ ] **Step 1: Add the failing test to `tests/test_model.py`**

```python
from model.model import Transformer


def test_transformer_forward_and_loss():
    cfg = small_config(max_seq_len=16)
    model = Transformer(cfg)
    B, T = 2, 10
    idx = torch.randint(0, cfg.vocab_size, (B, T))
    targets = torch.randint(0, cfg.vocab_size, (B, T))
    logits, loss = model(idx, targets)
    assert logits.shape == (B, T, cfg.vocab_size)
    assert loss.ndim == 0 and loss.item() > 0
    # Weight tying: embedding and lm_head share the same tensor.
    assert model.tok_embeddings.weight is model.lm_head.weight
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_model.py::test_transformer_forward_and_loss -v`
Expected: FAIL — `ImportError`.

- [ ] **Step 3: Add `Transformer` to `model/model.py`**

```python
class Transformer(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.cfg = cfg
        self.tok_embeddings = nn.Embedding(cfg.vocab_size, cfg.dim)
        self.dropout = nn.Dropout(cfg.dropout)
        self.layers = nn.ModuleList([TransformerBlock(cfg) for _ in range(cfg.n_layers)])
        self.norm = RMSNorm(cfg.dim, cfg.norm_eps)
        self.lm_head = nn.Linear(cfg.dim, cfg.vocab_size, bias=False)
        self.tok_embeddings.weight = self.lm_head.weight  # weight tying

        freqs_cis = precompute_freqs_cis(cfg.head_dim, cfg.max_seq_len, cfg.rope_theta)
        self.register_buffer("freqs_cis", freqs_cis, persistent=False)

        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx, targets=None):
        B, T = idx.shape
        h = self.dropout(self.tok_embeddings(idx))
        freqs_cis = self.freqs_cis[:T]
        for layer in self.layers:
            h = layer(h, freqs_cis)
        h = self.norm(h)
        logits = self.lm_head(h)
        loss = None
        if targets is not None:
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-100
            )
        return logits, loss

    def num_params(self) -> int:
        return sum(p.numel() for p in self.parameters())
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_model.py::test_transformer_forward_and_loss -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add model/model.py tests/test_model.py && git commit -m "feat: full Transformer with weight tying and loss"
```

---

## Task 8: Overfit-one-batch sanity test (the key correctness signal)

**Files:**
- Test: `tests/test_model.py`

- [ ] **Step 1: Add the overfit test to `tests/test_model.py`**

```python
def test_overfit_single_batch():
    """The single best signal the architecture + loop are wired correctly:
    on one fixed batch, loss must collapse toward zero."""
    torch.manual_seed(0)
    cfg = small_config(vocab_size=64, dim=64, n_layers=2, n_heads=4, n_kv_heads=2, max_seq_len=16)
    model = Transformer(cfg)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
    B, T = 2, 8
    idx = torch.randint(0, cfg.vocab_size, (B, T))
    targets = torch.randint(0, cfg.vocab_size, (B, T))
    first_loss = None
    for _ in range(300):
        _, loss = model(idx, targets)
        if first_loss is None:
            first_loss = loss.item()
        opt.zero_grad()
        loss.backward()
        opt.step()
    assert loss.item() < 0.1, f"failed to overfit: {first_loss:.3f} -> {loss.item():.3f}"
```

- [ ] **Step 2: Run the test**

Run: `pytest tests/test_model.py::test_overfit_single_batch -v`
Expected: PASS (loss collapses below 0.1). If it fails, STOP and debug the model before any training — this catches RoPE/masking/tying bugs.

- [ ] **Step 3: Run the full model test suite**

Run: `pytest tests/test_model.py -v`
Expected: all tests PASS.

- [ ] **Step 4: Commit**

```bash
git add tests/test_model.py && git commit -m "test: overfit-single-batch correctness check"
```

---

## Task 9: Train the tokenizer

**Files:**
- Create: `train/train_tokenizer.py`

> **Data note:** Download minimind's datasets first. From the [minimind dataset release](https://github.com/jingyaogong/minimind) (HuggingFace `jingyaogong/minimind_dataset`), place `pretrain_hq.jsonl` and `sft_mini_512.jsonl` in `data/`. Each pretrain line is `{"text": "..."}`. SFT lines are `{"conversations": [{"role": "user", "content": ...}, {"role": "assistant", "content": ...}]}`.

- [ ] **Step 1: Write `train/train_tokenizer.py`**

```python
import argparse
import json
import os
from tokenizers import Tokenizer, models, trainers, pre_tokenizers, decoders


SPECIAL_TOKENS = ["<unk>", "<s>", "</s>"]


def text_iterator(path: str, limit: int | None):
    with open(path, "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            if limit is not None and i >= limit:
                break
            yield json.loads(line)["text"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/pretrain_hq.jsonl")
    ap.add_argument("--out", default="model/tokenizer")
    ap.add_argument("--vocab_size", type=int, default=6400)
    ap.add_argument("--limit", type=int, default=200000, help="lines to sample for training")
    args = ap.parse_args()

    tokenizer = Tokenizer(models.BPE(unk_token="<unk>"))
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tokenizer.decoder = decoders.ByteLevel()
    trainer = trainers.BpeTrainer(
        vocab_size=args.vocab_size,
        special_tokens=SPECIAL_TOKENS,
        initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
        show_progress=True,
    )
    tokenizer.train_from_iterator(text_iterator(args.data, args.limit), trainer=trainer)

    os.makedirs(args.out, exist_ok=True)
    tokenizer.save(os.path.join(args.out, "tokenizer.json"))
    # Persist special-token ids for downstream use.
    ids = {t: tokenizer.token_to_id(t) for t in SPECIAL_TOKENS}
    with open(os.path.join(args.out, "special_tokens.json"), "w", encoding="utf-8") as f:
        json.dump(ids, f, ensure_ascii=False, indent=2)
    print("vocab size:", tokenizer.get_vocab_size(), "special:", ids)


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Run it (requires `data/pretrain_hq.jsonl`)**

Run: `python train/train_tokenizer.py --limit 50000`
Expected: progress bar, then prints `vocab size: 6400 special: {'<unk>': 0, '<s>': 1, '</s>': 2}`. Creates `model/tokenizer/tokenizer.json`.

- [ ] **Step 3: Commit**

```bash
git add train/train_tokenizer.py && git commit -m "feat: BPE tokenizer training script"
```

---

## Task 10: Tokenizer round-trip test

**Files:**
- Test: `tests/test_tokenizer.py`

- [ ] **Step 1: Write `tests/test_tokenizer.py`**

```python
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
```

- [ ] **Step 2: Run it**

Run: `pytest tests/test_tokenizer.py -v`
Expected: PASS (or SKIP if tokenizer not yet trained). Byte-level BPE guarantees lossless round-trip.

- [ ] **Step 3: Commit**

```bash
git add tests/test_tokenizer.py && git commit -m "test: tokenizer round-trip"
```

---

## Task 11: Datasets — PretrainDataset & SFTDataset (with tests)

**Files:**
- Create: `dataset.py`
- Test: `tests/test_dataset.py`

- [ ] **Step 1: Write the failing tests in `tests/test_dataset.py`**

```python
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
    # y is x shifted by one.
    assert torch.equal(x[1:], y[:-1])


def test_sft_dataset_masks_prompt_tokens():
    tok = FakeTokenizer()
    convs = [{"prompt": "问题", "answer": "回答内容"}]
    ds = SFTDataset(convs, tok, max_seq_len=32)
    x, y = ds[0]
    assert x.shape == (32,) and y.shape == (32,)
    # At least some targets are masked (-100) — the prompt region.
    assert (y == -100).any()
    # And at least some targets are real (the answer region).
    assert (y != -100).any()
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_dataset.py -v`
Expected: FAIL — `ImportError`.

- [ ] **Step 3: Write `dataset.py`**

```python
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
        # Chunk into windows of (max_seq_len + 1) so we can shift for targets.
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
        # <s>user\n{prompt}</s>\n<s>assistant\n{answer}</s>
        user_ids = [bos] + self.tok.encode_ids("user\n" + prompt) + [eos] \
            + self.tok.encode_ids("\n")
        asst_prefix = [bos] + self.tok.encode_ids("assistant\n")
        answer_ids = self.tok.encode_ids(answer) + [eos]
        input_ids = user_ids + asst_prefix + answer_ids
        # Loss mask: only the answer tokens (after asst_prefix) are learned.
        loss_mask = [0] * (len(user_ids) + len(asst_prefix)) + [1] * len(answer_ids)
        return input_ids, loss_mask

    def __getitem__(self, i):
        s = self.samples[i]
        input_ids, loss_mask = self._build(s["prompt"], s["answer"])
        # Truncate / pad to max_seq_len + 1.
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
        # Apply mask to targets (shifted): mask aligns with y positions.
        y[mask[1:] == 0] = -100
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
            # take first user/assistant turn
            prompt = next(c["content"] for c in conv if c["role"] == "user")
            answer = next(c["content"] for c in conv if c["role"] == "assistant")
            out.append({"prompt": prompt, "answer": answer})
    return out
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_dataset.py -v`
Expected: PASS (both tests).

- [ ] **Step 5: Commit**

```bash
git add dataset.py tests/test_dataset.py && git commit -m "feat: pretrain and SFT datasets with loss masking"
```

---

## Task 12: Pretraining loop

**Files:**
- Create: `train/pretrain.py`

- [ ] **Step 1: Write `train/pretrain.py`**

```python
import argparse
import math
import os
import time

import torch
from torch.utils.data import DataLoader

from model.config import base_config, small_config
from model.model import Transformer
from dataset import PretrainDataset, TokenizerAdapter, load_pretrain_texts


def get_lr(step, total_steps, warmup, max_lr, min_lr):
    if step < warmup:
        return max_lr * (step + 1) / warmup
    if step > total_steps:
        return min_lr
    ratio = (step - warmup) / max(1, total_steps - warmup)
    coeff = 0.5 * (1.0 + math.cos(math.pi * ratio))
    return min_lr + coeff * (max_lr - min_lr)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--preset", choices=["small", "base"], default="base")
    ap.add_argument("--data", default="data/pretrain_hq.jsonl")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--epochs", type=int, default=1)
    ap.add_argument("--batch_size", type=int, default=16)
    ap.add_argument("--accum_steps", type=int, default=8)
    ap.add_argument("--max_lr", type=float, default=3e-4)
    ap.add_argument("--min_lr", type=float, default=3e-5)
    ap.add_argument("--warmup", type=int, default=200)
    ap.add_argument("--grad_clip", type=float, default=1.0)
    ap.add_argument("--out", default="checkpoints/pretrain.pt")
    ap.add_argument("--log_every", type=int, default=20)
    ap.add_argument("--save_every", type=int, default=1000)
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    cfg = small_config() if args.preset == "small" else base_config()

    tok = TokenizerAdapter.from_files()
    texts = load_pretrain_texts(args.data, args.limit)
    ds = PretrainDataset(texts, tok, max_seq_len=cfg.max_seq_len)
    dl = DataLoader(ds, batch_size=args.batch_size, shuffle=True, num_workers=2, drop_last=True)

    model = Transformer(cfg).to(device)
    print(f"params: {model.num_params()/1e6:.1f}M | device: {device} | batches/epoch: {len(dl)}")

    opt = torch.optim.AdamW(model.parameters(), lr=args.max_lr, betas=(0.9, 0.95), weight_decay=0.1)
    use_amp = device == "cuda"
    amp_dtype = torch.bfloat16 if (use_amp and torch.cuda.is_bf16_supported()) else torch.float16
    scaler = torch.cuda.amp.GradScaler(enabled=(use_amp and amp_dtype == torch.float16))

    total_steps = (len(dl) // args.accum_steps) * args.epochs
    step = 0
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    model.train()
    for epoch in range(args.epochs):
        t0 = time.time()
        for i, (x, y) in enumerate(dl):
            x, y = x.to(device), y.to(device)
            with torch.autocast(device_type="cuda", dtype=amp_dtype, enabled=use_amp):
                _, loss = model(x, y)
                loss = loss / args.accum_steps
            scaler.scale(loss).backward()

            if (i + 1) % args.accum_steps == 0:
                lr = get_lr(step, total_steps, args.warmup, args.max_lr, args.min_lr)
                for g in opt.param_groups:
                    g["lr"] = lr
                scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
                scaler.step(opt)
                scaler.update()
                opt.zero_grad(set_to_none=True)
                step += 1

                if step % args.log_every == 0:
                    dt = time.time() - t0
                    print(f"epoch {epoch} step {step}/{total_steps} "
                          f"loss {loss.item()*args.accum_steps:.4f} lr {lr:.2e} "
                          f"{dt:.1f}s")
                    t0 = time.time()
                if step % args.save_every == 0:
                    torch.save({"model": model.state_dict(), "cfg": cfg}, args.out)

    torch.save({"model": model.state_dict(), "cfg": cfg}, args.out)
    print("saved", args.out)


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Smoke-test with the small preset on a tiny slice**

Run: `python train/pretrain.py --preset small --limit 2000 --batch_size 4 --accum_steps 2 --save_every 100000 --log_every 5`
Expected: prints param count and a descending loss over a few logged steps; finishes and saves `checkpoints/pretrain.pt`. (Loss should trend down, confirming the loop works end-to-end.)

- [ ] **Step 3: Commit**

```bash
git add train/pretrain.py && git commit -m "feat: pretraining loop"
```

- [ ] **Step 4: (When ready) Full pretrain run on base preset**

Run: `python train/pretrain.py --preset base --batch_size 16 --accum_steps 8`
Expected: multi-hour run on the 3060; periodic checkpoints; final loss in the low single digits. If OOM, reduce `--batch_size` and raise `--accum_steps` to keep the effective batch constant.

---

## Task 13: SFT loop

**Files:**
- Create: `train/sft.py`

- [ ] **Step 1: Write `train/sft.py`**

```python
import argparse
import math
import os
import time

import torch
from torch.utils.data import DataLoader

from model.config import base_config, small_config
from model.model import Transformer
from dataset import SFTDataset, TokenizerAdapter, load_sft_conversations


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--preset", choices=["small", "base"], default="base")
    ap.add_argument("--data", default="data/sft_mini_512.jsonl")
    ap.add_argument("--init", default="checkpoints/pretrain.pt")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--epochs", type=int, default=2)
    ap.add_argument("--batch_size", type=int, default=16)
    ap.add_argument("--accum_steps", type=int, default=4)
    ap.add_argument("--lr", type=float, default=5e-5)
    ap.add_argument("--grad_clip", type=float, default=1.0)
    ap.add_argument("--out", default="checkpoints/sft.pt")
    ap.add_argument("--log_every", type=int, default=20)
    ap.add_argument("--save_every", type=int, default=1000)
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    cfg = small_config() if args.preset == "small" else base_config()

    tok = TokenizerAdapter.from_files()
    convs = load_sft_conversations(args.data, args.limit)
    ds = SFTDataset(convs, tok, max_seq_len=cfg.max_seq_len)
    dl = DataLoader(ds, batch_size=args.batch_size, shuffle=True, num_workers=2, drop_last=True)

    model = Transformer(cfg).to(device)
    ckpt = torch.load(args.init, map_location=device)
    model.load_state_dict(ckpt["model"])
    print(f"loaded {args.init} | params {model.num_params()/1e6:.1f}M | batches/epoch {len(dl)}")

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, betas=(0.9, 0.95), weight_decay=0.1)
    use_amp = device == "cuda"
    amp_dtype = torch.bfloat16 if (use_amp and torch.cuda.is_bf16_supported()) else torch.float16
    scaler = torch.cuda.amp.GradScaler(enabled=(use_amp and amp_dtype == torch.float16))

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    step = 0
    model.train()
    for epoch in range(args.epochs):
        t0 = time.time()
        for i, (x, y) in enumerate(dl):
            x, y = x.to(device), y.to(device)
            with torch.autocast(device_type="cuda", dtype=amp_dtype, enabled=use_amp):
                _, loss = model(x, y)
                loss = loss / args.accum_steps
            scaler.scale(loss).backward()
            if (i + 1) % args.accum_steps == 0:
                scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
                scaler.step(opt)
                scaler.update()
                opt.zero_grad(set_to_none=True)
                step += 1
                if step % args.log_every == 0:
                    dt = time.time() - t0
                    print(f"epoch {epoch} step {step} loss {loss.item()*args.accum_steps:.4f} {dt:.1f}s")
                    t0 = time.time()
                if step % args.save_every == 0:
                    torch.save({"model": model.state_dict(), "cfg": cfg}, args.out)

    torch.save({"model": model.state_dict(), "cfg": cfg}, args.out)
    print("saved", args.out)


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Smoke-test (needs a pretrain checkpoint; use the small one from Task 12)**

Run: `python train/sft.py --preset small --init checkpoints/pretrain.pt --limit 500 --batch_size 4 --accum_steps 2 --epochs 1 --log_every 5 --save_every 100000`
Expected: loads checkpoint, loss descends over logged steps, saves `checkpoints/sft.pt`.

- [ ] **Step 3: Commit**

```bash
git add train/sft.py && git commit -m "feat: SFT loop with loss masking"
```

---

## Task 14: Inference — generation, completion & chat

**Files:**
- Create: `inference.py`

- [ ] **Step 1: Write `inference.py`**

```python
import argparse
import torch
import torch.nn.functional as F

from model.model import Transformer
from dataset import TokenizerAdapter


@torch.no_grad()
def generate(model, tok, prompt_ids, max_new_tokens=200, temperature=0.8, top_p=0.9, device="cpu"):
    model.eval()
    eos = tok.token_id("</s>")
    idx = torch.tensor([prompt_ids], dtype=torch.long, device=device)
    for _ in range(max_new_tokens):
        idx_cond = idx[:, -model.cfg.max_seq_len:]
        logits, _ = model(idx_cond)
        logits = logits[:, -1, :] / max(temperature, 1e-5)
        probs = F.softmax(logits, dim=-1)
        # nucleus (top-p) sampling
        sorted_probs, sorted_idx = torch.sort(probs, descending=True)
        cum = torch.cumsum(sorted_probs, dim=-1)
        mask = cum - sorted_probs > top_p
        sorted_probs[mask] = 0.0
        sorted_probs /= sorted_probs.sum(dim=-1, keepdim=True)
        next_sorted = torch.multinomial(sorted_probs, num_samples=1)
        next_id = sorted_idx.gather(-1, next_sorted)
        idx = torch.cat([idx, next_id], dim=1)
        if next_id.item() == eos:
            break
    return idx[0].tolist()


def build_chat_prompt(tok, user_msg):
    bos, eos = tok.token_id("<s>"), tok.token_id("</s>")
    ids = [bos] + tok.encode_ids("user\n" + user_msg) + [eos] + tok.encode_ids("\n")
    ids += [bos] + tok.encode_ids("assistant\n")
    return ids


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="checkpoints/sft.pt")
    ap.add_argument("--mode", choices=["complete", "chat"], default="chat")
    ap.add_argument("--prompt", default="你好")
    ap.add_argument("--max_new_tokens", type=int, default=200)
    ap.add_argument("--temperature", type=float, default=0.8)
    ap.add_argument("--top_p", type=float, default=0.9)
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    tok = TokenizerAdapter.from_files()
    ckpt = torch.load(args.ckpt, map_location=device)
    model = Transformer(ckpt["cfg"]).to(device)
    model.load_state_dict(ckpt["model"])

    if args.mode == "chat":
        prompt_ids = build_chat_prompt(tok, args.prompt)
    else:
        prompt_ids = [tok.token_id("<s>")] + tok.encode_ids(args.prompt)

    out_ids = generate(model, tok, prompt_ids, args.max_new_tokens,
                        args.temperature, args.top_p, device)
    gen_ids = out_ids[len(prompt_ids):]
    print(tok.decode(gen_ids))


if __name__ == "__main__":
    main()
```

> **Note on KV cache:** the `Attention`/`TransformerBlock` modules support an optional `kv_cache` path (built in Tasks 4 & 6) for future optimization. This `generate` uses the simpler full-recompute path for clarity; switching to the cached path is a clean follow-up that does not change outputs.

- [ ] **Step 2: Smoke-test with the small SFT checkpoint**

Run: `python inference.py --ckpt checkpoints/sft.pt --mode chat --prompt "你好" --max_new_tokens 40`
Expected: prints some generated Chinese text. With only the smoke-test training it will be incoherent — that's fine; the goal here is that generation runs end-to-end without error.

- [ ] **Step 3: Commit**

```bash
git add inference.py && git commit -m "feat: inference with top-p sampling, completion and chat modes"
```

---

## Task 15: Full test run + README usage docs

**Files:**
- Modify: `README.md`

- [ ] **Step 1: Run the whole test suite**

Run: `pytest -v`
Expected: all tests in `tests/` PASS (tokenizer tests may SKIP if not trained).

- [ ] **Step 2: Append usage docs to `README.md`**

```markdown
## Pipeline

1. Download `pretrain_hq.jsonl` and `sft_mini_512.jsonl` into `data/`.
2. Train tokenizer: `python train/train_tokenizer.py`
3. Pretrain: `python train/pretrain.py --preset base`
4. SFT: `python train/sft.py --preset base`
5. Chat: `python inference.py --ckpt checkpoints/sft.pt --mode chat --prompt "你好"`

Use `--preset small` end-to-end first to validate the pipeline quickly.

## Sanity checks
- `pytest tests/test_model.py::test_overfit_single_batch` — must pass before training.
- Tokenizer round-trip: `pytest tests/test_tokenizer.py`.
```

- [ ] **Step 3: Commit**

```bash
git add README.md && git commit -m "docs: pipeline usage and sanity checks"
```

---

## Self-Review Notes (for the implementer)

- **Spec coverage:** tokenizer (T9–10), modern decoder RMSNorm/RoPE/GQA/SwiGLU (T2–7), weight tying (T7), presets (T1), pretrain (T12), SFT with loss masking (T11, T13), inference/chat (T14), memory tactics — bf16 + grad accumulation + seq 512 (T12–13), per-stage sanity checks (T8, T10, T12 step2, T13 step2). All design sections map to tasks.
- **Validate small-first:** always run the `--preset small` smoke path (Tasks 12–14 step 2) before committing GPU hours to the base preset.
- **If overfit test (T8) fails:** stop and debug RoPE application, the causal flag in attention, loss masking, or weight tying before training anything.
```

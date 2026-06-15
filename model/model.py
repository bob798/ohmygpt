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

        xq = xq.transpose(1, 2)
        xk = xk.transpose(1, 2)
        xv = xv.transpose(1, 2)

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

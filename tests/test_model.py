import torch
from model.config import small_config
from model.model import (
    RMSNorm,
    precompute_freqs_cis,
    apply_rotary_emb,
    Attention,
    FeedForward,
    TransformerBlock,
    Transformer,
)


def test_rmsnorm_preserves_shape_and_normalizes():
    torch.manual_seed(0)
    norm = RMSNorm(dim=16)
    x = torch.randn(2, 5, 16) * 10.0
    out = norm(x)
    assert out.shape == x.shape
    rms = out.pow(2).mean(dim=-1).sqrt()
    assert torch.allclose(rms, torch.ones_like(rms), atol=1e-2)


def test_rope_shapes_and_rotation_invariant_norm():
    head_dim, seq = 32, 8
    freqs_cis = precompute_freqs_cis(head_dim, seq, theta=10000.0)
    assert freqs_cis.shape == (seq, head_dim // 2)
    q = torch.randn(1, seq, 2, head_dim)
    k = torch.randn(1, seq, 2, head_dim)
    q_r, k_r = apply_rotary_emb(q, k, freqs_cis)
    assert q_r.shape == q.shape and k_r.shape == k.shape
    assert torch.allclose(q_r.norm(dim=-1), q.norm(dim=-1), atol=1e-4)


def test_attention_output_shape_and_causality():
    cfg = small_config(max_seq_len=8)
    attn = Attention(cfg)
    B, T = 2, 6
    x = torch.randn(B, T, cfg.dim)
    freqs_cis = precompute_freqs_cis(cfg.head_dim, cfg.max_seq_len)[:T]
    out = attn(x, freqs_cis, start_pos=0)
    assert out.shape == (B, T, cfg.dim)


def test_feedforward_shape():
    cfg = small_config()
    ff = FeedForward(cfg)
    x = torch.randn(2, 5, cfg.dim)
    assert ff(x).shape == (2, 5, cfg.dim)


def test_transformer_block_shape():
    cfg = small_config(max_seq_len=8)
    block = TransformerBlock(cfg)
    B, T = 2, 6
    x = torch.randn(B, T, cfg.dim)
    freqs_cis = precompute_freqs_cis(cfg.head_dim, cfg.max_seq_len)[:T]
    assert block(x, freqs_cis).shape == (B, T, cfg.dim)


def test_transformer_forward_and_loss():
    cfg = small_config(max_seq_len=16)
    model = Transformer(cfg)
    B, T = 2, 10
    idx = torch.randint(0, cfg.vocab_size, (B, T))
    targets = torch.randint(0, cfg.vocab_size, (B, T))
    logits, loss = model(idx, targets)
    assert logits.shape == (B, T, cfg.vocab_size)
    assert loss.ndim == 0 and loss.item() > 0
    assert model.tok_embeddings.weight is model.lm_head.weight


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

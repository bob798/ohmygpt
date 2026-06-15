# ohmygpt — A From-Scratch LLM (minimind-style) — Design

**Date:** 2026-06-15
**Status:** Approved design, pending implementation plan

## Goal & Constraints

Build a small Chinese language model from scratch to **learn the internals** of modern
LLMs end-to-end. Code clarity and pedagogical transparency take priority over raw
performance. Inspired by [minimind](https://github.com/jingyaogong/minimind).

- **Hardware:** single RTX 3060 (12GB VRAM).
- **Language:** Chinese (uses minimind's published datasets to keep data-prep trivial).
- **Scope (this spec):** tokenizer → model architecture → pretraining → SFT.
  Outcome: a base model that completes text and a chat model that follows instructions.
- **Deferred (future specs):** LoRA, DPO, MoE variant. The codebase is structured so
  these drop in cleanly later.
- **Philosophy:** every core mechanism is hand-written and readable. Libraries are used
  only for boring plumbing (data loading, BPE training). No accelerate/deepspeed — we
  write the training loop ourselves so every step is visible.

## Project Structure

```
ohmygpt/
├── model/
│   ├── config.py          # dataclass: dims, layers, heads, vocab, etc.
│   ├── model.py           # the Transformer (RMSNorm, RoPE, GQA, SwiGLU)
│   └── tokenizer/         # trained BPE tokenizer files live here
├── data/                  # downloaded minimind datasets (gitignored)
├── train/
│   ├── train_tokenizer.py # train BPE tokenizer from corpus
│   ├── pretrain.py        # next-token pretraining loop
│   └── sft.py             # instruction fine-tuning loop
├── dataset.py             # PretrainDataset + SFTDataset (Dataset classes)
├── inference.py           # load checkpoint, sample/chat
├── configs/               # small/base config presets
├── requirements.txt
└── README.md
```

**Dependencies:** `torch`, `tokenizers` (HF BPE trainer), `numpy`, `tqdm`, and optionally
`transformers` for its `PreTrainedTokenizerFast` wrapper. Nothing heavier.

## Model Architecture

Decoder-only transformer, modern Llama/Qwen-style components, all hand-written.

### Config presets

| Param        | `base` (~26M) | `small` (~6M, for fast debug) |
|--------------|---------------|-------------------------------|
| vocab_size   | 6400          | 6400                          |
| dim (hidden) | 512           | 256                           |
| n_layers     | 8             | 4                             |
| n_heads      | 16            | 8                             |
| n_kv_heads   | 8 (GQA)       | 4 (GQA)                       |
| max_seq_len  | 512           | 512                           |
| FFN hidden   | ~1376 (≈8/3·dim, rounded) | ~704              |

### Forward pass (each piece a small, documented module)

1. **Token embedding** → `(B, T, dim)`. Embedding and output `lm_head` weights are **tied**.
2. **N × TransformerBlock**, each:
   - `RMSNorm` → **Attention** → residual add
     - RoPE applied to Q and K
     - GQA: fewer KV heads than Q heads, KV heads repeated to match
     - causal mask
     - optional KV-cache for generation
   - `RMSNorm` → **SwiGLU FFN** (`w2(silu(w1(x)) * w3(x))`) → residual add
3. Final `RMSNorm` → `lm_head` → logits `(B, T, vocab)`.

### Why these components (the learning payoff)

- **RMSNorm** — simpler and faster than LayerNorm; no mean-centering.
- **RoPE** — relative positions baked into attention via rotation; no learned position
  table; extrapolates better to longer sequences.
- **GQA** — fewer KV heads cuts KV-cache memory; the trick that makes inference cheap.
- **SwiGLU** — gated FFN used by modern models; outperforms plain GELU MLP.

**Loss:** cross-entropy on next-token prediction.
**Sampling:** temperature + top-p (nucleus).

## Data Pipeline & Training Stages

### Datasets (minimind's published Chinese data, into `data/`, gitignored)

- `pretrain_hq.jsonl` — ~1.6GB clean Chinese text, each line `{"text": "..."}`.
- `sft_mini_512.jsonl` — instruction/response pairs for chat tuning.

### Stage 0 — Tokenizer (`train/train_tokenizer.py`)

Train a byte-level BPE (vocab 6400) on a sample of the pretrain corpus via HF `tokenizers`.
Special tokens defined up front: `<unk>`, `<s>`, `</s>`. Chat template:
`<s>user\n...</s>\n<s>assistant\n...</s>`. Saved to `model/tokenizer/`.
*Payoff: see exactly how raw text becomes integer IDs.*

### Stage 1 — Pretrain (`train/pretrain.py`)

- `PretrainDataset` packs text into fixed 512-token windows.
- Loop: AdamW, cosine LR schedule with warmup, gradient accumulation (fit 12GB),
  `torch.cuda.amp` mixed precision (bf16/fp16), gradient clipping.
- Periodic checkpoints.
- **Output:** base model that completes Chinese text. ~1 epoch over HQ set ≈ a few hours
  on a 3060.

### Stage 2 — SFT (`train/sft.py`)

- `SFTDataset` formats each pair with the chat template and **masks the loss so only
  assistant tokens contribute** (prompt tokens are masked out — a key detail to see).
- Same loop, lower LR, initialized from the pretrain checkpoint.
- **Output:** chat model that follows instructions.

### Inference (`inference.py`)

Load a checkpoint; KV-cached autoregressive generation. Two modes: raw completion (base)
and chat (applies template, multi-turn).

### Memory tactics for 12GB

Mixed precision + gradient accumulation + seq_len 512 + small batch. Batch size and
accumulation steps are exposed in config for tuning.

## Verification & Learning Checkpoints

Cheap per-stage sanity checks that catch bugs before they waste GPU hours:

- **Tokenizer:** round-trip `decode(encode(text)) == text` on Chinese samples; print vocab
  size and a sample tokenization.
- **Model:** `overfit-one-batch` test — loss should drop near zero on a single batch within
  a few hundred steps. Best single signal the architecture + loop are wired correctly.
- **Pretrain:** loss curve descends; periodically generate a completion to eyeball coherence.
- **SFT:** verify loss mask (prompt tokens masked); model responds to a held-out instruction.

Minimal checks only — not a heavy test suite. They catch the bugs that otherwise ruin
training runs: wrong masking, incorrect RoPE, weight-tying mistakes.

## Out of Scope (future specs)

LoRA fine-tuning, DPO preference alignment, MoE variant, multi-GPU/DDP, model distillation.

<p align="center">
  <img src="assets/banner.svg" alt="ohmygpt — a from-scratch Chinese LLM" width="100%">
</p>

<p align="center"><a href="README.md">中文</a> · <b>English</b></p>

# ohmygpt

> A small Chinese LLM built **from scratch**, for learning how modern LLMs actually work — minimind-style.

`tokenizer → pretrain → SFT`. Every core mechanism is hand-written and readable; libraries are used only for boring plumbing (data loading, BPE training).

## About

ohmygpt's goal is to **learn LLM internals**, not to chase SOTA. It reproduces the key design of a modern decoder (the same components as Llama / Qwen) and provides a complete-but-minimal training pipeline that trains a small model from zero — capable of completing Chinese text and holding a basic conversation — on a single **RTX 3060 (12GB)**.

The code is deliberately lightweight and transparent:

- **No heavy frameworks** — no accelerate/deepspeed; the training loop is hand-written so every step is visible.
- **Modern architecture, all hand-written** — RMSNorm, RoPE rotary positions, grouped-query attention (GQA), SwiGLU feed-forward, tied embeddings.
- **Complete pipeline** — train tokenizer → pretrain a base model → instruction tuning (SFT, with prompt-token loss masking) → inference (top-p sampling + multi-turn chat).
- **Correctness first** — includes an overfit-one-batch unit test as the key signal that the architecture is wired correctly.

Design docs & implementation plan live in [`docs/superpowers/`](docs/superpowers/). Inspired by [jingyaogong/minimind](https://github.com/jingyaogong/minimind).

## Features

| Module | Files | Notes |
|--------|-------|-------|
| Model | `model/config.py`, `model/model.py` | Decoder: RMSNorm · RoPE · GQA · SwiGLU · tied weights |
| Tokenizer | `train/train_tokenizer.py` | byte-level BPE, vocab 6400, specials `<unk>/<s>/</s>` + chat template |
| Datasets | `dataset.py` | `PretrainDataset` (packed fixed windows) + `SFTDataset` (chat template, loss only on the answer) |
| Training | `train/pretrain.py`, `train/sft.py` | AdamW, cosine schedule + warmup, bf16/fp16 AMP, gradient accumulation, grad clipping |
| Inference | `inference.py` | top-p (nucleus) sampling, KV-cache fast path, completion & chat modes |
| Tests | `tests/` | RMSNorm/RoPE/attention/FFN/full-model units + overfit check + tokenizer round-trip + loss-mask + cached-vs-naive generation |

## Model presets

Two configs: use `small` to validate the pipeline quickly, then `base` for the real run.

| Param | `small` (~6M) | `base` (~26M) |
|-------|---------------|---------------|
| dim | 256 | 512 |
| n_layers | 4 | 8 |
| n_heads | 8 | 16 |
| n_kv_heads (GQA) | 4 | 8 |
| max_seq_len | 512 | 512 |
| vocab_size | 6400 | 6400 |

## Setup

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

> Training needs an NVIDIA GPU (RTX 3060 12GB or better recommended). CPU/MPS can run the unit tests but is not suitable for actual training.

## Pipeline

1. Download the minimind datasets into `data/`:
   ```bash
   python scripts/download_data.py            # or: --endpoint https://hf-mirror.com
   ```
2. Train the tokenizer: `python train/train_tokenizer.py`
3. Pretrain: `python train/pretrain.py --preset base`
4. SFT: `python train/sft.py --preset base`
5. Chat: `python inference.py --ckpt checkpoints/sft.pt --mode chat --prompt "你好"`

**Run `--preset small` end-to-end first** to confirm the pipeline before committing GPU hours:

```bash
python train/pretrain.py --preset small --limit 2000 --batch_size 4 --accum_steps 2
```

If you run out of VRAM, lower `--batch_size` and raise `--accum_steps` to keep the effective batch constant.

## Sanity checks

Run these before training — they catch wiring bugs before they waste GPU time:

- `pytest tests/test_model.py::test_overfit_single_batch` — single-batch overfit; loss must drop below 0.1.
- `pytest tests/test_generate.py` — cached generation must equal naive full-recompute.
- `pytest tests/test_tokenizer.py` — tokenizer encode/decode round-trip.

Full suite: `pytest -v`.

## Acknowledgements

- Inspiration and datasets from [jingyaogong/minimind](https://github.com/jingyaogong/minimind).
- Architecture follows the modern decoder design of the Llama / Qwen families.

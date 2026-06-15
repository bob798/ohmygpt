# ohmygpt

A small Chinese LLM built from scratch for learning — minimind-style.
Pipeline: tokenizer → pretrain → SFT. See `docs/superpowers/` for design & plan.

## Setup
```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

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

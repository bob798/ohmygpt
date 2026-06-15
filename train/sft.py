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

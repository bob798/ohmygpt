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
    kv_caches = [None] * model.cfg.n_layers
    # Prefill the prompt, then decode incrementally.
    cur = idx
    start_pos = 0
    generated = []
    for step in range(max_new_tokens):
        logits, kv_caches = model.forward_cached(cur, start_pos, kv_caches)
        start_pos += cur.shape[1]
        logits = logits[:, -1, :] / max(temperature, 1e-5)
        probs = F.softmax(logits, dim=-1)
        sorted_probs, sorted_idx = torch.sort(probs, descending=True)
        cum = torch.cumsum(sorted_probs, dim=-1)
        mask = cum - sorted_probs > top_p
        sorted_probs[mask] = 0.0
        denom = sorted_probs.sum(dim=-1, keepdim=True)
        # Fall back to the top token if everything underflowed to zero.
        if (denom <= 0).any():
            next_sorted = torch.zeros(sorted_probs.size(0), 1, dtype=torch.long, device=sorted_probs.device)
        else:
            sorted_probs /= denom
            next_sorted = torch.multinomial(sorted_probs, num_samples=1)
        next_id = sorted_idx.gather(-1, next_sorted)
        generated.append(next_id.item())
        if next_id.item() == eos:
            break
        # If we ever exceed context, stop (cache has no eviction).
        if start_pos + 1 > model.cfg.max_seq_len:
            break
        cur = next_id  # next step processes just the new token
    return prompt_ids + generated


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
    ckpt = torch.load(args.ckpt, map_location=device, weights_only=False)
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

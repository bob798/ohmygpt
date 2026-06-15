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
    ids = {t: tokenizer.token_to_id(t) for t in SPECIAL_TOKENS}
    with open(os.path.join(args.out, "special_tokens.json"), "w", encoding="utf-8") as f:
        json.dump(ids, f, ensure_ascii=False, indent=2)
    print("vocab size:", tokenizer.get_vocab_size(), "special:", ids)


if __name__ == "__main__":
    main()

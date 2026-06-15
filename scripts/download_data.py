"""Download the minimind datasets used by ohmygpt into `data/`.

Dependency-free (stdlib only). Pulls files from the HuggingFace dataset repo
`jingyaogong/minimind_dataset`. If huggingface.co is slow or blocked, pass a
mirror, e.g. `--endpoint https://hf-mirror.com`.

Usage:
    python scripts/download_data.py                 # pretrain + sft (default)
    python scripts/download_data.py --files pretrain_hq.jsonl
    python scripts/download_data.py --endpoint https://hf-mirror.com
"""
import argparse
import os
import sys
import urllib.request

REPO = "jingyaogong/minimind_dataset"
DEFAULT_FILES = ["pretrain_hq.jsonl", "sft_mini_512.jsonl"]


def file_url(endpoint: str, repo: str, filename: str) -> str:
    # HF dataset resolve URL: <endpoint>/datasets/<repo>/resolve/main/<file>
    return f"{endpoint.rstrip('/')}/datasets/{repo}/resolve/main/{filename}"


def _progress(block_num, block_size, total_size):
    if total_size <= 0:
        return
    downloaded = block_num * block_size
    pct = min(100, downloaded * 100 // total_size)
    mb = downloaded / 1e6
    total_mb = total_size / 1e6
    sys.stdout.write(f"\r  {pct:3d}%  {mb:7.1f} / {total_mb:.1f} MB")
    sys.stdout.flush()


def download(endpoint: str, repo: str, filename: str, out_dir: str, force: bool):
    dest = os.path.join(out_dir, filename)
    if os.path.exists(dest) and not force:
        print(f"skip {filename} (exists; use --force to re-download)")
        return
    url = file_url(endpoint, repo, filename)
    print(f"downloading {filename}\n  from {url}")
    tmp = dest + ".part"
    try:
        urllib.request.urlretrieve(url, tmp, _progress)
        os.replace(tmp, dest)
        print(f"\n  saved -> {dest}")
    except Exception as e:
        if os.path.exists(tmp):
            os.remove(tmp)
        print(f"\n  FAILED: {e}", file=sys.stderr)
        print("  Tip: try a mirror with --endpoint https://hf-mirror.com", file=sys.stderr)
        raise


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--endpoint", default=os.environ.get("HF_ENDPOINT", "https://huggingface.co"),
                    help="HF endpoint (default: env HF_ENDPOINT or huggingface.co)")
    ap.add_argument("--repo", default=REPO, help="HF dataset repo id")
    ap.add_argument("--files", nargs="+", default=DEFAULT_FILES, help="files to fetch")
    ap.add_argument("--out", default="data", help="output directory")
    ap.add_argument("--force", action="store_true", help="re-download even if present")
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    for f in args.files:
        download(args.endpoint, args.repo, f, args.out, args.force)
    print("done.")


if __name__ == "__main__":
    main()

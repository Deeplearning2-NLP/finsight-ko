"""Download raw datasets for the proposal stage (T1).

Saves HF dataset repos as plain files under data/raw/ and prints row counts.
Usage: python scripts/download_data.py [--only dartdoc findartbench autorag]
"""
from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path

from huggingface_hub import snapshot_download

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"

HF_DATASETS = {
    "dartdoc": "chaannwooff/Dartdoc",
    "findartbench": "davidkim205/FinDartBench",
}


def count_jsonl(path: Path) -> int:
    with path.open("rb") as f:
        return sum(1 for _ in f)


def download_hf(name: str, repo: str) -> None:
    out = RAW / name
    snapshot_download(repo_id=repo, repo_type="dataset", local_dir=out,
                      allow_patterns=["*.jsonl", "*.parquet", "README.md"])
    files = sorted(out.rglob("*.jsonl"))
    n = sum(count_jsonl(p) for p in files)
    print(json.dumps({"dataset": name, "repo": repo, "files": len(files), "rows": n}, ensure_ascii=False))


def download_autorag() -> None:
    subprocess.run(["bash", str(ROOT / "scripts" / "download_autorag.sh"), str(RAW / "autorag")], check=True)
    import pandas as pd
    qa = pd.read_parquet(RAW / "autorag" / "qa.parquet")
    corpus = pd.read_parquet(RAW / "autorag" / "corpus.parquet")
    print(json.dumps({"dataset": "autorag", "queries": len(qa), "docs": len(corpus)}))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", nargs="*", default=["dartdoc", "findartbench", "autorag"])
    args = ap.parse_args()
    for name in args.only:
        if name == "autorag":
            download_autorag()
        else:
            download_hf(name, HF_DATASETS[name])


if __name__ == "__main__":
    main()

"""BM25 tokenizer comparison on AutoRAGRetrieval.

Usage:
    bash scripts/download_autorag.sh
    python scripts/eval_autorag_bm25.py --data data/raw/autorag
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from finsight.bm25 import BM25Retriever
from finsight.metrics import evaluate

# Recall@k from the public leaderboard (README of the benchmark repo), for reference
LEADERBOARD = {
    "KU-HIAI-ONTHEIT-large-v1.1": {"R@1": 0.7193, "R@3": 0.8860, "R@5": 0.9298, "R@10": 0.9474},
    "bge-m3": {"R@1": 0.6754, "R@3": 0.8684, "R@5": 0.9298, "R@10": 0.9649},
    "multilingual-e5-large-instruct": {"R@1": 0.4649, "R@3": 0.6667, "R@5": 0.7632, "R@10": 0.8158},
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/raw/autorag")
    ap.add_argument("--out", default="outputs/autorag_bm25.json")
    args = ap.parse_args()

    qa = pd.read_parquet(Path(args.data) / "qa.parquet")
    corpus = pd.read_parquet(Path(args.data) / "corpus.parquet")
    gold = [{d for grp in row for d in grp} for row in qa["retrieval_gt"]]

    results = {}
    for tok in ["whitespace", "kiwi_all", "kiwi_content"]:
        r = BM25Retriever(corpus["doc_id"], corpus["contents"], tokenizer=tok)
        runs = [(r.search(q, 10), g) for q, g in zip(qa["query"], gold)]
        results[f"BM25 {tok}"] = evaluate(runs)

    df = pd.DataFrame(results).T
    print(f"queries={len(qa)} docs={len(corpus)}")
    print(df.round(4).to_string())
    print("\nleaderboard (dense, reference):")
    print(pd.DataFrame(LEADERBOARD).T.to_string())

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(results, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

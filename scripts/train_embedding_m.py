"""M3: fine-tune KURE-v1 on the M1 pairs with different negative mixes (chosen on dev).

Same hyper-parameters as T6/S1 (CachedMNRL, batch 32 / mini-batch 8, lr 1e-5, warmup 10%, 1 epoch, max 512,
bf16, NO_DUPLICATES, seed 42). --neg:
  none    : in-batch negatives only
  easy    : + a random chunk of another training company
  hard    : + other-fiscal-year negative where the question states a year, else other-section negative
  hard50  : hard for a seeded half of the pairs, easy for the rest
  section : + other-section negative of the same company
Leakage asserts: no evaluation company, filing, chunk id or FinDartBench question in the training pairs.

Usage: CUDA_VISIBLE_DEVICES=0 python scripts/train_embedding_m.py --neg hard
Outputs: outputs/midterm/kure-m-<neg>/ (checkpoint, git-ignored), outputs/midterm/train_meta_<neg>.json
"""
from __future__ import annotations

import argparse
import json
import random
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from build_corpus import COMPANIES as EVAL_COMPANIES  # noqa: E402
from eval_findartbench_retrieval import load  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs" / "midterm"


def check_leakage(pairs: pd.DataFrame) -> None:
    assert not set(pairs.corp_name) & set(EVAL_COMPANIES), "evaluation company in training pairs"
    ids = set(pairs.positive_id)
    for col in ("neg_year_id", "neg_section_id", "neg_easy_id"):
        if col in pairs:
            ids |= set(pairs[col].dropna())
    eval_rcepts = set(pd.read_csv(ROOT / "data/processed/filings_a.csv", dtype=str).rcept_no)
    assert not {i.split("_")[0] for i in ids} & eval_rcepts, "evaluation filing in training pairs"
    eval_chunks = set(pd.read_json(ROOT / "data/processed/corpus_a.jsonl", lines=True, dtype={"chunk_id": str}).chunk_id)
    assert not ids & eval_chunks, "evaluation chunk id in training pairs"
    fdb_q = set(load(ROOT / "data/raw/findartbench/data.jsonl").question.str.strip())
    assert not set(pairs.anchor.str.strip()) & fdb_q, "FinDartBench question in training pairs"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--neg", choices=["none", "easy", "hard", "hard50", "section"], required=True)
    ap.add_argument("--pairs", default=str(ROOT / "data/train/pairs_m1.jsonl"))
    ap.add_argument("--base", default="nlpai-lab/KURE-v1")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--tag", default="")
    args = ap.parse_args()
    tag = args.tag or args.neg
    OUT.mkdir(parents=True, exist_ok=True)

    pairs = pd.read_json(args.pairs, lines=True, dtype={"positive_id": str, "neg_year_id": str,
                                                         "neg_section_id": str, "neg_easy_id": str})
    check_leakage(pairs)
    print(f"leakage checks passed: {len(pairs)} pairs, {pairs.corp_name.nunique()} companies", flush=True)

    rng = random.Random(args.seed)
    hard = [y if isinstance(y, str) else (s if isinstance(s, str) else e)
            for y, s, e in zip(pairs.neg_year, pairs.neg_section, pairs.neg_easy)]
    neg = {"easy": list(pairs.neg_easy), "hard": hard,
           "hard50": [h if rng.random() < 0.5 else e for h, e in zip(hard, pairs.neg_easy)],
           "section": [s if isinstance(s, str) else e for s, e in zip(pairs.neg_section, pairs.neg_easy)]}.get(args.neg)

    import torch
    from datasets import Dataset
    from sentence_transformers import (SentenceTransformer, SentenceTransformerTrainer,
                                       SentenceTransformerTrainingArguments)
    from sentence_transformers.losses import CachedMultipleNegativesRankingLoss
    from sentence_transformers.training_args import BatchSamplers

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    cols = {"anchor": pairs.anchor.tolist(), "positive": pairs.positive.tolist()}
    if neg is not None:
        cols["negative"] = neg
    ds = Dataset.from_dict(cols).shuffle(seed=args.seed)
    model = SentenceTransformer(args.base, device="cuda")
    model.max_seq_length = 512
    loss = CachedMultipleNegativesRankingLoss(model, mini_batch_size=8)
    targs = SentenceTransformerTrainingArguments(
        output_dir=str(OUT / f"_tmp_{tag}"), num_train_epochs=1, per_device_train_batch_size=32, learning_rate=1e-5,
        warmup_ratio=0.1, bf16=True, batch_sampler=BatchSamplers.NO_DUPLICATES, seed=args.seed,
        save_strategy="no", logging_steps=20, report_to="none")
    trainer = SentenceTransformerTrainer(model=model, args=targs, train_dataset=ds, loss=loss)
    torch.cuda.reset_peak_memory_stats()
    t0 = time.time()
    trainer.train()
    secs = time.time() - t0
    out_dir = OUT / f"kure-m-{tag}"
    model.save(str(out_dir))
    log = [h for h in trainer.state.log_history if "loss" in h]
    meta = {"neg": args.neg, "pairs": len(ds), "train_seconds": round(secs, 1),
            "gpu_peak_mem_gb": round(torch.cuda.max_memory_allocated() / 2**30, 2),
            "gpu": torch.cuda.get_device_name(0), "steps": trainer.state.global_step,
            "loss_first": log[0]["loss"] if log else None, "loss_last": log[-1]["loss"] if log else None,
            "neg_source_counts": None if neg is None else {
                "year": sum(isinstance(x, str) and x == y for x, y in zip(neg, pairs.neg_year)),
                "section": sum(isinstance(x, str) and x == s for x, s in zip(neg, pairs.neg_section)),
                "easy": sum(x == e for x, e in zip(neg, pairs.neg_easy))},
            "checkpoint": str(out_dir.relative_to(ROOT))}
    (OUT / f"train_meta_{tag}.json").write_text(json.dumps(meta, ensure_ascii=False, indent=1))
    print(json.dumps(meta, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()

"""S1: fine-tune KURE-v1 on pairs from the 20 non-evaluation companies' originals (company-split).

Same hyper-parameters as the T6 smoke run (CachedMultipleNegativesRankingLoss, batch 32 / mini-batch 8,
lr 1e-5, warmup 10%, 1 epoch, max_seq_length 512, bf16, NO_DUPLICATES batches, seed 42).
  --neg none : (a) in-batch negatives only            (anchor, positive)
  --neg year : (b) + one other-fiscal-year hard negative  (anchor, positive, negative) from pairs_s1.jsonl
Asserts that nothing from the evaluation side is in the training data: no evaluation company, no
evaluation filing / chunk id, no FinDartBench question text.

Usage: CUDA_VISIBLE_DEVICES=2 python scripts/train_embedding_s1.py --neg none
Outputs: outputs/proposal/kure-s1-<a|b>[-n<N>]/ (one checkpoint), s1_train_meta_<a|b>[...].json
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
OUT = ROOT / "outputs" / "proposal"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--neg", choices=["none", "year"], required=True)
    ap.add_argument("--pairs", default=str(ROOT / "data/train/pairs_s1.jsonl"))
    ap.add_argument("--n", type=int, default=0, help="use the first N pairs after a seeded shuffle (S2); 0 = all")
    ap.add_argument("--base", default="nlpai-lab/KURE-v1")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    tag = {"none": "a", "year": "b"}[args.neg] + (f"-n{args.n}" if args.n else "")

    pairs = pd.read_json(args.pairs, lines=True, dtype={"positive_id": str, "negative_id": str})
    # ---- leakage checks (S1 requirement)
    assert not set(pairs.corp_name) & set(EVAL_COMPANIES), "evaluation company in training pairs"
    eval_rcepts = set(pd.read_csv(ROOT / "data/processed/filings_a.csv", dtype=str).rcept_no)
    ids = set(pairs.positive_id) | set(pairs.negative_id.dropna())
    assert not {i.split("_")[0] for i in ids} & eval_rcepts, "evaluation filing in training pairs"
    eval_chunks = set(pd.read_json(ROOT / "data/processed/corpus_a.jsonl", lines=True,
                                   dtype={"chunk_id": str}).chunk_id)
    assert not ids & eval_chunks, "evaluation chunk id in training pairs"
    fdb_q = set(load(ROOT / "data/raw/findartbench/data.jsonl").question.str.strip())
    assert not set(pairs.anchor.str.strip()) & fdb_q, "FinDartBench question in training pairs"
    print(f"leakage checks passed: {len(pairs)} pairs, {pairs.corp_name.nunique()} companies", flush=True)

    if args.n:
        idx = list(range(len(pairs)))
        random.Random(args.seed).shuffle(idx)
        pairs = pairs.iloc[sorted(idx[:args.n])].reset_index(drop=True)
    if args.neg == "year":
        pairs = pairs[pairs.negative.notna()].reset_index(drop=True)

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
    if args.neg == "year":
        cols["negative"] = pairs.negative.tolist()
    ds = Dataset.from_dict(cols).shuffle(seed=args.seed)

    model = SentenceTransformer(args.base, device="cuda")
    model.max_seq_length = 512
    loss = CachedMultipleNegativesRankingLoss(model, mini_batch_size=8)
    targs = SentenceTransformerTrainingArguments(
        output_dir=str(OUT / f"_trainer_tmp_{tag}"), num_train_epochs=1, per_device_train_batch_size=32,
        learning_rate=1e-5, warmup_ratio=0.1, bf16=True, batch_sampler=BatchSamplers.NO_DUPLICATES,
        seed=args.seed, save_strategy="no", logging_steps=20, report_to="none")
    trainer = SentenceTransformerTrainer(model=model, args=targs, train_dataset=ds, loss=loss)
    torch.cuda.reset_peak_memory_stats()
    t0 = time.time()
    trainer.train()
    secs = time.time() - t0
    out_dir = OUT / f"kure-s1-{tag}"
    model.save(str(out_dir))
    log = [h for h in trainer.state.log_history if "loss" in h]
    meta = {"setting": {"a": "(a) in-batch only", "b": "(b) + other-fiscal-year hard negative"}[tag[0]],
            "pairs": len(ds), "base": args.base, "epochs": 1, "batch": 32, "mini_batch": 8, "lr": 1e-5,
            "max_seq_length": 512, "seed": args.seed, "train_seconds": round(secs, 1),
            "gpu_peak_mem_gb": round(torch.cuda.max_memory_allocated() / 2**30, 2),
            "gpu": torch.cuda.get_device_name(0), "steps": trainer.state.global_step,
            "loss_first": log[0]["loss"] if log else None, "loss_last": log[-1]["loss"] if log else None,
            "checkpoint": str(out_dir.relative_to(ROOT))}
    (OUT / f"s1_train_meta_{tag}.json").write_text(json.dumps(meta, ensure_ascii=False, indent=1))
    print(json.dumps(meta, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()

"""Fine-tuning smoke test for the retrieval embedding (T6).

Train pairs come from the *train* split of outputs/proposal/splits.json only:
  anchor   = FinDartBench question
  positive = the <=512-token chunk of its gold context with the highest BM25 (Kiwi content) score
  negative = the highest-scoring chunk from a *different* context of the same company (hard negative)
Loss: CachedMultipleNegativesRankingLoss (in-batch + hard negatives), NO_DUPLICATES batches
(many questions share a context, so duplicate positives inside a batch would be false negatives).
Dev split is evaluated after training; test is evaluated separately by eval_findartbench_retrieval.py.

Usage: CUDA_VISIBLE_DEVICES=1 python scripts/train_embedding_smoke.py
"""
from __future__ import annotations

import argparse
import json
import random
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from eval_findartbench_retrieval import load, make_splits  # noqa: E402

from finsight.bm25 import BM25Retriever
from finsight.chunking import chunk_text

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs" / "proposal"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="nlpai-lab/KURE-v1")
    ap.add_argument("--out", default=str(OUT / "kure-v1-findart-smoke"))
    ap.add_argument("--epochs", type=int, default=1)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--mini-batch", type=int, default=8)
    ap.add_argument("--lr", type=float, default=1e-5)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    import torch
    from datasets import Dataset
    from sentence_transformers import (SentenceTransformer, SentenceTransformerTrainer,
                                       SentenceTransformerTrainingArguments)
    from sentence_transformers.losses import CachedMultipleNegativesRankingLoss
    from sentence_transformers.training_args import BatchSamplers

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    df, splits = make_splits(load(ROOT / "data/raw/findartbench/data.jsonl"), args.seed)
    saved = json.loads((OUT / "splits.json").read_text())
    assert saved["train"]["ctx_ids"] == splits["train"]["ctx_ids"], "splits.json mismatch"
    train = df[df.split == "train"]

    model = SentenceTransformer(args.base, device="cuda")
    model.max_seq_length = 512
    tok = model.tokenizer

    def ntok(s: str) -> int:
        return len(tok(s, add_special_tokens=False)["input_ids"])

    # chunks of train contexts only
    ctx = train.drop_duplicates("ctx_id")[["ctx_id", "company", "context"]]
    chunks, chunk_ctx, chunk_comp = [], [], []
    for r in ctx.itertuples():
        for piece in chunk_text(r.context, ntok, 512, 50):
            chunks.append(piece)
            chunk_ctx.append(r.ctx_id)
            chunk_comp.append(r.company)
    chunk_ctx = np.array(chunk_ctx)
    chunk_comp = np.array(chunk_comp)
    bm = BM25Retriever(list(range(len(chunks))), chunks, tokenizer="kiwi_content")

    anchors, positives, negatives = [], [], []
    n_no_neg = 0
    for r in train.itertuples():
        s = bm.bm25.get_scores(bm.tok(r.question))
        own = np.where(chunk_ctx == r.ctx_id)[0]
        pos = own[np.argmax(s[own])]
        other = np.where((chunk_ctx != r.ctx_id) & (chunk_comp == r.company))[0]
        if len(other) == 0:
            other = np.where(chunk_ctx != r.ctx_id)[0]
            n_no_neg += 1
        neg = other[np.argmax(s[other])]
        anchors.append(r.question)
        positives.append(chunks[pos])
        negatives.append(chunks[neg])
    ds = Dataset.from_dict({"anchor": anchors, "positive": positives, "negative": negatives}).shuffle(seed=args.seed)
    print(f"train pairs={len(ds)} contexts={len(ctx)} chunks={len(chunks)} no_same_company_neg={n_no_neg}", flush=True)

    loss = CachedMultipleNegativesRankingLoss(model, mini_batch_size=args.mini_batch)
    targs = SentenceTransformerTrainingArguments(
        output_dir=str(OUT / "_trainer_tmp"), num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch, learning_rate=args.lr, warmup_ratio=0.1,
        bf16=True, batch_sampler=BatchSamplers.NO_DUPLICATES, seed=args.seed,
        save_strategy="no", logging_steps=10, report_to="none")
    trainer = SentenceTransformerTrainer(model=model, args=targs, train_dataset=ds, loss=loss)
    torch.cuda.reset_peak_memory_stats()
    t0 = time.time()
    trainer.train()
    train_sec = time.time() - t0
    peak = torch.cuda.max_memory_allocated() / 2**30
    model.save(args.out)  # single checkpoint
    log = [h for h in trainer.state.log_history if "loss" in h]
    meta = {"base": args.base, "train_pairs": len(ds), "epochs": args.epochs, "batch": args.batch,
            "mini_batch": args.mini_batch, "lr": args.lr, "seed": args.seed, "max_seq_length": 512,
            "train_seconds": round(train_sec, 1), "gpu_peak_mem_gb": round(peak, 2),
            "gpu": torch.cuda.get_device_name(0), "loss_first": log[0]["loss"] if log else None,
            "loss_last": log[-1]["loss"] if log else None, "steps": trainer.state.global_step,
            "positive_mining": "BM25 kiwi_content best chunk in gold context",
            "hard_negative": "BM25 best chunk from another context of the same company"}
    (OUT / "t6_train_meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=1))
    print(json.dumps(meta, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()

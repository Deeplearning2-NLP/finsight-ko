"""M6: train a cross-encoder reranker (question, chunk) -> relevant? on the M1 pairs (20 non-evaluation companies).

Positives: (anchor, positive). Negatives: the stored other-year / other-section / easy negatives (label 0).
Input (question + chunk) is capped at 512 tokens (CLAUDE.md operating condition). BinaryCrossEntropyLoss,
1 epoch, lr 2e-5, batch 16, warmup 10%, bf16, seed 42. Leakage asserts as in train_embedding_m.py.
Base models (compared on dev): klue/roberta-base (KLUE-BERT-style sentence-pair classifier, trained here)
and BAAI/bge-reranker-v2-m3 (a public reranker, fine-tuned here; also evaluated zero-shot).

Usage: CUDA_VISIBLE_DEVICES=1 python scripts/train_reranker_m.py --base klue/roberta-base --tag klue
Outputs: outputs/midterm/reranker-<tag>/ (git-ignored), outputs/midterm/reranker_meta_<tag>.json
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
from train_embedding_m import check_leakage  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs" / "midterm"


def fix_token_types(ce) -> None:
    """RoBERTa-style models (klue/roberta-base: type_vocab_size=1) must not get token_type_ids=1 for the
    second segment (device-side assert). Drop token_type_ids from the tokenizer inputs for such models."""
    if getattr(ce.model.config, "type_vocab_size", 2) == 1:
        ce.tokenizer.model_input_names = [n for n in ce.tokenizer.model_input_names if n != "token_type_ids"]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--pairs", default=str(ROOT / "data/train/pairs_m1.jsonl"))
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    pairs = pd.read_json(args.pairs, lines=True, dtype={"positive_id": str, "neg_year_id": str,
                                                         "neg_section_id": str, "neg_easy_id": str})
    check_leakage(pairs)
    print(f"leakage checks passed: {len(pairs)} pairs", flush=True)
    rows = []
    for r in pairs.itertuples():
        rows.append((r.anchor, r.positive, 1.0))
        for n in (r.neg_year, r.neg_section, r.neg_easy):
            if isinstance(n, str):
                rows.append((r.anchor, n, 0.0))
    random.Random(args.seed).shuffle(rows)

    import torch
    from datasets import Dataset
    from sentence_transformers.cross_encoder import CrossEncoder, CrossEncoderTrainer, CrossEncoderTrainingArguments
    from sentence_transformers.cross_encoder.losses import BinaryCrossEntropyLoss

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    ds = Dataset.from_dict({"query": [a for a, _, _ in rows], "passage": [b for _, b, _ in rows],
                            "label": [c for _, _, c in rows]})
    model = CrossEncoder(args.base, num_labels=1, max_length=512, device="cuda")
    fix_token_types(model)
    loss = BinaryCrossEntropyLoss(model)
    targs = CrossEncoderTrainingArguments(
        output_dir=str(OUT / f"_tmp_rr_{args.tag}"), num_train_epochs=1, per_device_train_batch_size=16,
        learning_rate=args.lr, warmup_ratio=0.1, bf16=True, seed=args.seed, save_strategy="no",
        logging_steps=50, report_to="none")
    trainer = CrossEncoderTrainer(model=model, args=targs, train_dataset=ds, loss=loss)
    torch.cuda.reset_peak_memory_stats()
    t0 = time.time()
    trainer.train()
    secs = time.time() - t0
    out_dir = OUT / f"reranker-{args.tag}"
    model.save(str(out_dir))
    log = [h for h in trainer.state.log_history if "loss" in h]
    meta = {"base": args.base, "rows": len(rows), "positives": int(sum(c for _, _, c in rows)), "lr": args.lr,
            "train_seconds": round(secs, 1), "gpu_peak_mem_gb": round(torch.cuda.max_memory_allocated() / 2**30, 2),
            "steps": trainer.state.global_step, "loss_first": log[0]["loss"] if log else None,
            "loss_last": log[-1]["loss"] if log else None, "checkpoint": str(out_dir.relative_to(ROOT))}
    (OUT / f"reranker_meta_{args.tag}.json").write_text(json.dumps(meta, ensure_ascii=False, indent=1))
    print(json.dumps(meta, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()

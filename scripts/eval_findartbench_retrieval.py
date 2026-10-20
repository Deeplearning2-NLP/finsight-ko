"""Disclosure-domain retrieval baselines on FinDartBench (T4).

- 사업보고서 questions only; each question's gold = its own `context` passage.
- Splits are made on groups of (doc_id, context) connected components so that no
  doc_id and no context is shared across train / dev / test (seed fixed, saved).
- Corpus = all unique contexts, re-chunked to <=512 tokens (overlap 50); a context's
  score is the max over its chunks (MaxP). "trunc512" rows embed each full context
  truncated at 512 tokens instead, to show the cost of truncation.
- Methods: BM25 (whitespace / Kiwi all / Kiwi content), dense (mE5-base, bge-m3,
  KURE-v1), and RRF(best BM25, best dense) where "best" is chosen on dev.

Usage: CUDA_VISIBLE_DEVICES=0 python scripts/eval_findartbench_retrieval.py
Outputs: outputs/proposal/splits.json, retrieval_results.json, retrieval_failures.json
"""
from __future__ import annotations

import argparse
import hashlib
import json
import random
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

from finsight.bm25 import BM25Retriever
from finsight.chunking import chunk_text
from finsight.metrics import evaluate

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs" / "proposal"
KS = (1, 5, 10)

DENSE = {
    "mE5-base": ("intfloat/multilingual-e5-base", "query: ", "passage: "),
    "bge-m3": ("BAAI/bge-m3", "", ""),
    "KURE-v1": ("nlpai-lab/KURE-v1", "", ""),
}


def ctx_id(text: str) -> str:
    return "c" + hashlib.md5(text.encode()).hexdigest()[:10]


def load(path: Path) -> pd.DataFrame:
    df = pd.read_json(path, lines=True, dtype={"id": str, "doc_id": str})
    df = df[df.doc_type == "사업보고서"].copy()
    df["ctx_id"] = df.context.map(ctx_id)
    return df


def make_splits(df: pd.DataFrame, seed: int, ratios=(0.55, 0.15, 0.30)) -> dict:
    """Union-find over doc_id <-> ctx_id, then per-company random split of groups."""
    parent: dict[str, str] = {}

    def find(x):
        parent.setdefault(x, x)
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for d, c in zip(df.doc_id, df.ctx_id):
        parent[find("d" + d)] = find(c)
    df = df.assign(group=[find(c) for c in df.ctx_id])
    rng = random.Random(seed)
    assign: dict[str, str] = {}
    for _, sub in df.groupby("company"):
        groups = sorted(sub.group.unique())
        rng.shuffle(groups)
        n = len(groups)
        n_test = max(1, round(n * ratios[2]))
        n_dev = max(1, round(n * ratios[1]))
        for i, g in enumerate(groups):
            assign[g] = "test" if i < n_test else "dev" if i < n_test + n_dev else "train"
    df["split"] = df.group.map(assign)
    out = {"seed": seed, "ratios": dict(zip(["train", "dev", "test"], ratios)), "unit": "doc_id-context group"}
    for s in ("train", "dev", "test"):
        sub = df[df.split == s]
        out[s] = {"n_questions": len(sub), "n_contexts": sub.ctx_id.nunique(), "n_doc_ids": sub.doc_id.nunique(),
                  "ctx_ids": sorted(sub.ctx_id.unique()), "doc_ids": sorted(sub.doc_id.unique()),
                  "question_ids": sorted(sub.id.tolist())}
    for a in ("train", "dev", "test"):
        for b in ("train", "dev", "test"):
            if a < b:
                assert not set(out[a]["ctx_ids"]) & set(out[b]["ctx_ids"])
                assert not set(out[a]["doc_ids"]) & set(out[b]["doc_ids"])
    return df, out


def rank_contexts(chunk_scores: np.ndarray, chunk_ctx: list[str], k: int = 10) -> list[str]:
    """MaxP: order chunks by score, keep the first occurrence of each context."""
    ranked, seen = [], set()
    for i in np.argsort(-chunk_scores):
        c = chunk_ctx[i]
        if c not in seen:
            seen.add(c)
            ranked.append(c)
            if len(ranked) == k:
                break
    return ranked


def rrf(rankings: list[list[str]], k: int = 60, top: int = 10) -> list[str]:
    s: dict[str, float] = defaultdict(float)
    for r in rankings:
        for i, d in enumerate(r):
            s[d] += 1 / (k + i + 1)
    return sorted(s, key=lambda d: -s[d])[:top]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=str(ROOT / "data/raw/findartbench/data.jsonl"))
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--models", nargs="*", default=list(DENSE))
    ap.add_argument("--extra-model", nargs=2, metavar=("NAME", "PATH"), action="append", default=[],
                    help="additional dense model (e.g. fine-tuned checkpoint), no prefixes")
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--tag", default="")
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)

    df, splits = make_splits(load(Path(args.data)), args.seed)
    (OUT / "splits.json").write_text(json.dumps(splits, ensure_ascii=False, indent=1))
    print({s: {k: v for k, v in splits[s].items() if k.startswith("n_")} for s in ("train", "dev", "test")})

    ctx = df.drop_duplicates("ctx_id").set_index("ctx_id")
    ctx_ids = list(ctx.index)
    eval_q = {s: df[df.split == s] for s in ("dev", "test")}

    from transformers import AutoTokenizer
    kure_tok = AutoTokenizer.from_pretrained("nlpai-lab/KURE-v1")

    def ntok(s: str) -> int:
        return len(kure_tok(s, add_special_tokens=False)["input_ids"])

    # corpus chunks (one chunking for all methods, KURE tokenizer budget)
    chunk_text_l, chunk_ctx = [], []
    ctx_len = {}
    for c in ctx_ids:
        t = ctx.loc[c, "context"]
        ctx_len[c] = ntok(t) + 2
        for piece in chunk_text(t, ntok, 512, 50):
            chunk_text_l.append(piece)
            chunk_ctx.append(c)
    lens = np.array(list(ctx_len.values()))
    info = {"n_contexts": len(ctx_ids), "n_chunks": len(chunk_text_l),
            "context_tokens": {"mean": round(float(lens.mean()), 1), "p50": int(np.median(lens)),
                               "p90": int(np.percentile(lens, 90)), "max": int(lens.max())},
            "contexts_over_512": round(float((lens > 512).mean()), 4),
            "tokens_lost_if_truncated_512": round(float(np.clip(lens - 512, 0, None).sum() / lens.sum()), 4),
            "test_questions_gold_over_512": round(float(np.mean([ctx_len[c] > 512 for c in eval_q["test"].ctx_id])), 4)}
    print(info)

    results: dict[str, dict] = {}
    rankings: dict[str, dict[str, list[list[str]]]] = {}

    def record(name: str, ranks: dict[str, list[list[str]]]):
        rankings[name] = ranks
        results[name] = {s: evaluate([(r, {g}) for r, g in zip(ranks[s], eval_q[s].ctx_id)], ks=KS)
                         for s in ("dev", "test")}
        print(name, {s: {k: round(v, 3) for k, v in results[name][s].items()} for s in ("dev", "test")}, flush=True)

    # BM25 over chunks (MaxP)
    for tok in ("whitespace", "kiwi_all", "kiwi_content"):
        bm = BM25Retriever(list(range(len(chunk_text_l))), chunk_text_l, tokenizer=tok)
        ranks = {}
        for s in ("dev", "test"):
            ranks[s] = [rank_contexts(bm.bm25.get_scores(bm.tok(q)), chunk_ctx) for q in eval_q[s].question]
        record(f"BM25 {tok}", ranks)

    # dense
    import torch
    from sentence_transformers import SentenceTransformer
    models = {m: DENSE[m] for m in args.models}
    models.update({n: (p, "", "") for n, p in args.extra_model})
    gpu_mem = {}
    for name, (path, qp, pp) in models.items():
        model = SentenceTransformer(path, device="cuda")
        model.max_seq_length = 512
        torch.cuda.reset_peak_memory_stats()
        enc = lambda xs: model.encode(xs, batch_size=args.batch, normalize_embeddings=True,
                                      convert_to_numpy=True, show_progress_bar=False)
        E_chunk = enc([pp + t for t in chunk_text_l])
        E_full = enc([pp + ctx.loc[c, "context"] for c in ctx_ids])  # truncated at 512 by the model
        ranks_maxp, ranks_trunc = {}, {}
        for s in ("dev", "test"):
            Q = enc([qp + q for q in eval_q[s].question])
            ranks_maxp[s] = [rank_contexts(row, chunk_ctx) for row in Q @ E_chunk.T]
            ranks_trunc[s] = [[ctx_ids[i] for i in np.argsort(-row)[:10]] for row in Q @ E_full.T]
        gpu_mem[name] = round(torch.cuda.max_memory_allocated() / 2**30, 2)
        record(f"{name} (chunks, MaxP)", ranks_maxp)
        record(f"{name} (trunc512)", ranks_trunc)
        del model
        torch.cuda.empty_cache()

    # hybrid: pick best BM25 and best dense by dev MRR@10 (never by test)
    def best(prefix_filter):
        cands = [n for n in results if prefix_filter(n)]
        return max(cands, key=lambda n: results[n]["dev"]["MRR@10"])
    b_bm = best(lambda n: n.startswith("BM25"))
    b_de = best(lambda n: not n.startswith(("BM25", "RRF")))
    record(f"RRF({b_bm} + {b_de})",
           {s: [rrf([a, b]) for a, b in zip(rankings[b_bm][s], rankings[b_de][s])] for s in ("dev", "test")})

    # failures of the best system on test (by dev MRR) for error-analysis notes
    best_sys = max(results, key=lambda n: results[n]["dev"]["MRR@10"])
    fails = []
    tq = eval_q["test"].reset_index(drop=True)
    for i, r in enumerate(rankings[best_sys]["test"]):
        g = tq.ctx_id[i]
        if r[0] != g:
            fails.append({"id": tq.id[i], "company": tq.company[i], "question": tq.question[i],
                          "gold_rank": (r.index(g) + 1) if g in r else None,
                          "gold_head": ctx.loc[g, "context"][:300], "top1_company": ctx.loc[r[0], "company"],
                          "top1_head": ctx.loc[r[0], "context"][:300]})
    random.Random(args.seed).shuffle(fails)

    tag = f"_{args.tag}" if args.tag else ""
    (OUT / f"retrieval_results{tag}.json").write_text(json.dumps(
        {"info": info, "splits": {s: {k: v for k, v in splits[s].items() if k.startswith("n_")}
                                  for s in ("train", "dev", "test")},
         "gpu_peak_mem_gb": gpu_mem, "best_system_by_dev": best_sys, "results": results},
        ensure_ascii=False, indent=1))
    (OUT / f"retrieval_failures{tag}.json").write_text(json.dumps(
        {"system": best_sys, "n_fail_test": len(fails), "n_test": len(tq), "sample": fails[:30]},
        ensure_ascii=False, indent=1))
    table = pd.DataFrame({n: r["test"] for n, r in results.items()}).T.round(3)
    print(table.to_string())


if __name__ == "__main__":
    main()

"""S1 evaluation: company-split fine-tuned KURE models under the P4 conditions.

Questions, gold chunks, splits and the RRF / R1-filter rules are exactly those of eval_query_filter.py
(BM25 Kiwi-content scores are read from its cache outputs/proposal/r1_bm25_scores.npz, same question order).
Models: public KURE-v1, T6 smoke (trained on FinDartBench train = same 10 companies), S1 (a) in-batch only,
S1 (b) + other-year hard negative. For each: dense alone, RRF(BM25 + dense), RRF + R1 metadata filter
(the variant chosen on dev in R1: extracted company/year, y and y+1 window, year-only -> year filter).
Cluster bootstrap (gold contexts, 1,000, seed 42) and the year-confusion rate (top-1 is the same company and
same section path as a gold chunk but another fiscal year; P4/Q3 definition).

Usage:
  CUDA_VISIBLE_DEVICES=2 python scripts/eval_s1.py --encode outputs/proposal/kure-s1-a emb_s1_a.npy   # cache
  CUDA_VISIBLE_DEVICES=2 python scripts/eval_s1.py                                                   # evaluate
  CUDA_VISIBLE_DEVICES=2 python scripts/eval_s1.py --s2                                              # S2 curve (dev)
Outputs: outputs/proposal/s1_results.json, s2_results.json
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from eval_corpus_a_diagnostics import cluster_bootstrap, hits, metrics, rr  # noqa: E402
from eval_corpus_a_retrieval import FDB_TO_DART  # noqa: E402
from eval_findartbench_retrieval import load, rrf  # noqa: E402

from finsight.query_meta import extract

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs" / "proposal"
RANGE = {2022, 2023, 2024}
MODELS = [("KURE-v1 공개", "nlpai-lab/KURE-v1", "emb_kure_base.npy"),
          ("스모크 (같은 10개사 학습)", str(OUT / "kure-v1-findart-smoke"), "emb_kure_ft.npy"),
          ("S1 (a) in-batch", str(OUT / "kure-s1-a"), "emb_s1_a.npy"),
          ("S1 (b) + 다른 연도 hard neg", str(OUT / "kure-s1-b"), "emb_s1_b.npy")]


def load_corpus():
    c = pd.read_json(ROOT / "data/processed/corpus_a.jsonl", lines=True, dtype={"rcept_no": str, "chunk_id": str})
    assert not c.chunk_id.duplicated().any()
    return c


def encode_corpus(model_dir: str, cache: str) -> None:
    from sentence_transformers import SentenceTransformer
    c = load_corpus()
    m = SentenceTransformer(model_dir, device="cuda")
    m.max_seq_length = 512
    E = m.encode(c.text.tolist(), batch_size=64, normalize_embeddings=True, convert_to_numpy=True,
                 show_progress_bar=False)
    np.save(OUT / cache, E.astype(np.float16))
    print("saved", cache, E.shape)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--encode", nargs=2, metavar=("MODEL_DIR", "CACHE"))
    ap.add_argument("--s2", action="store_true", help="pair-count curve for setting (b): dev only, test is not computed")
    args = ap.parse_args()
    if args.encode:
        encode_corpus(*args.encode)
        return

    corpus = load_corpus()
    ids = corpus.chunk_id.to_numpy()
    corp_arr, year_arr = corpus.corp_name.to_numpy(), corpus.fiscal_year.to_numpy()
    sec_arr = corpus.section_path.to_numpy()
    idx_of = {c: i for i, c in enumerate(ids)}
    mapping = json.loads((OUT / "corpus_a_mapping.json").read_text())["contexts"]
    fdb = load(ROOT / "data/raw/findartbench/data.jsonl")
    splits = json.loads((OUT / "splits.json").read_text())
    split_of = {qid: s for s in ("train", "dev", "test") for qid in splits[s]["question_ids"]}
    fdb["split"] = fdb.id.map(split_of)
    fdb["corp"] = fdb.company.map(lambda c: FDB_TO_DART.get(c, c))
    fdb = fdb[fdb.ctx_id.map(lambda c: mapping[c]["mapped"])]
    splits_used = ("dev",) if args.s2 else ("dev", "test")
    Q = {s: fdb[fdb.split == s].reset_index(drop=True) for s in splits_used}
    gold = {s: [set(mapping[c]["gold"]) for c in Q[s].ctx_id] for s in Q}
    z = np.load(OUT / "r1_bm25_scores.npz")
    bm_sc = {s: z[s].astype(np.float32) for s in Q}
    assert all(bm_sc[s].shape == (len(Q[s]), len(corpus)) for s in Q), "BM25 cache does not match question order"
    meta = {s: [extract(q) for q in Q[s].question] for s in Q}

    def filt(s, i):  # R1 chosen variant
        md = meta[s][i]
        comps, years = set(md["companies"]), set(md["fiscal_years"]) & RANGE
        years = years | {y + 1 for y in years}
        if comps and years:
            return np.isin(corp_arr, list(comps)) & np.isin(year_arr, list(years))
        if comps:
            return np.isin(corp_arr, list(comps))
        if years:
            return np.isin(year_arr, list(years))
        return None

    masks = {s: [filt(s, i) for i in range(len(Q[s]))] for s in Q}

    def topk(v, mask):
        v = v if mask is None else np.where(mask, v, -np.inf)
        t = np.argpartition(-v, 10)[:10]
        t = t[np.argsort(-v[t])]
        return [ids[i] for i in t if np.isfinite(v[i])]

    bm_rank = {cond: {s: [topk(bm_sc[s][i], None if cond == "none" else masks[s][i]) for i in range(len(Q[s]))]
                      for s in Q} for cond in ("none", "filter")}

    from sentence_transformers import SentenceTransformer
    import torch
    models = MODELS
    if args.s2:
        models = [MODELS[0]] + [(f"S1 (b) {n}쌍", str(OUT / f"kure-s1-b-n{n}"), f"emb_s1_b_n{n}.npy")
                                for n in (1000, 2500)] + [("S1 (b) 5000쌍", MODELS[3][1], MODELS[3][2])]
    rankings, results = {}, {}
    for name, path, cache in models:
        E = np.load(OUT / cache).astype(np.float32)
        m = SentenceTransformer(path, device="cuda")
        m.max_seq_length = 512
        S = {s: m.encode(Q[s].question.tolist(), batch_size=64, normalize_embeddings=True) @ E.T for s in Q}
        del m
        torch.cuda.empty_cache()
        dense = {s: [topk(S[s][i], None) for i in range(len(Q[s]))] for s in Q}
        dense_f = {s: [topk(S[s][i], masks[s][i]) for i in range(len(Q[s]))] for s in Q}
        for label, rk in [("dense", dense),
                          ("RRF", {s: [rrf([a, b]) for a, b in zip(bm_rank["none"][s], dense[s])] for s in Q}),
                          ("RRF + 메타데이터 필터",
                           {s: [rrf([a, b]) for a, b in zip(bm_rank["filter"][s], dense_f[s])] for s in Q})]:
            rankings[(name, label)] = rk
            results[f"{name} | {label}"] = {s: metrics(rk[s], gold[s]) for s in Q}
        print(name, {k: {m_: round(v, 3) for m_, v in results[f'{name} | {k2}'][splits_used[-1]].items()}
                     for k, k2 in [("dense", "dense"), ("RRF", "RRF")]}, flush=True)
    if args.s2:
        (OUT / "s2_results.json").write_text(json.dumps({"n_dev": len(Q["dev"]), "results": results},
                                                        ensure_ascii=False, indent=1))
        print(pd.DataFrame({k: v["dev"] for k, v in results.items()}).T.round(3).to_string())
        return

    t = "test"
    clusters = Q[t].ctx_id.to_numpy()

    def compare(a, b, label="dense"):
        ra, rb = rankings[(a, label)][t], rankings[(b, label)][t]
        out = {"a": a, "b": b, "ranking": label}
        for k in (1, 5):
            ha, hb = hits(ra, gold[t], k), hits(rb, gold[t], k)
            out[f"R@{k}"] = {"a": float(ha.mean()), "b": float(hb.mean()), "diff": float(hb.mean() - ha.mean()),
                             "cluster_bootstrap": cluster_bootstrap(hb.astype(float) - ha.astype(float), clusters)}
        d = rr(rb, gold[t]) - rr(ra, gold[t])
        out["MRR@10"] = {"diff": float(d.mean()), "cluster_bootstrap": cluster_bootstrap(d, clusters)}
        return out

    pub, smoke, a, b = (n for n, _, _ in MODELS)
    sig = [compare(pub, a), compare(pub, b), compare(a, b), compare(pub, smoke), compare(a, b, "RRF")]

    def year_confusion(rk):
        n = 0
        for r, g in zip(rk, gold[t]):
            top = idx_of[r[0]]
            if r[0] in g:
                continue
            gi = [idx_of[x] for x in g]
            if corp_arr[top] == corp_arr[gi[0]] and year_arr[top] != year_arr[gi[0]] and \
                    sec_arr[top] in {sec_arr[x] for x in gi}:
                n += 1
        return n / len(rk)

    confusion = {f"{n} | {lab}": round(year_confusion(rankings[(n, lab)][t]), 4)
                 for n, _, _ in MODELS for lab in ("dense", "RRF", "RRF + 메타데이터 필터")}
    # breakdown (analysis of the same run, no selection): questions with / without an extracted year
    has_year = np.array([bool(set(md["fiscal_years"]) & RANGE) for md in meta[t]])
    breakdown = {}
    for n, _, _ in MODELS:
        rk = rankings[(n, "dense")][t]
        for lab, sel in (("연도 있음", has_year), ("연도 없음", ~has_year)):
            ii = np.where(sel)[0]
            sub = [rk[i] for i in ii]
            g = [gold[t][i] for i in ii]
            gold_bak = gold[t]
            gold[t] = g
            yc = year_confusion(sub)
            gold[t] = gold_bak
            breakdown[f"{n} | dense | {lab}"] = {**metrics(sub, g), "n": int(sel.sum()), "year_confusion": round(yc, 4)}
    out = {"n": {s: len(Q[s]) for s in Q}, "results": results, "significance_test": sig,
           "year_confusion_top1_test": confusion, "test_breakdown_by_year_in_question": breakdown}
    (OUT / "s1_results.json").write_text(json.dumps(out, ensure_ascii=False, indent=1))
    print(pd.DataFrame({k: v["test"] for k, v in results.items()}).T.round(3).to_string())
    print(json.dumps(confusion, ensure_ascii=False, indent=1))
    for k, v in breakdown.items():
        print(k, {m_: (round(x, 3) if isinstance(x, float) else x) for m_, x in v.items()})
    for c in sig:
        print(c["a"], "->", c["b"], c["ranking"], {k: (round(c[k]["diff"], 4), [round(x, 4) for x in c[k]["cluster_bootstrap"]["ci95"]])
                                                   for k in ("R@1", "R@5", "MRR@10")})


if __name__ == "__main__":
    main()

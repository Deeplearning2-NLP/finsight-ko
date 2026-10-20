"""Diagnostics on the scope-A retrieval benchmark (Q2, Q3, Q4 sample, Q7).

Same questions, gold chunks and splits as eval_corpus_a_retrieval.py (P4; mapping read from
outputs/proposal/corpus_a_mapping.json). Every method scores the full corpus once per query; the
conditions only mask the candidate set:
  none          full corpus (= P4)
  company       chunks of the question's company (FinDartBench `company`)
  company_year  + fiscal year of the report the gold context was mapped to (oracle upper bound;
                a real system can only do this when the question states the year)
Lenient metric: a hit on a near-duplicate of a gold chunk (same company and section path, another
fiscal year, character 5-gram Jaccard >= 0.9) also counts.
Q2: paired McNemar (exact) on hit@1/hit@5 + cluster bootstrap over gold contexts (1,000, seed 42).
Q4: 30 random test failures (gold outside top 5) of RRF/none for manual error analysis.
Q7: TF-IDF (Kiwi content morphemes) cosine as a classic embedding-style baseline.
RRF pair and k=60 are those chosen on dev in P4; nothing here is tuned on test.

Usage: CUDA_VISIBLE_DEVICES=0 python scripts/eval_corpus_a_diagnostics.py
Outputs: outputs/proposal/diag_results.json, diag_stats.json, diag_rankings.json, error_analysis_30.md
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from eval_corpus_a_retrieval import FDB_TO_DART, grams  # noqa: E402
from eval_findartbench_retrieval import load, rrf  # noqa: E402

from finsight.tokenize import TOKENIZERS

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs" / "proposal"
KS = (1, 5, 10)
CONDS = ("none", "company", "company_year")
DUP_JACCARD = 0.9
SEED = 42


def metrics(ranked, gold) -> dict:
    out = {f"R@{k}": float(np.mean([bool(set(r[:k]) & g) for r, g in zip(ranked, gold)])) for k in KS}
    out["MRR@10"] = float(np.mean([next((1 / (i + 1) for i, d in enumerate(r[:10]) if d in g), 0.0)
                                   for r, g in zip(ranked, gold)]))
    return out


def hits(ranked, gold, k) -> np.ndarray:
    return np.array([bool(set(r[:k]) & g) for r, g in zip(ranked, gold)])


def rr(ranked, gold) -> np.ndarray:
    return np.array([next((1 / (i + 1) for i, d in enumerate(r[:10]) if d in g), 0.0) for r, g in zip(ranked, gold)])


def mcnemar_exact(a: np.ndarray, b: np.ndarray) -> dict:
    from scipy.stats import binomtest
    b_only, a_only = int((~a & b).sum()), int((a & ~b).sum())
    p = binomtest(b_only, b_only + a_only, 0.5).pvalue if b_only + a_only else 1.0
    return {"a_only": a_only, "b_only": b_only, "p": float(p)}


def cluster_bootstrap(x: np.ndarray, clusters: np.ndarray, n: int = 1000, seed: int = SEED) -> dict:
    """95% CI of mean(x) resampling clusters (gold contexts) with replacement."""
    rng = np.random.default_rng(seed)
    ids = np.unique(clusters)
    by = {c: np.where(clusters == c)[0] for c in ids}
    sums = np.array([x[by[c]].sum() for c in ids])
    cnts = np.array([len(by[c]) for c in ids])
    stats = []
    for _ in range(n):
        pick = rng.integers(0, len(ids), len(ids))
        stats.append(sums[pick].sum() / cnts[pick].sum())
    lo, hi = np.percentile(stats, [2.5, 97.5])
    return {"mean": float(x.mean()), "ci95": [float(lo), float(hi)], "excludes_0": bool(lo > 0 or hi < 0)}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ft-model", default=str(OUT / "kure-v1-findart-smoke"))
    ap.add_argument("--batch", type=int, default=64)
    args = ap.parse_args()

    corpus = pd.read_json(ROOT / "data/processed/corpus_a.jsonl", lines=True,
                          dtype={"rcept_no": str, "chunk_id": str})
    assert not corpus.chunk_id.duplicated().any()
    ids = corpus.chunk_id.to_numpy()
    texts = corpus.text.tolist()
    corp_arr = corpus.corp_name.to_numpy()
    year_arr = corpus.fiscal_year.to_numpy()
    idx_of = {c: i for i, c in enumerate(ids)}

    mapping = json.loads((OUT / "corpus_a_mapping.json").read_text())["contexts"]
    fdb = load(ROOT / "data/raw/findartbench/data.jsonl")
    splits = json.loads((OUT / "splits.json").read_text())
    split_of = {qid: s for s in ("train", "dev", "test") for qid in splits[s]["question_ids"]}
    fdb["split"] = fdb.id.map(split_of)
    fdb["corp"] = fdb.company.map(lambda c: FDB_TO_DART.get(c, c))
    fdb = fdb[fdb.ctx_id.map(lambda c: mapping[c]["mapped"])]
    Q = {s: fdb[fdb.split == s].reset_index(drop=True) for s in ("dev", "test")}
    gold = {s: [set(mapping[c]["gold"]) for c in Q[s].ctx_id] for s in Q}
    oracle_year = {s: np.array([mapping[c]["fiscal_year"] for c in Q[s].ctx_id]) for s in Q}
    print({s: len(Q[s]) for s in Q}, flush=True)

    # near-duplicates of gold chunks in other fiscal years (lenient metric)
    gram_cache: dict[int, set] = {}

    def g(i):
        if i not in gram_cache:
            gram_cache[i] = grams(texts[i])
        return gram_cache[i]

    groups = defaultdict(list)
    for i, (c, sp) in enumerate(zip(corp_arr, corpus.section_path)):
        groups[(c, sp)].append(i)
    dup_of: dict[str, set] = {}
    all_gold = {cid for s in gold for gs in gold[s] for cid in gs}
    for cid in all_gold:
        i = idx_of[cid]
        a = g(i)
        dups = set()
        for j in groups[(corp_arr[i], corpus.section_path[i])]:
            if year_arr[j] == year_arr[i] or not a:
                continue
            b = g(j)
            if b and len(a & b) / len(a | b) >= DUP_JACCARD:
                dups.add(ids[j])
        dup_of[cid] = dups
    lenient = {s: [gs | set().union(*(dup_of[c] for c in gs)) for gs in gold[s]] for s in gold}
    n_with_dup = sum(1 for c in all_gold if dup_of[c])
    print(f"gold chunks {len(all_gold)}, with near-duplicate in another year: {n_with_dup}", flush=True)

    # ---------------- scoring: every method yields a function query_index -> score vector
    def masks(s, qi):
        corp_ok = corp_arr == Q[s].corp[qi]
        return {"none": None, "company": corp_ok, "company_year": corp_ok & (year_arr == oracle_year[s][qi])}

    rankings: dict[str, dict[str, dict[str, list[list[str]]]]] = {}

    def rank_all(name, score_fn):
        rankings[name] = {c: {} for c in CONDS}
        for s in Q:
            per = {c: [] for c in CONDS}
            for qi in range(len(Q[s])):
                sc = score_fn(s, qi)
                for c, m in masks(s, qi).items():
                    v = sc if m is None else np.where(m, sc, -np.inf)
                    top = np.argpartition(-v, 10)[:10]
                    top = top[np.argsort(-v[top])]
                    per[c].append([ids[i] for i in top if np.isfinite(v[i])])
            for c in CONDS:
                rankings[name][c][s] = per[c]
        print(name, "ranked", flush=True)

    from rank_bm25 import BM25Okapi
    tok_cache = {}
    for tok in ("whitespace", "kiwi_content"):
        toks = [TOKENIZERS[tok](t) for t in texts]
        tok_cache[tok] = toks
        bm = BM25Okapi(toks)
        qtok = {s: [TOKENIZERS[tok](x) for x in Q[s].question] for s in Q}
        rank_all(f"BM25 {tok}", lambda s, qi, bm=bm, qtok=qtok: bm.get_scores(qtok[s][qi]))

    # Q7: TF-IDF over Kiwi content morphemes, cosine (default sklearn settings, not tuned)
    from sklearn.feature_extraction.text import TfidfVectorizer
    tf = TfidfVectorizer(analyzer=lambda x: x, sublinear_tf=False)
    X = tf.fit_transform(tok_cache["kiwi_content"])
    tq = {s: tf.transform([TOKENIZERS["kiwi_content"](x) for x in Q[s].question]) for s in Q}
    tsc = {s: (tq[s] @ X.T).toarray() for s in Q}
    rank_all("TF-IDF kiwi_content", lambda s, qi: tsc[s][qi])

    import torch
    from sentence_transformers import SentenceTransformer
    for name, path, cache in [("KURE-v1", "nlpai-lab/KURE-v1", "emb_kure_base.npy"),
                              ("KURE-v1 스모크 미세조정", args.ft_model, "emb_kure_ft.npy")]:
        m = SentenceTransformer(path, device="cuda")
        m.max_seq_length = 512
        cp = OUT / cache
        if cp.exists():
            E = np.load(cp).astype(np.float32)
        else:
            E = m.encode(texts, batch_size=args.batch, normalize_embeddings=True, convert_to_numpy=True,
                         show_progress_bar=False)
            np.save(cp, E.astype(np.float16))
        S = {s: m.encode(Q[s].question.tolist(), batch_size=args.batch, normalize_embeddings=True) @ E.T for s in Q}
        rank_all(name, lambda s, qi, S=S: S[s][qi])
        del m
        torch.cuda.empty_cache()

    rrf_name = "RRF(BM25 kiwi_content + KURE-v1 스모크 미세조정)"  # pair chosen on dev in P4
    rankings[rrf_name] = {c: {s: [rrf([a, b]) for a, b in zip(rankings["BM25 kiwi_content"][c][s],
                                                              rankings["KURE-v1 스모크 미세조정"][c][s])]
                              for s in Q} for c in CONDS}

    results = {n: {c: {s: {"strict": metrics(rankings[n][c][s], gold[s]),
                           "lenient": metrics(rankings[n][c][s], lenient[s])} for s in Q} for c in CONDS}
               for n in rankings}
    (OUT / "diag_results.json").write_text(json.dumps(
        {"n_questions": {s: len(Q[s]) for s in Q}, "gold_chunks": len(all_gold),
         "gold_chunks_with_near_dup": n_with_dup, "results": results}, ensure_ascii=False, indent=1))
    (OUT / "diag_rankings.json").write_text(json.dumps(
        {n: {c: rankings[n][c] for c in CONDS} for n in rankings}, ensure_ascii=False))
    rows = {(n, c): results[n][c]["test"]["strict"] for n in results for c in CONDS}
    print(pd.DataFrame(rows).T.round(3).to_string(), flush=True)

    # ---------------- Q2: significance on test, condition none
    t = "test"
    clusters = Q[t].ctx_id.to_numpy()
    pairs = [("KURE-v1", "KURE-v1 스모크 미세조정"), ("BM25 kiwi_content", "KURE-v1 스모크 미세조정"),
             ("KURE-v1 스모크 미세조정", rrf_name)]
    stats = {"n_test": len(Q[t]), "n_clusters": int(len(np.unique(clusters))), "pairs": []}
    for a, b in pairs:
        ra, rb = rankings[a]["none"][t], rankings[b]["none"][t]
        entry = {"a": a, "b": b}
        for k in (1, 5):
            ha, hb = hits(ra, gold[t], k), hits(rb, gold[t], k)
            entry[f"R@{k}"] = {"a": float(ha.mean()), "b": float(hb.mean()), "diff": float(hb.mean() - ha.mean()),
                               "mcnemar": mcnemar_exact(ha, hb),
                               "cluster_bootstrap": cluster_bootstrap(hb.astype(float) - ha.astype(float), clusters)}
        da = rr(rb, gold[t]) - rr(ra, gold[t])
        entry["MRR@10"] = {"diff": float(da.mean()), "cluster_bootstrap": cluster_bootstrap(da, clusters)}
        stats["pairs"].append(entry)
    (OUT / "diag_stats.json").write_text(json.dumps(stats, ensure_ascii=False, indent=1))
    print(json.dumps(stats, ensure_ascii=False, indent=1), flush=True)

    # ---------------- Q4: 30 failures (gold outside top 5) of RRF / none on test
    meta = corpus.set_index("chunk_id")
    fails = [i for i, (r, gs) in enumerate(zip(rankings[rrf_name]["none"][t], gold[t])) if not set(r[:5]) & gs]
    rnd = random.Random(SEED)
    sample = rnd.sample(fails, 30)
    lines = [f"# 오류 분석 30건 (RRF, 필터 없음, test)", "",
             f"test {len(Q[t])}문항 중 정답 청크가 top-5 밖인 실패 {len(fails)}건에서 seed {SEED}로 30건 추출.", ""]
    for n, i in enumerate(sample, 1):
        row = Q[t].iloc[i]
        r = rankings[rrf_name]["none"][t][i]
        gs = sorted(gold[t][i])
        gm = meta.loc[gs[0]]
        top = meta.loc[r[0]]
        rank = next((k + 1 for k, d in enumerate(r) if d in gold[t][i]), None)
        len_hit = next((k + 1 for k, d in enumerate(r) if d in lenient[t][i]), None)
        lines += [f"## {n}. [{row.company}] {row.question}", "",
                  f"- qid `{row.id}`, 정답 청크 {len(gs)}개 (FY{gm.fiscal_year}), RRF 순위: {rank or '10위 밖'}"
                  f", 관대한 지표 순위: {len_hit or '10위 밖'}",
                  f"- **정답** `{gs[0]}` {gm.corp_name} FY{gm.fiscal_year} · {gm.section_path}",
                  f"  > {gm.text[:350].replace(chr(10), ' ')}",
                  f"- **1위** `{r[0]}` {top.corp_name} FY{top.fiscal_year} · {top.section_path}",
                  f"  > {top.text[:350].replace(chr(10), ' ')}",
                  f"- 근거 문단 앞부분 (FinDartBench): {row.context[:200].replace(chr(10), ' ')}",
                  "- 유형: ", "- 왜 틀렸나: ", ""]
    (OUT / "error_analysis_30.md").write_text("\n".join(lines))
    print("failures", len(fails), "sampled 30")


if __name__ == "__main__":
    main()

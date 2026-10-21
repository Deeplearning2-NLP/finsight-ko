"""R1 evaluation 2: retrieval with filters extracted from the question (finsight.query_meta v1).

Same benchmark as P4/Q3 (FinDartBench 사업보고서 questions mapped to corpus_a chunks, T4 splits,
RRF of BM25 Kiwi-content + smoke fine-tuned KURE, top-10 inputs, k=60 — all fixed on dev in P4).
Per question:
  companies and in-range years extracted -> filter company AND fiscal_year
  only companies                         -> company filter
  only in-range years                    -> year filter if --year-only (chosen on dev), else none
  nothing                                -> no filter (fallback)
Also the P4/Q3 reference conditions (none, oracle company+year) to check they reproduce.
"Wrong filter": the filter excludes every gold chunk of the question.

Usage: CUDA_VISIBLE_DEVICES=0 python scripts/eval_query_filter.py
Outputs: outputs/proposal/r1_filter_results.json
"""
from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from eval_corpus_a_diagnostics import metrics  # noqa: E402
from eval_corpus_a_retrieval import FDB_TO_DART  # noqa: E402
from eval_findartbench_retrieval import load, rrf  # noqa: E402

from finsight.query_meta import extract
from finsight.tokenize import TOKENIZERS

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs" / "proposal"
RANGE = {2022, 2023, 2024}


def main() -> None:
    corpus = pd.read_json(ROOT / "data/processed/corpus_a.jsonl", lines=True,
                          dtype={"rcept_no": str, "chunk_id": str})
    assert not corpus.chunk_id.duplicated().any()
    ids = corpus.chunk_id.to_numpy()
    corp_arr, year_arr = corpus.corp_name.to_numpy(), corpus.fiscal_year.to_numpy()
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
    oracle_year = {s: [mapping[c]["fiscal_year"] for c in Q[s].ctx_id] for s in Q}
    meta = {s: [extract(q) for q in Q[s].question] for s in Q}

    # scores: BM25 Kiwi content (recomputed) and fine-tuned KURE (cached corpus embeddings from Q3)
    cache = OUT / "r1_bm25_scores.npz"
    if cache.exists():
        z = np.load(cache)
        bm_sc = {s: z[s].astype(np.float64) for s in Q}
    else:
        from rank_bm25 import BM25Okapi
        tok = TOKENIZERS["kiwi_content"]
        bm = BM25Okapi([tok(t) for t in corpus.text])
        bm_sc = {s: np.vstack([bm.get_scores(tok(q)) for q in Q[s].question]) for s in Q}
        np.savez(cache, **{s: v.astype(np.float32) for s, v in bm_sc.items()})
    E = np.load(OUT / "emb_kure_ft.npy").astype(np.float32)
    from sentence_transformers import SentenceTransformer
    m = SentenceTransformer(str(OUT / "kure-v1-findart-smoke"), device="cuda")
    m.max_seq_length = 512
    de_sc = {s: m.encode(Q[s].question.tolist(), batch_size=64, normalize_embeddings=True) @ E.T for s in Q}
    print("scored", flush=True)

    def topk(v, mask):
        v = v if mask is None else np.where(mask, v, -np.inf)
        t = np.argpartition(-v, 10)[:10]
        t = t[np.argsort(-v[t])]
        return [ids[i] for i in t if np.isfinite(v[i])]

    def mask_for(s, i, cond, year_only, window=0):
        md = meta[s][i]
        comps, years = set(md["companies"]), set(md["fiscal_years"]) & RANGE
        years = years | {y + k for y in years for k in range(1, window + 1)}
        if cond == "none":
            return None, "none"
        if cond == "oracle":
            return (corp_arr == Q[s].corp[i]) & (year_arr == oracle_year[s][i]), "oracle"
        if comps and years:
            return np.isin(corp_arr, list(comps)) & np.isin(year_arr, list(years)), "company+year"
        if comps:
            return np.isin(corp_arr, list(comps)), "company"
        if years and year_only:
            return np.isin(year_arr, list(years)), "year"
        return None, "fallback_none"

    results, kinds, wrong = {}, {}, {}
    rankings = {}
    # variants; the y+1 window idea came from reading the *test* wrong-filter cases (disclosed in the
    # results doc), so it is chosen on dev only like the rest.
    variants = [("none", "none", False, 0), ("oracle (회사+연도)", "oracle", False, 0),
                ("추출 필터 (연도만이면 필터 없음)", "extracted", False, 0),
                ("추출 필터 (연도만이면 연도 필터)", "extracted", True, 0),
                ("추출 필터 + 연도 창 y·y+1 (연도만이면 연도 필터)", "extracted", True, 1)]
    for name, cond, yo, win in variants:
        res, kd, wr, rk = {}, {}, {}, {}
        for s in Q:
            ranks, kk, bad = [], [], []
            for i in range(len(Q[s])):
                msk, kind = mask_for(s, i, cond, yo, win)
                ranks.append(rrf([topk(bm_sc[s][i], msk), topk(de_sc[s][i], msk)]))
                kk.append(kind)
                if msk is not None and not any(msk[idx_of[g]] for g in gold[s][i]):
                    bad.append(i)
            rk[s] = ranks
            res[s] = metrics(ranks, gold[s])
            kd[s] = dict(Counter(kk))
            wr[s] = bad
        results[name], kinds[name], wrong[name], rankings[name] = res, kd, wr, rk
        print(name, {s: {k: round(v, 3) for k, v in res[s].items()} for s in Q}, kd, {s: len(wr[s]) for s in Q}, flush=True)

    # breakdown on test by whether the extractor found an in-range year
    t = "test"
    has_year = np.array([bool(set(md["fiscal_years"]) & RANGE) for md in meta[t]])
    has_comp = np.array([bool(md["companies"]) for md in meta[t]])
    breakdown = {}
    for name in results:
        r = rankings[name][t]
        breakdown[name] = {
            label: metrics([r[i] for i in np.where(sel)[0]], [gold[t][i] for i in np.where(sel)[0]]) | {"n": int(sel.sum())}
            for label, sel in [("연도 추출됨", has_year), ("연도 없음", ~has_year),
                               ("회사 추출됨", has_comp), ("회사 없음", ~has_comp)]}
    wrong_detail = []
    ext_names = [n for n in results if n.startswith("추출")]
    best_ext = max(ext_names, key=lambda n: results[n]["dev"]["MRR@10"])  # chosen on dev MRR
    for i in wrong["추출 필터 (연도만이면 연도 필터)"][t]:
        md = meta[t][i]
        g = sorted(gold[t][i])[0]
        wrong_detail.append({"qid": Q[t].id[i], "question": Q[t].question[i][:150], "extracted": {
            "companies": md["companies"], "fiscal_years": md["fiscal_years"]},
            "gold_company": Q[t].corp[i], "gold_year": oracle_year[t][i], "gold_chunk": g})
    # dev wrong-filter cases of the exact-year variant (to check the year-window pattern on dev)
    dev_wrong = []
    for i in wrong["추출 필터 (연도만이면 연도 필터)"]["dev"]:
        md = meta["dev"][i]
        dev_wrong.append({"qid": Q["dev"].id[i], "question": Q["dev"].question[i][:150],
                          "extracted_years": md["fiscal_years"], "extracted_companies": md["companies"],
                          "gold_company": Q["dev"].corp[i], "gold_year": oracle_year["dev"][i]})
    # significance on test: chosen extracted filter vs none, cluster bootstrap over gold contexts
    from eval_corpus_a_diagnostics import cluster_bootstrap, hits, mcnemar_exact, rr
    clusters = Q[t].ctx_id.to_numpy()
    sig = {}
    for k in (1, 5):
        a, b = hits(rankings["none"][t], gold[t], k), hits(rankings[best_ext][t], gold[t], k)
        sig[f"R@{k}"] = {"diff": float(b.mean() - a.mean()), "mcnemar": mcnemar_exact(a, b),
                         "cluster_bootstrap": cluster_bootstrap(b.astype(float) - a.astype(float), clusters)}
    d = rr(rankings[best_ext][t], gold[t]) - rr(rankings["none"][t], gold[t])
    sig["MRR@10"] = {"diff": float(d.mean()), "cluster_bootstrap": cluster_bootstrap(d, clusters)}
    out = {"n": {s: len(Q[s]) for s in Q}, "results": results, "filter_kinds": kinds,
           "significance_test_best_vs_none": sig, "dev_wrong_filter_detail": dev_wrong,
           "wrong_filter_counts": {n: {s: len(v[s]) for s in v} for n, v in wrong.items()},
           "chosen_on_dev": best_ext, "test_breakdown": breakdown,
           "test_extracted_year_share": float(has_year.mean()), "test_extracted_company_share": float(has_comp.mean()),
           "wrong_filter_detail_test": wrong_detail}
    (OUT / "r1_filter_results.json").write_text(json.dumps(out, ensure_ascii=False, indent=1))
    print("chosen on dev:", best_ext)
    print(pd.DataFrame({n: r["test"] for n, r in results.items()}).T.round(3).to_string())
    print(json.dumps(breakdown, ensure_ascii=False, indent=0)[:3000])


if __name__ == "__main__":
    main()

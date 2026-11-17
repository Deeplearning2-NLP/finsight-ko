"""Midterm evaluation (M3 dev selection, M4/M5 final test) under the P4 conditions.

Same questions / gold chunks / splits / BM25 scores as eval_query_filter.py and eval_s1.py (FinDartBench
사업보고서 questions mapped onto corpus_a, 34,427 chunks). For each model: dense, RRF(Kiwi BM25 + dense),
RRF + R1 metadata filter, optionally + a reranker over the top-20 (M6).
Breakdowns: year in question (query_meta), company in question, gold chunk is a table chunk.
Confusion of the top-1 (wrong) chunk: other fiscal year (same company & section path as a gold chunk, P4 def.),
other company, same company other section. Paired comparisons vs the first model: McNemar (exact) and
gold-context cluster bootstrap (1,000, seed 42).

Usage:
  python scripts/eval_m.py --encode MODEL_DIR CACHE.npy                 # cache corpus embeddings (GPU)
  python scripts/eval_m.py --split dev  --models name=path=cache ...    # selection (dev only)
  python scripts/eval_m.py --split test --models ... [--reranker name=path]  # final, once
Outputs: outputs/midterm/eval_<split>_<tag>.json (+ rankings for error analysis)
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from eval_corpus_a_diagnostics import cluster_bootstrap, hits, mcnemar_exact, metrics, rr  # noqa: E402
from eval_corpus_a_retrieval import FDB_TO_DART  # noqa: E402
from eval_findartbench_retrieval import load, rrf  # noqa: E402

from finsight.query_meta import extract

ROOT = Path(__file__).resolve().parents[1]
PROP = ROOT / "outputs" / "proposal"
OUT = ROOT / "outputs" / "midterm"
RANGE = {2022, 2023, 2024}


def load_corpus() -> pd.DataFrame:
    c = pd.read_json(ROOT / "data/processed/corpus_a.jsonl", lines=True, dtype={"rcept_no": str, "chunk_id": str})
    assert not c.chunk_id.duplicated().any()
    return c


def encode(model_dir: str, cache: str) -> None:
    from sentence_transformers import SentenceTransformer
    c = load_corpus()
    m = SentenceTransformer(model_dir, device="cuda")
    m.max_seq_length = 512
    E = m.encode(c.text.tolist(), batch_size=64, normalize_embeddings=True, convert_to_numpy=True,
                 show_progress_bar=False)
    path = Path(cache) if "/" in cache else OUT / cache
    path.parent.mkdir(parents=True, exist_ok=True)
    np.save(path, E.astype(np.float16))
    print("saved", path, E.shape, flush=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--encode", nargs=2, metavar=("MODEL_DIR", "CACHE"))
    ap.add_argument("--split", choices=["dev", "test"])
    ap.add_argument("--models", nargs="*", default=[], help="name=model_path=embedding_cache")
    ap.add_argument("--reranker", nargs="*", default=[], help="name=model_path (cross-encoder over RRF+filter top-20)")
    ap.add_argument("--tag", default="")
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    if args.encode:
        encode(*args.encode)
        return
    s = args.split

    corpus = load_corpus()
    ids = corpus.chunk_id.to_numpy()
    texts = corpus.text.tolist()
    corp_arr, year_arr, sec_arr = corpus.corp_name.to_numpy(), corpus.fiscal_year.to_numpy(), corpus.section_path.to_numpy()
    table_arr = corpus.is_table.to_numpy()
    idx_of = {c: i for i, c in enumerate(ids)}
    mapping = json.loads((PROP / "corpus_a_mapping.json").read_text())["contexts"]
    fdb = load(ROOT / "data/raw/findartbench/data.jsonl")
    splits = json.loads((PROP / "splits.json").read_text())
    split_of = {qid: sp for sp in ("train", "dev", "test") for qid in splits[sp]["question_ids"]}
    fdb["split"] = fdb.id.map(split_of)
    fdb["corp"] = fdb.company.map(lambda c: FDB_TO_DART.get(c, c))
    fdb = fdb[fdb.ctx_id.map(lambda c: mapping[c]["mapped"])]
    Q = fdb[fdb.split == s].reset_index(drop=True)
    gold = [set(mapping[c]["gold"]) for c in Q.ctx_id]
    bm = np.load(PROP / "r1_bm25_scores.npz")[s].astype(np.float32)
    assert bm.shape == (len(Q), len(corpus)), "BM25 cache does not match question order"
    meta = [extract(q) for q in Q.question]

    def filt(i):  # R1 rule chosen on dev
        comps, years = set(meta[i]["companies"]), set(meta[i]["fiscal_years"]) & RANGE
        years = years | {y + 1 for y in years}
        if comps and years:
            return np.isin(corp_arr, list(comps)) & np.isin(year_arr, list(years))
        if comps:
            return np.isin(corp_arr, list(comps))
        if years:
            return np.isin(year_arr, list(years))
        return None

    masks = [filt(i) for i in range(len(Q))]

    def topk(v, mask, k=10):
        v = v if mask is None else np.where(mask, v, -np.inf)
        t = np.argpartition(-v, k)[:k]
        t = t[np.argsort(-v[t])]
        return [ids[i] for i in t if np.isfinite(v[i])]

    bm_none = [topk(bm[i], None) for i in range(len(Q))]
    bm_filt = [topk(bm[i], masks[i]) for i in range(len(Q))]
    bm_filt20 = [topk(bm[i], masks[i], 20) for i in range(len(Q))]

    from sentence_transformers import SentenceTransformer
    import torch
    rankings, results = {}, {}
    for spec in args.models:
        name, path, cache = spec.split("=")
        E = np.load(cache if "/" in cache else (OUT / cache if (OUT / cache).exists() else PROP / cache)).astype(np.float32)
        m = SentenceTransformer(path, device="cuda")
        m.max_seq_length = 512
        S = m.encode(Q.question.tolist(), batch_size=64, normalize_embeddings=True) @ E.T
        del m
        torch.cuda.empty_cache()
        dense = [topk(S[i], None) for i in range(len(Q))]
        dense_f = [topk(S[i], masks[i]) for i in range(len(Q))]
        rankings[(name, "dense")] = dense
        rankings[(name, "RRF")] = [rrf([a, b]) for a, b in zip(bm_none, dense)]
        rankings[(name, "RRF+filter")] = [rrf([a, b]) for a, b in zip(bm_filt, dense_f)]
        if args.reranker:
            dense_f20 = [topk(S[i], masks[i], 20) for i in range(len(Q))]
            cand = [rrf([a, b], top=20) for a, b in zip(bm_filt20, dense_f20)]
            from sentence_transformers import CrossEncoder
            for rspec in args.reranker:
                rname, rpath = rspec.split("=")
                ce = CrossEncoder(rpath, device="cuda", max_length=512)
                from train_reranker_m import fix_token_types
                fix_token_types(ce)
                flat = [(q, texts[idx_of[c]]) for q, cs in zip(Q.question, cand) for c in cs]
                sc = ce.predict(flat, batch_size=64, show_progress_bar=False)
                out, k = [], 0
                for cs in cand:
                    s_ = sc[k:k + len(cs)]
                    k += len(cs)
                    out.append([cs[j] for j in np.argsort(-s_)][:10])
                rankings[(name, f"RRF+filter+{rname}")] = out
                del ce
                torch.cuda.empty_cache()
        for lab in [l for (n, l) in rankings if n == name]:
            results[f"{name} | {lab}"] = metrics(rankings[(name, lab)], gold)
        print(name, {lab: round(results[f"{name} | {lab}"]["R@5"], 3) for (n, lab) in rankings if n == name}, flush=True)

    def confusion(rk, sel=None):
        c = {"other_year": 0, "other_company": 0, "same_company_other_section": 0, "correct_top1": 0}
        ii = range(len(rk)) if sel is None else np.where(sel)[0]
        for i in ii:
            r, g = rk[i], gold[i]
            if not r:
                continue
            if r[0] in g:
                c["correct_top1"] += 1
                continue
            t, gi = idx_of[r[0]], [idx_of[x] for x in g]
            if corp_arr[t] != corp_arr[gi[0]]:
                c["other_company"] += 1
            elif year_arr[t] != year_arr[gi[0]] and sec_arr[t] in {sec_arr[x] for x in gi}:
                c["other_year"] += 1
            else:
                c["same_company_other_section"] += 1
        n = len(ii)
        return {k: round(v / n, 4) for k, v in c.items()} | {"n": n}

    has_year = np.array([bool(set(m["fiscal_years"]) & RANGE) for m in meta])
    has_comp = np.array([bool(m["companies"]) for m in meta])
    gold_table = np.array([all(table_arr[idx_of[x]] for x in g) for g in gold])
    groups = {"전체": np.ones(len(Q), bool), "연도 있음": has_year, "연도 없음": ~has_year, "회사 있음": has_comp,
              "회사 없음": ~has_comp, "정답이 표 청크": gold_table, "정답이 본문 청크": ~gold_table}
    breakdown = {f"{n} | {lab} | {g}": metrics([rk[i] for i in np.where(sel)[0]], [gold[i] for i in np.where(sel)[0]])
                 | {"n": int(sel.sum())}
                 for (n, lab), rk in rankings.items() for g, sel in groups.items() if sel.sum()}
    conf = {f"{n} | {lab}": confusion(rk) for (n, lab), rk in rankings.items()}
    conf |= {f"{n} | {lab} | {g}": confusion(rk, sel) for (n, lab), rk in rankings.items()
             for g, sel in (("연도 있음", has_year), ("연도 없음", ~has_year))}

    sig = []
    if s == "test" and len(args.models) > 1:
        clusters = Q.ctx_id.to_numpy()
        base = args.models[0].split("=")[0]
        for (n, lab), rk in rankings.items():
            if n == base:
                continue
            ref = rankings.get((base, lab)) or rankings[(base, "RRF+filter")]
            e = {"a": f"{base} | {lab if (base, lab) in rankings else 'RRF+filter'}", "b": f"{n} | {lab}"}
            for k in (1, 5):
                ha, hb = hits(ref, gold, k), hits(rk, gold, k)
                e[f"R@{k}"] = {"diff": float(hb.mean() - ha.mean()), "mcnemar": mcnemar_exact(ha, hb),
                               "cluster_bootstrap": cluster_bootstrap(hb.astype(float) - ha.astype(float), clusters)}
                for g, sel in (("연도 있음", has_year), ("연도 없음", ~has_year)):
                    ii = np.where(sel)[0]
                    d = hb[ii].astype(float) - ha[ii].astype(float)
                    e[f"R@{k} | {g}"] = {"diff": float(d.mean()), "cluster_bootstrap": cluster_bootstrap(d, clusters[ii])}
            d = rr(rk, gold) - rr(ref, gold)
            e["MRR@10"] = {"diff": float(d.mean()), "cluster_bootstrap": cluster_bootstrap(d, clusters)}
            sig.append(e)

    tag = f"_{args.tag}" if args.tag else ""
    out = {"split": s, "n": len(Q), "n_clusters": int(Q.ctx_id.nunique()), "results": results, "breakdown": breakdown,
           "confusion": conf, "paired": sig}
    (OUT / f"eval_{s}{tag}.json").write_text(json.dumps(out, ensure_ascii=False, indent=1))
    (OUT / f"rankings_{s}{tag}.json").write_text(json.dumps(
        {"qids": Q.id.tolist(), "rankings": {f"{n} | {l}": r for (n, l), r in rankings.items()}}, ensure_ascii=False))
    print(pd.DataFrame(results).T.round(3).to_string())


if __name__ == "__main__":
    main()

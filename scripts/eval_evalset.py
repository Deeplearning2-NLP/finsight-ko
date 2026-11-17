"""Retrieval on the in-house evaluation-set draft (data/eval/evalset_draft.jsonl, types 1-4 with gold chunk ids).

Nothing is tuned here: the embedding variant, reranker, RRF and metadata-filter rule are the ones chosen on the
FinDartBench dev split (outputs/midterm/selection_*.json), so the draft set is held out. Systems: BM25 whitespace,
BM25 Kiwi content, public KURE-v1, the chosen fine-tuned KURE (and S1 (a) for reference): dense, RRF, RRF + filter,
RRF + filter + reranker(s). Metrics overall and by question type; paired bootstrap over items (each item has its
own gold) and McNemar vs public KURE in the same configuration.

Usage: CUDA_VISIBLE_DEVICES=0 python scripts/eval_evalset.py [--file data/eval/evalset_draft.jsonl]
Output: outputs/midterm/eval_evalset_draft.json
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
from eval_findartbench_retrieval import rrf  # noqa: E402

from finsight.query_meta import extract
from finsight.tokenize import TOKENIZERS

ROOT = Path(__file__).resolve().parents[1]
M = ROOT / "outputs" / "midterm"
P = ROOT / "outputs" / "proposal"
RANGE = {2022, 2023, 2024}
TYPE_NAME = {1: "단일 수치", 2: "연도 비교", 3: "기업 비교", 4: "서술 요약"}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--file", default=str(ROOT / "data/eval/evalset_draft.jsonl"))
    ap.add_argument("--tag", default="draft")
    args = ap.parse_args()
    items = [json.loads(l) for l in open(args.file, encoding="utf-8")]
    items = [x for x in items if x["type"] in TYPE_NAME and x["gold_chunk_ids"]]
    corpus = pd.read_json(ROOT / "data/processed/corpus_a.jsonl", lines=True, dtype={"rcept_no": str, "chunk_id": str})
    ids = corpus.chunk_id.to_numpy()
    corp_arr, year_arr = corpus.corp_name.to_numpy(), corpus.fiscal_year.to_numpy()
    qs = [x["question"] for x in items]
    gold = [set(x["gold_chunk_ids"]) for x in items]
    meta = [extract(q) for q in qs]

    def filt(i):
        comps, years = set(meta[i]["companies"]), set(meta[i]["fiscal_years"]) & RANGE
        years = years | {y + 1 for y in years}
        if comps and years:
            return np.isin(corp_arr, list(comps)) & np.isin(year_arr, list(years))
        if comps:
            return np.isin(corp_arr, list(comps))
        if years:
            return np.isin(year_arr, list(years))
        return None

    masks = [filt(i) for i in range(len(items))]

    def topk(v, mask, k=10):
        v = v if mask is None else np.where(mask, v, -np.inf)
        t = np.argpartition(-v, k)[:k]
        t = t[np.argsort(-v[t])]
        return [ids[i] for i in t if np.isfinite(v[i])]

    from rank_bm25 import BM25Okapi
    R = {}
    bm_scores = {}
    for tok in ("whitespace", "kiwi_content"):
        bm = BM25Okapi([TOKENIZERS[tok](t) for t in corpus.text])
        bm_scores[tok] = np.vstack([bm.get_scores(TOKENIZERS[tok](q)) for q in qs])
        R[f"BM25 {tok}"] = [topk(bm_scores[tok][i], None) for i in range(len(items))]
    bk = bm_scores["kiwi_content"]
    bm_none = R["BM25 kiwi_content"]
    bm_f = [topk(bk[i], masks[i]) for i in range(len(items))]
    bm_f20 = [topk(bk[i], masks[i], 20) for i in range(len(items))]

    sel_e = json.load(open(M / "selection_embed.json"))["best_embedding"]
    sel_r = json.load(open(M / "selection_rerank.json"))
    best = sel_e.split(" ", 1)[1]
    models = [("KURE-v1 공개", "nlpai-lab/KURE-v1", P / "emb_kure_base.npy"),
              (sel_e + " (dev 선택)", str(M / f"kure-m-{best}"), M / f"emb_m_{best}.npy"),
              ("S1 (a) 연도 포함 질문만", str(P / "kure-s1-a"), P / "emb_s1_a.npy")]
    rerankers = [("bge-zero", "BAAI/bge-reranker-v2-m3")] + \
                [(n, str(M / f"reranker-{t}")) for n, t in (("klue-ft", "klue"), ("bge-ft", "bge"))
                 if (M / f"reranker-{t}").exists()]
    import torch
    from sentence_transformers import CrossEncoder, SentenceTransformer
    texts = corpus.text.tolist()
    idx_of = {c: i for i, c in enumerate(ids)}
    for name, path, cache in models:
        E = np.load(cache).astype(np.float32)
        mdl = SentenceTransformer(path, device="cuda")
        mdl.max_seq_length = 512
        S = mdl.encode(qs, batch_size=64, normalize_embeddings=True) @ E.T
        del mdl
        dense = [topk(S[i], None) for i in range(len(items))]
        R[f"{name} | dense"] = dense
        R[f"{name} | RRF"] = [rrf([a, b]) for a, b in zip(bm_none, dense)]
        R[f"{name} | RRF+filter"] = [rrf([a, b]) for a, b in zip(bm_f, [topk(S[i], masks[i]) for i in range(len(items))])]
        cand = [rrf([a, b], top=20) for a, b in zip(bm_f20, [topk(S[i], masks[i], 20) for i in range(len(items))])]
        for rn, rp in rerankers:
            ce = CrossEncoder(rp, device="cuda", max_length=512)
            from train_reranker_m import fix_token_types
            fix_token_types(ce)
            flat = [(q, texts[idx_of[c]]) for q, cs in zip(qs, cand) for c in cs]
            sc = ce.predict(flat, batch_size=64, show_progress_bar=False)
            out, k = [], 0
            for cs in cand:
                s_ = sc[k:k + len(cs)]
                k += len(cs)
                out.append([cs[j] for j in np.argsort(-s_)][:10])
            R[f"{name} | RRF+filter+{rn}"] = out
            del ce
        torch.cuda.empty_cache()
        print(name, "done", flush=True)

    types = np.array([x["type"] for x in items])
    res = {k: metrics(v, gold) for k, v in R.items()}
    by_type = {f"{k} | {TYPE_NAME[t]}": metrics([v[i] for i in np.where(types == t)[0]],
                                                [gold[i] for i in np.where(types == t)[0]]) | {"n": int((types == t).sum())}
               for k, v in R.items() for t in TYPE_NAME if (types == t).sum()}
    clusters = np.arange(len(items))
    paired = []
    for k, v in R.items():
        if k.startswith(models[0][0]) or k.startswith("BM25"):
            continue
        cfg = k.split(" | ", 1)[1]
        ref = R[f"{models[0][0]} | {cfg}"]
        e = {"a": f"{models[0][0]} | {cfg}", "b": k}
        for kk in (1, 5):
            ha, hb = hits(ref, gold, kk), hits(v, gold, kk)
            e[f"R@{kk}"] = {"diff": float(hb.mean() - ha.mean()), "mcnemar": mcnemar_exact(ha, hb),
                            "bootstrap": cluster_bootstrap(hb.astype(float) - ha.astype(float), clusters)}
        d = rr(v, gold) - rr(ref, gold)
        e["MRR@10"] = {"diff": float(d.mean()), "bootstrap": cluster_bootstrap(d, clusters)}
        paired.append(e)
    out = {"file": args.file, "n_items": len(items), "by_type_n": {TYPE_NAME[t]: int((types == t).sum()) for t in TYPE_NAME},
           "chosen_on_fdb_dev": {"embedding": sel_e, "reranker": sel_r["best_reranker"]},
           "results": res, "by_type": by_type, "paired_vs_public": paired}
    (M / f"eval_evalset_{args.tag}.json").write_text(json.dumps(out, ensure_ascii=False, indent=1))
    print(pd.DataFrame(res).T.round(3).to_string())


if __name__ == "__main__":
    main()

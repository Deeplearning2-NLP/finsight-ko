"""Retrieval on the scope-A original-filing corpus (P4).

FinDartBench 사업보고서 questions are mapped onto P1 chunks by character overlap (no LLM):
  - text is normalized (markdown symbols and whitespace removed) and turned into character 5-gram sets
  - for each FinDartBench context, the report (rcept_no) of the same company with the highest
    coverage |ctx ∩ report| / |ctx| is chosen; gold chunks = chunks of that report whose own 5-grams
    are >= CONTAIN (0.5) contained in the context
  - a context is "mapped" if coverage by its gold chunks >= COVER (0.5); others are dropped and counted
Splits are the T4 ones (outputs/proposal/splits.json): dev for choosing the RRF pair, test for reporting.
Hit@k: any gold chunk in the top k. MRR@10 on the first gold chunk.

Usage: CUDA_VISIBLE_DEVICES=0 python scripts/eval_corpus_a_retrieval.py
Outputs: outputs/proposal/corpus_a_mapping.json, corpus_a_retrieval.json
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from eval_findartbench_retrieval import load, rrf  # noqa: E402

from finsight.bm25 import BM25Retriever

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs" / "proposal"
KS = (1, 5, 10)
N = 5
CONTAIN, COVER = 0.5, 0.5
FDB_TO_DART = {"한국전력": "한국전력공사"}


def norm(s: str) -> str:
    s = re.sub(r"^\[.*?\]\n", "", s)  # chunk section prefix
    return re.sub(r"[\s|#*\-:·ㆍ]+", "", s)


def grams(s: str) -> set[str]:
    s = norm(s)
    return {s[i:i + N] for i in range(max(0, len(s) - N + 1))}


def hit_metrics(ranked: list[list[str]], gold: list[set[str]]) -> dict:
    out = {f"R@{k}": float(np.mean([bool(set(r[:k]) & g) for r, g in zip(ranked, gold)])) for k in KS}
    rr = []
    for r, g in zip(ranked, gold):
        rr.append(next((1 / (i + 1) for i, d in enumerate(r[:10]) if d in g), 0.0))
    out["MRR@10"] = float(np.mean(rr))
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", default=str(ROOT / "data/processed/corpus_a.jsonl"))
    ap.add_argument("--ft-model", default=str(OUT / "kure-v1-findart-smoke"))
    ap.add_argument("--batch", type=int, default=64)
    args = ap.parse_args()

    # chunk_id must stay a string: "20230307000542_0000" would otherwise be parsed as a number
    corpus = pd.read_json(args.corpus, lines=True, dtype={"rcept_no": str, "chunk_id": str})
    assert not corpus.chunk_id.duplicated().any(), "duplicate chunk ids"
    ids = corpus.chunk_id.tolist()
    texts = corpus.text.tolist()
    print("corpus chunks", len(corpus), flush=True)

    # ---------------- mapping FinDartBench contexts -> chunks
    fdb = load(ROOT / "data/raw/findartbench/data.jsonl")
    splits = json.loads((OUT / "splits.json").read_text())
    split_of = {q: s for s in ("train", "dev", "test") for q in splits[s]["question_ids"]}
    fdb["split"] = fdb.id.map(split_of)
    fdb["corp"] = fdb.company.map(lambda c: FDB_TO_DART.get(c, c))
    chunk_grams = [grams(t) for t in texts]
    by_report = defaultdict(list)
    for i, r in enumerate(corpus.rcept_no):
        by_report[r].append(i)
    report_corp = corpus.drop_duplicates("rcept_no").set_index("rcept_no").corp_name.to_dict()
    report_year = corpus.drop_duplicates("rcept_no").set_index("rcept_no").fiscal_year.to_dict()
    report_union = {rc: set().union(*(chunk_grams[i] for i in idx)) for rc, idx in by_report.items()}

    mapping = {}
    for ctx_id, sub in fdb.groupby("ctx_id"):
        corp = sub.corp.iloc[0]
        cg = grams(sub.context.iloc[0])
        best = None
        for rc in by_report:
            if report_corp[rc] != corp or not cg:
                continue
            cov = len(cg & report_union[rc]) / len(cg)
            if best is None or cov > best[1]:
                best = (rc, cov)
        if best is None:
            mapping[ctx_id] = {"corp": corp, "mapped": False, "reason": "no report"}
            continue
        rc = best[0]
        gold = [i for i in by_report[rc] if chunk_grams[i] and len(chunk_grams[i] & cg) / len(chunk_grams[i]) >= CONTAIN]
        gold_cov = len(cg & set().union(*(chunk_grams[i] for i in gold))) / len(cg) if gold else 0.0
        mapping[ctx_id] = {"corp": corp, "rcept_no": rc, "fiscal_year": int(report_year[rc]),
                           "report_coverage": round(best[1], 3), "gold_coverage": round(gold_cov, 3),
                           "gold": [ids[i] for i in gold], "mapped": gold_cov >= COVER,
                           "gold_core_section": any(corpus.is_core_section[i] for i in gold)}
    mp = pd.DataFrame.from_dict(mapping, orient="index")
    fdb["mapped"] = fdb.ctx_id.map(lambda c: mapping[c]["mapped"])
    summary = {
        "criterion": f"char {N}-gram; report = max coverage; gold chunk containment >= {CONTAIN}; mapped if gold coverage >= {COVER}",
        "contexts": len(mp), "contexts_mapped": int(mp.mapped.sum()),
        "context_map_fail_rate": round(1 - float(mp.mapped.mean()), 4),
        "questions": len(fdb), "questions_mapped": int(fdb.mapped.sum()),
        "question_map_fail_rate": round(1 - float(fdb.mapped.mean()), 4),
        "gold_chunks_per_context": pd.Series([len(m.get("gold", [])) for m in mapping.values() if m["mapped"]]).describe().round(2).to_dict(),
        "mapped_fiscal_year": mp[mp.mapped].fiscal_year.value_counts().sort_index().to_dict(),
        "gold_coverage_quantiles": mp.gold_coverage.quantile([0.1, 0.25, 0.5, 0.75]).round(3).to_dict(),
        "by_split": {s: {"questions": int((fdb.split == s).sum()), "mapped": int(((fdb.split == s) & fdb.mapped).sum())}
                     for s in ("train", "dev", "test")},
    }
    (OUT / "corpus_a_mapping.json").write_text(json.dumps({"summary": summary, "contexts": mapping},
                                                          ensure_ascii=False, indent=1, default=str))
    print(json.dumps(summary, ensure_ascii=False, indent=1, default=str), flush=True)

    q = {s: fdb[(fdb.split == s) & fdb.mapped].reset_index(drop=True) for s in ("dev", "test")}
    gold = {s: [set(mapping[c]["gold"]) for c in q[s].ctx_id] for s in q}

    # ---------------- retrieval
    results, rankings = {}, {}

    def record(name, ranks):
        rankings[name] = ranks
        results[name] = {s: hit_metrics(ranks[s], gold[s]) for s in q}
        print(name, {s: {k: round(v, 3) for k, v in results[name][s].items()} for s in q}, flush=True)

    for tok in ("whitespace", "kiwi_content"):
        bm = BM25Retriever(ids, texts, tokenizer=tok)
        record(f"BM25 {tok}", {s: [bm.search(x, 10) for x in q[s].question] for s in q})

    import torch
    from sentence_transformers import SentenceTransformer
    for name, path in [("KURE-v1", "nlpai-lab/KURE-v1"), ("KURE-v1 스모크 미세조정", args.ft_model)]:
        m = SentenceTransformer(path, device="cuda")
        m.max_seq_length = 512
        E = m.encode(texts, batch_size=args.batch, normalize_embeddings=True, convert_to_numpy=True,
                     show_progress_bar=False)
        ranks = {}
        for s in q:
            Q = m.encode(q[s].question.tolist(), batch_size=args.batch, normalize_embeddings=True)
            S = Q @ E.T
            ranks[s] = [[ids[i] for i in np.argsort(-row)[:10]] for row in S]
        record(name, ranks)
        del m, E
        torch.cuda.empty_cache()

    # RRF: pick the dense model by dev MRR, BM25 = kiwi_content (best lexical in T4 and here on dev)
    bm_best = max(["BM25 whitespace", "BM25 kiwi_content"], key=lambda n: results[n]["dev"]["MRR@10"])
    de_best = max(["KURE-v1", "KURE-v1 스모크 미세조정"], key=lambda n: results[n]["dev"]["MRR@10"])
    record(f"RRF({bm_best} + {de_best})",
           {s: [rrf([a, b]) for a, b in zip(rankings[bm_best][s], rankings[de_best][s])] for s in q})

    out_path = OUT / "corpus_a_retrieval.json"
    out = {"corpus_chunks": len(corpus), "n_questions": {s: len(q[s]) for s in q}, "results": results}
    out_path.write_text(json.dumps(out, ensure_ascii=False, indent=1))

    # year confusion: top-1 wrong but same company & same section path as a gold chunk, other fiscal year
    meta = corpus.set_index("chunk_id")[["corp_name", "fiscal_year", "section_path"]]
    best_sys = max(results, key=lambda n: results[n]["dev"]["MRR@10"])
    conf = defaultdict(int)
    for r, g in zip(rankings[best_sys]["test"], gold["test"]):
        if r[0] in g:
            conf["correct"] += 1
            continue
        top, gm = meta.loc[r[0]], meta.loc[list(g)]
        if (top.corp_name == gm.corp_name.iloc[0]) and (top.fiscal_year != gm.fiscal_year.iloc[0]) \
                and top.section_path in set(gm.section_path):
            conf["same_section_other_year"] += 1
        elif top.corp_name == gm.corp_name.iloc[0]:
            conf["same_company_other_chunk"] += 1
        else:
            conf["other_company"] += 1
    out.update({"best_system_by_dev": best_sys, "top1_breakdown_test": dict(conf)})
    out_path.write_text(json.dumps(out, ensure_ascii=False, indent=1))
    print(pd.DataFrame({n: r["test"] for n, r in results.items()}).T.round(3).to_string())
    print("top1 breakdown (test):", dict(conf))


if __name__ == "__main__":
    main()

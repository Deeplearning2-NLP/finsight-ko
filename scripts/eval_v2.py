"""v2 improvements over the v1 final system, selected stage by stage on dev (rules: docs/decisions.md 2026-11-19).

Pipeline per question: metadata (companies, fiscal years; "N 사업보고서" rule A) -> filter mask ({y, y+1} window;
year-less rule E) -> Kiwi BM25 top-d + dense top-d under the mask -> RRF -> cross-encoder reranker (D) over the d (C)
candidates -> top 10. Multi-company questions (B): one masked retrieval per company (same question, or the question
with the other company names removed), each reranked, then interleaved round-robin.

Data: --data v1 (data/eval/evalset_v1.jsonl, items with gold chunks) or --data fdb (FinDartBench on corpus_a, as
eval_m.py). Metrics: R@1/5/10, MRR@10, and on multi-company items "all companies' gold in top-5" (cover@5).
--stages runs the dev selection (A-D on v1 dev, E on fdb dev); --final runs a fixed config list (test, once).
Score caches (BM25 rows, dense query scores, reranker pair scores) live in outputs/midterm/cache_v2/.

Usage: CUDA_VISIBLE_DEVICES=0 python scripts/eval_v2.py --data v1 --split dev --stages
       CUDA_VISIBLE_DEVICES=0 python scripts/eval_v2.py --data fdb --split dev --stages --base outputs/midterm/selection_v2.json
       CUDA_VISIBLE_DEVICES=0 python scripts/eval_v2.py --data v1 --split test --final
Outputs: outputs/midterm/eval_v2_<data>_<split>.json, outputs/midterm/selection_v2.json (dev stages)
"""
from __future__ import annotations

import argparse
import hashlib
import json
import pickle
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
import os
# G1: FINSIGHT_CORPUS=unitfix_AB|unitfix_ABC selects data/processed/corpus_a_<tag>.jsonl (same chunk ids) with its
# own caches and M3 embeddings (scripts/unit_fix_encode.py); FinDartBench's precomputed BM25 rows are not used then.
CORPUS_TAG = os.environ.get("FINSIGHT_CORPUS", "")
C = M / ("cache_v2" + (f"_{CORPUS_TAG}" if CORPUS_TAG else ""))
RANGE = {2022, 2023, 2024}
LATEST = 2024
EMB = {"public": ("nlpai-lab/KURE-v1", P / "emb_kure_base.npy"), "m3": (str(M / "kure-m-hard"), M / "emb_m_hard.npy")}
if CORPUS_TAG:
    EMB = {"m3": (str(M / "kure-m-hard"), M / f"emb_m_hard_{CORPUS_TAG}.npy")}
RERANKERS = {"bge-ft": M / "reranker-bge", "klue-ft": M / "reranker-klue", "bge-mined": M / "reranker-bge_mined",
             "klue-mined": M / "reranker-klue_mined", "bge-zero": "BAAI/bge-reranker-v2-m3"}
BASE = {"emb": "m3", "yr": None, "dec": "off", "depth": 20, "rr": "bge-ft", "yl": "none"}

ACCOUNTS = ["당기순이익", "순이익", "영업이익", "매출액", "영업수익", "매출", "자산총계", "부채총계", "자본총계"]
_CLEAN = ["중 어느 회사의", "중 어느 회사가", "중 어느 쪽이", "중 어디가", "중 어디", "어느 회사의", "어느 회사가", "각각", "두 회사"]


def subquery(q: str, meta: dict, c: str, mode: str) -> str:
    """Per-company sub-question for multi-company questions (stage B).
    strip: the question with the other companies' names removed.
    template: numeric -> "<company>의 <years> 사업연도 연결 기준 <account>은 얼마인가?"; otherwise the question with all
    company names removed, comparison words dropped and "<company>의" prefixed. Rule-based (no LLM)."""
    spans = [s for s in meta["spans"] if s["kind"].startswith("company")]
    if mode == "strip":
        for s in spans:
            if s["value"] != c:
                q = q.replace(s["text"], " ")
        return " ".join(q.split())
    acc = next((a for a in ACCOUNTS if a in q), None)
    years = sorted(set(meta["fiscal_years"]) & RANGE)
    if acc and years:
        return f"{c}의 {', '.join(map(str, years))} 사업연도 연결 기준 {acc}은 얼마인가?"
    import re
    for s in sorted(spans, key=lambda s: -len(s["text"])):
        q = re.sub(re.escape(s["text"]) + r"(이랑|은|는|이|가|의|와|과|도|랑)?", " ", q)
    for w in _CLEAN:
        q = q.replace(w, " ")
    return f"{c}의 " + " ".join(q.split())


def key(cfg: dict) -> str:
    return "|".join(f"{k}={cfg[k]}" for k in ("emb", "yr", "dec", "depth", "rr", "yl"))


class Env:
    def __init__(self, data: str, split: str):
        C.mkdir(parents=True, exist_ok=True)
        self.corpus = pd.read_json(ROOT / f"data/processed/corpus_a{'_' + CORPUS_TAG if CORPUS_TAG else ''}.jsonl", lines=True,
                                   dtype={"rcept_no": str, "chunk_id": str})
        self.ids = self.corpus.chunk_id.to_numpy()
        self.texts = self.corpus.text.tolist()
        self.corp_arr, self.year_arr = self.corpus.corp_name.to_numpy(), self.corpus.fiscal_year.to_numpy()
        self.idx_of = {c: i for i, c in enumerate(self.ids)}
        self.data, self.split = data, split
        if data == "v1":
            items = [json.loads(l) for l in open(ROOT / "data/eval/evalset_v1.jsonl", encoding="utf-8")]
            items = [x for x in items if x.get("split") == split and x["type"] in (1, 2, 3, 4) and x["gold_chunk_ids"]]
            self.q = [x["question"] for x in items]
            self.gold = [set(x["gold_chunk_ids"]) for x in items]
            self.qid = [x["qid"] for x in items]
            self.types = np.array([x["type"] for x in items])
            self.origin = np.array([x["origin"] for x in items])
            self.clusters = np.arange(len(items))
            self.bm_rows = None
        else:
            from eval_corpus_a_retrieval import FDB_TO_DART  # noqa: F401
            from eval_findartbench_retrieval import load
            mapping = json.loads((P / "corpus_a_mapping.json").read_text())["contexts"]
            fdb = load(ROOT / "data/raw/findartbench/data.jsonl")
            sp = json.loads((P / "splits.json").read_text())
            so = {qid: s for s in ("train", "dev", "test") for qid in sp[s]["question_ids"]}
            fdb["split"] = fdb.id.map(so)
            fdb = fdb[fdb.ctx_id.map(lambda c: mapping[c]["mapped"])]
            Q = fdb[fdb.split == split].reset_index(drop=True)
            self.q = Q.question.tolist()
            self.gold = [set(mapping[c]["gold"]) for c in Q.ctx_id]
            self.qid = Q.id.tolist()
            self.types = np.zeros(len(Q), int)
            self.origin = np.array(["fdb"] * len(Q))
            self.clusters = Q.ctx_id.to_numpy()
            self.bm_rows = None if CORPUS_TAG else \
                {q: r for q, r in zip(self.q, np.load(P / "r1_bm25_scores.npz")[split].astype(np.float32))}
        self._bm = None
        self._dense: dict[str, dict[str, np.ndarray]] = {}
        self._rr_cache: dict[str, dict] = {}

    # ---------------------------------------------------------------- scores
    def bm25(self, text: str) -> np.ndarray:
        if self.bm_rows is not None and text in self.bm_rows:
            return self.bm_rows[text]
        f = C / f"bm25_{hashlib.md5(text.encode()).hexdigest()}.npy"
        if f.exists():
            return np.load(f)
        if self._bm is None:
            tok_cache = C / "corpus_kiwi_content.pkl"
            if tok_cache.exists():
                toks = pickle.load(open(tok_cache, "rb"))
            else:
                toks = [TOKENIZERS["kiwi_content"](t) for t in self.texts]
                pickle.dump(toks, open(tok_cache, "wb"))
            from rank_bm25 import BM25Okapi
            self._bm = BM25Okapi(toks)
        s = self._bm.get_scores(TOKENIZERS["kiwi_content"](text)).astype(np.float32)
        np.save(f, s)
        return s

    def dense_prepare(self, emb: str, texts: list[str]) -> None:
        d = self._dense.setdefault(emb, {})
        need = [t for t in dict.fromkeys(texts) if t not in d]
        if not need:
            return
        f = C / f"dense_{emb}.pkl"
        if f.exists() and not d:
            d.update(pickle.load(open(f, "rb")))
            need = [t for t in need if t not in d]
        if need:
            import torch
            from sentence_transformers import SentenceTransformer
            path, cache = EMB[emb]
            if not hasattr(self, "_E") or self._E[0] != emb:
                self._E = (emb, np.load(cache).astype(np.float32))
            m = SentenceTransformer(path, device="cuda")
            m.max_seq_length = 512
            V = m.encode(need, batch_size=64, normalize_embeddings=True) @ self._E[1].T
            del m
            torch.cuda.empty_cache()
            for t, v in zip(need, V):
                d[t] = v.astype(np.float16)
            pickle.dump(d, open(f, "wb"))

    def rerank_scores(self, name: str, pairs: list[tuple[str, str]]) -> None:
        cache = self._rr_cache.setdefault(name, {})
        f = C / f"rr_{name}.pkl"
        if f.exists() and not cache:
            cache.update(pickle.load(open(f, "rb")))
        need = list(dict.fromkeys(p for p in pairs if p not in cache))
        if not need:
            return
        import torch
        from sentence_transformers import CrossEncoder
        from train_reranker_m import fix_token_types
        ce = CrossEncoder(str(RERANKERS[name]), device="cuda", max_length=512)
        fix_token_types(ce)
        sc = ce.predict([(q, self.texts[self.idx_of[c]]) for q, c in need], batch_size=64, show_progress_bar=False)
        for p, s in zip(need, sc):
            cache[p] = float(s)
        del ce
        torch.cuda.empty_cache()
        pickle.dump(cache, open(f, "wb"))

    # ---------------------------------------------------------------- retrieval
    def mask(self, comps: set, years: set):
        years = (years & RANGE)
        years = years | {y + 1 for y in years}
        m = None
        if comps:
            m = np.isin(self.corp_arr, list(comps))
        if years:
            y = np.isin(self.year_arr, list(years))
            m = y if m is None else m & y
        return m

    def topk(self, v, mask, k):
        v = v if mask is None else np.where(mask, v, -np.inf)
        t = np.argpartition(-v, k)[:k]
        t = t[np.argsort(-v[t])]
        return [self.ids[i] for i in t if np.isfinite(v[i])]

    def plan(self, cfg: dict) -> list[list[tuple[str, list[str]]]]:
        """Per question: list of (query text used for reranking, candidate chunk ids) — one entry per sub-retrieval."""
        out = []
        d = cfg["depth"] if cfg["rr"] != "none" else 10
        for q in self.q:
            meta = extract(q, report_rule=cfg["yr"])
            comps, years = set(meta["companies"]), set(meta["fiscal_years"])
            subs = [(q, comps)]
            if cfg["dec"] != "off" and len(comps) >= 2:
                subs = []
                for c in sorted(comps):
                    qt = q if cfg["dec"] == "same" else subquery(q, meta, c, cfg["dec"])
                    subs.append((qt, {c}))
            entry = []
            for qt, cs in subs:
                masks = [self.mask(cs, years)]
                if not (years & RANGE) and cfg["yl"] != "none":
                    latest = self.mask(cs, {LATEST})
                    masks = [latest] if cfg["yl"] == "latest" else masks + [latest]
                bm, de = self.bm25(qt), self._dense[cfg["emb"]][qt].astype(np.float32)
                lists = []
                for mk in masks:
                    lists += [self.topk(bm, mk, d), self.topk(de, mk, d)]
                entry.append((qt, rrf(lists, top=d)))
            out.append(entry)
        return out

    def texts_needed(self, cfg: dict) -> list[str]:
        ts = []
        for q in self.q:
            ts.append(q)
            if cfg["dec"] in ("strip", "template"):
                meta = extract(q, report_rule=cfg["yr"])
                comps = set(meta["companies"])
                if len(comps) >= 2:
                    ts += [subquery(q, meta, c, cfg["dec"]) for c in sorted(comps)]
        return ts

    def run(self, cfg: dict) -> list[list[str]]:
        self.dense_prepare(cfg["emb"], self.texts_needed(cfg))
        plan = self.plan(cfg)
        if cfg["rr"] != "none":
            self.rerank_scores(cfg["rr"], [(qt, c) for entry in plan for qt, cs in entry for c in cs])
            sc = self._rr_cache[cfg["rr"]]
        ranked = []
        for entry in plan:
            lists = []
            for qt, cs in entry:
                lists.append(sorted(cs, key=lambda c: -sc[(qt, c)]) if cfg["rr"] != "none" else cs)
            if len(lists) == 1:
                ranked.append(lists[0][:10])
                continue
            seen, out = set(), []
            for i in range(max(map(len, lists))):  # round-robin interleave
                for l in lists:
                    if i < len(l) and l[i] not in seen:
                        seen.add(l[i])
                        out.append(l[i])
            ranked.append(out[:10])
        return ranked

    # ---------------------------------------------------------------- metrics
    def score(self, ranked) -> dict:
        out = metrics(ranked, self.gold)
        multi = [i for i, g in enumerate(self.gold) if len({self.corp_arr[self.idx_of[x]] for x in g}) >= 2]
        if multi:
            cov = []
            for i in multi:
                need = {self.corp_arr[self.idx_of[x]] for x in self.gold[i]}
                got = {self.corp_arr[self.idx_of[x]] for x in ranked[i][:5] if x in self.gold[i]}
                cov.append(need <= got)
            out["cover@5_multi"] = float(np.mean(cov))
            out["n_multi"] = len(multi)
        return out


def stages_v1(env: Env) -> dict:
    log, cur = [], dict(BASE)

    def stage(name, field, options, crit="MRR@10"):
        nonlocal cur
        base_s = env.score(env.run(cur))
        res = {str(cur[field]): base_s}
        for o in options:
            if o == cur[field]:
                continue
            c = dict(cur, **{field: o})
            res[str(o)] = env.score(env.run(c))
        best, best_v = cur[field], base_s[crit]
        for o in options:
            s = res[str(o)]
            ok = s[crit] > best_v + 0.01
            if crit != "MRR@10":
                ok = ok and s["MRR@10"] >= base_s["MRR@10"] - 0.01
            if ok:
                best, best_v = o, s[crit]
        log.append({"stage": name, "field": field, "criterion": crit, "results": res, "chosen": best})
        print(name, {k: (round(v["MRR@10"], 3), round(v.get("cover@5_multi", -1), 3)) for k, v in res.items()},
              "->", best, flush=True)
        cur = dict(cur, **{field: best})

    stage("A report-year rule", "yr", [None, "fiscal", "filing", "both"])
    stage("B multi-company decomposition", "dec", ["off", "same", "strip", "template"], crit="cover@5_multi")
    stage("C rerank depth", "depth", [20, 40])
    avail = [r for r in ("bge-ft", "klue-ft", "bge-mined", "klue-mined") if Path(str(RERANKERS[r])).exists()]
    stage("D reranker", "rr", avail)
    return {"stages": log, "chosen": cur}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", choices=["v1", "fdb"], required=True)
    ap.add_argument("--split", choices=["dev", "test"], required=True)
    ap.add_argument("--stages", action="store_true")
    ap.add_argument("--final", action="store_true")
    args = ap.parse_args()
    env = Env(args.data, args.split)
    sel_f = M / "selection_v2.json"
    if args.stages:
        assert args.split == "dev"
        if args.data == "v1":
            sel = stages_v1(env)
        else:  # stage E on FinDartBench dev, everything else as chosen on v1 dev
            sel = json.loads(sel_f.read_text())
            cur = dict(sel["chosen"])
            res = {}
            for o in ("none", "latest", "latest_soft"):
                res[o] = env.score(env.run(dict(cur, yl=o)))
            ref = {"v1 final (FDB dev)": env.score(env.run(dict(BASE)))}
            best, bv = "none", res["none"]["MRR@10"]
            for o in ("latest", "latest_soft"):
                if res[o]["MRR@10"] > bv + 0.01:
                    best, bv = o, res[o]["MRR@10"]
            print("E", {k: round(v["MRR@10"], 3) for k, v in res.items()}, "->", best, ref, flush=True)
            sel["stages"] = [s for s in sel["stages"] if not s["stage"].startswith("E")] + [
                {"stage": "E year-less rule (FinDartBench dev)", "field": "yl", "criterion": "MRR@10",
                 "results": res, "chosen": best, "reference": ref}]
            sel["chosen"] = dict(cur, yl=best)
        sel_f.write_text(json.dumps(sel, ensure_ascii=False, indent=1, default=str))
        return
    sel = json.loads(sel_f.read_text())
    final = dict(sel["chosen"])
    configs = {
        "v1 final (M3 + RRF+filter + bge-ft)": dict(BASE),
        "public KURE + RRF+filter (no rerank)": dict(BASE, emb="public", rr="none"),
        "M3 + RRF+filter (no rerank)": dict(BASE, rr="none"),
        "public KURE + v2 rules + same reranker": dict(final, emb="public"),
        "v2 final": final,
        "v2 final with klue reranker": dict(final, rr="klue-mined" if "mined" in final["rr"] else "klue-ft"),
        "v2 final without decomposition": dict(final, dec="off"),
        "v2 final without year rules": dict(final, yr=None, yl="none"),
    }
    R = {n: env.run(c) for n, c in configs.items()}
    res = {n: env.score(r) | {"config": key(configs[n])} for n, r in R.items()}
    by_type = {}
    for n, r in R.items():
        for t in sorted(set(env.types)):
            ii = np.where(env.types == t)[0]
            by_type[f"{n} | type{t}"] = metrics([r[i] for i in ii], [env.gold[i] for i in ii]) | {"n": len(ii)}
        for o in sorted(set(env.origin)):
            ii = np.where(env.origin == o)[0]
            by_type[f"{n} | {o}"] = metrics([r[i] for i in ii], [env.gold[i] for i in ii]) | {"n": len(ii)}
    paired = []
    for a, b in [("v1 final (M3 + RRF+filter + bge-ft)", "v2 final"), ("public KURE + RRF+filter (no rerank)", "v2 final"),
                 ("public KURE + v2 rules + same reranker", "v2 final"), ("v2 final without decomposition", "v2 final"),
                 ("v2 final without year rules", "v2 final")]:
        e = {"a": a, "b": b}
        for k in (1, 5):
            ha, hb = hits(R[a], env.gold, k), hits(R[b], env.gold, k)
            e[f"R@{k}"] = {"diff": float(hb.mean() - ha.mean()), "mcnemar_p": mcnemar_exact(ha, hb)["p"],
                           "ci95": cluster_bootstrap(hb.astype(float) - ha.astype(float), env.clusters)["ci95"]}
        d = rr(R[b], env.gold) - rr(R[a], env.gold)
        e["MRR@10"] = {"diff": float(d.mean()), "ci95": cluster_bootstrap(d, env.clusters)["ci95"]}
        paired.append(e)
    out = {"data": args.data, "split": args.split, "n": len(env.q), "final_config": key(final),
           "results": res, "breakdown": by_type, "paired": paired}
    (M / f"eval_v2_{args.data}_{args.split}.json").write_text(json.dumps(out, ensure_ascii=False, indent=1, default=str))
    (M / f"rankings_v2_{args.data}_{args.split}.json").write_text(json.dumps({"qids": env.qid, "rankings": R}, ensure_ascii=False))
    print(pd.DataFrame(res).T.drop(columns="config").round(3).to_string(), flush=True)


if __name__ == "__main__":
    main()

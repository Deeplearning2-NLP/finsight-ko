"""M5: 30 random test failures (gold outside top-5) of the final system, for manual error analysis.

Reads outputs/midterm/rankings_test_final.json, picks the system key given by --system (default: the dev-chosen
embedding with RRF+filter+dev-chosen reranker), samples 30 failures with seed 42 and writes question, gold chunk
and top-1 chunk side by side, with the automatic confusion type. Classification is done by reading.

Usage: python scripts/m5_errors.py [--system "M3 hard | RRF+filter+bge-ft"]
Output: outputs/midterm/error_analysis_30.md
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from eval_findartbench_retrieval import load  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
M = ROOT / "outputs/midterm"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--system", default="")
    args = ap.parse_args()
    if not args.system:
        emb = json.load(open(M / "selection_embed.json"))["best_embedding"]
        rr = json.load(open(M / "selection_rerank.json"))["best_reranker"]
        args.system = f"{emb} | RRF+filter+{rr}"
    rk = json.load(open(M / "rankings_test_final.json"))
    ranks = rk["rankings"][args.system]
    corpus = pd.read_json(ROOT / "data/processed/corpus_a.jsonl", lines=True,
                          dtype={"rcept_no": str, "chunk_id": str}).set_index("chunk_id")
    mapping = json.load(open(ROOT / "outputs/proposal/corpus_a_mapping.json"))["contexts"]
    fdb = load(ROOT / "data/raw/findartbench/data.jsonl").set_index("id")
    fails = []
    for qid, r in zip(rk["qids"], ranks):
        g = set(mapping[fdb.loc[qid, "ctx_id"]]["gold"])
        if not set(r[:5]) & g:
            fails.append((qid, r, sorted(g)))
    sample = random.Random(42).sample(fails, 30)
    lines = [f"# 중간 발표 오류 분석 30건 — {args.system}", "",
             f"test {len(ranks)}문항 중 정답이 top-5 밖인 실패 {len(fails)}건에서 seed 42로 30건.", ""]
    for n, (qid, r, g) in enumerate(sample, 1):
        gm, top = corpus.loc[g[0]], corpus.loc[r[0]] if r else None
        auto = "-" if top is None else ("다른 회사" if top.corp_name != gm.corp_name else
                                        "다른 사업연도" if top.fiscal_year != gm.fiscal_year and top.section_path == gm.section_path
                                        else "같은 회사 다른 청크")
        lines += [f"## {n}. {fdb.loc[qid, 'question']}", "",
                  f"- qid `{qid}`, 자동 분류: {auto}",
                  f"- **정답** `{g[0]}` {gm.corp_name} FY{gm.fiscal_year} · {gm.section_path}",
                  f"  > {gm.text[:300].replace(chr(10), ' ')}",
                  f"- **1위** `{r[0]}` {top.corp_name} FY{top.fiscal_year} · {top.section_path}" if top is not None else "- 1위 없음",
                  f"  > {top.text[:300].replace(chr(10), ' ')}" if top is not None else "",
                  "- 유형: ", "- 왜 틀렸나: ", ""]
    (M / "error_analysis_30.md").write_text("\n".join(lines))
    print(len(fails), "failures; wrote", M / "error_analysis_30.md")


if __name__ == "__main__":
    main()

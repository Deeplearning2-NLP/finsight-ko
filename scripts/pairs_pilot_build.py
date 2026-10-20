"""Q6 step 3: parse pilot generations into training pairs with year hard negatives (no training).

- Parse the model's JSON; keep questions that name the company and the fiscal year.
- Cap at 300 pairs (seed 42).
- Hard negative: among chunks of the same company and section (II/IV) from *another* fiscal year,
  the one with the highest BM25 (Kiwi content) score for the question.
- Lexical copying check: Kiwi content-morpheme Jaccard(question, positive chunk); FinDartBench
  question-context Jaccard (사업보고서) as a reference distribution.
- 30 random pairs are written to a review sheet for manual judgement.

Usage: python scripts/pairs_pilot_build.py
Outputs: data/train/pairs_pilot.jsonl, outputs/proposal/pairs_pilot_stats.json, pairs_pilot_review_30.md
"""
from __future__ import annotations

import json
import random
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from rank_bm25 import BM25Okapi

sys.path.insert(0, str(Path(__file__).resolve().parent))
from eval_findartbench_retrieval import load  # noqa: E402

from finsight.tokenize import content_morphs

ROOT = Path(__file__).resolve().parents[1]
TRAIN = ROOT / "data" / "train"
OUT = ROOT / "outputs" / "proposal"
N_PAIRS = 300
SEED = 42
DEICTIC = re.compile(r"(해당|위의|위 표|이 표|본문|상기)")


def parse(raw: str) -> list[dict]:
    s = re.sub(r"^```(?:json)?|```$", "", raw.strip(), flags=re.M).strip()
    m = re.search(r"\{.*\}", s, re.S)
    if not m:
        return []
    try:
        qs = json.loads(m.group(0)).get("questions", [])
    except json.JSONDecodeError:
        return []
    return [q for q in qs if isinstance(q, dict) and str(q.get("question", "")).strip()]


def jaccard(a: set, b: set) -> float:
    return len(a & b) / len(a | b) if a | b else 0.0


def describe(x) -> dict:
    x = np.asarray(x)
    return {"n": int(len(x)), "mean": round(float(x.mean()), 3), "p10": round(float(np.percentile(x, 10)), 3),
            "p50": round(float(np.percentile(x, 50)), 3), "p90": round(float(np.percentile(x, 90)), 3)}


def main() -> None:
    src = {c["chunk_id"]: c for c in map(json.loads, open(TRAIN / "pilot_source_chunks.jsonl", encoding="utf-8"))}
    pool = pd.read_json(TRAIN / "pilot_pool_chunks.jsonl", lines=True, dtype={"rcept_no": str, "chunk_id": str})
    gens = [json.loads(l) for l in open(TRAIN / "pilot_generations.jsonl", encoding="utf-8")]
    cfg = json.loads((TRAIN / "pilot_generation_config.json").read_text())
    gen_tag = "llm:" + cfg["model"].split("/")[-1]

    cands, n_parse_fail, n_raw_q = [], 0, 0
    for g in gens:
        qs = parse(g["raw"])
        n_parse_fail += not qs
        n_raw_q += len(qs)
        c = src[g["chunk_id"]]
        for k, q in enumerate(qs):
            text = q["question"].strip()
            cands.append({"chunk_id": c["chunk_id"], "k": k, "question": text, "answer": str(q.get("answer", "")),
                          "is_numeric": bool(q.get("is_numeric")),
                          "has_corp": c["corp_name"] in text, "has_year": str(c["fiscal_year"]) in text,
                          "has_deictic": bool(DEICTIC.search(text))})
    cdf = pd.DataFrame(cands)
    keep = cdf[cdf.has_corp & cdf.has_year].reset_index(drop=True)
    idx = list(range(len(keep)))
    random.Random(SEED).shuffle(idx)
    keep = keep.iloc[sorted(idx[:N_PAIRS])].reset_index(drop=True)

    # hard negatives: same company & section, other fiscal year, best BM25 for the question
    pool_tok = {cid: content_morphs(t) for cid, t in zip(pool.chunk_id, pool.text)}
    pairs = []
    for i, r in keep.iterrows():
        c = src[r.chunk_id]
        cand = pool[(pool.corp_name == c["corp_name"]) & (pool.section == c["section"])
                    & (pool.fiscal_year != c["fiscal_year"])]
        neg = None
        if len(cand):
            bm = BM25Okapi([pool_tok[x] for x in cand.chunk_id])
            neg = cand.iloc[int(np.argmax(bm.get_scores(content_morphs(r.question))))]
        qt, pt = set(content_morphs(r.question)), set(pool_tok[c["chunk_id"]])
        pairs.append({"pair_id": f"pilot-{i + 1:04d}", "anchor": r.question, "positive": c["text"],
                      "positive_id": c["chunk_id"], "negative": None if neg is None else neg.text,
                      "negative_id": None if neg is None else neg.chunk_id,
                      "negative_fiscal_year": None if neg is None else int(neg.fiscal_year),
                      "corp_name": c["corp_name"], "industry": c["industry"], "fiscal_year": c["fiscal_year"],
                      "section": c["section"], "answer": r.answer, "is_numeric": r.is_numeric,
                      "has_deictic": r.has_deictic, "lexical_jaccard": round(jaccard(qt, pt), 4),
                      "generated_by": gen_tag, "review": None})
    with open(TRAIN / "pairs_pilot.jsonl", "w", encoding="utf-8") as f:
        for p in pairs:
            f.write(json.dumps(p, ensure_ascii=False) + "\n")

    # reference: FinDartBench question-context overlap (사업보고서, one question per context)
    fdb = load(ROOT / "data/raw/findartbench/data.jsonl").drop_duplicates("ctx_id")
    fdb_j = [jaccard(set(content_morphs(q)), set(content_morphs(c))) for q, c in zip(fdb.question, fdb.context)]
    pdf = pd.DataFrame(pairs)
    stats = {"generation": {k: v for k, v in cfg.items() if "prompt" not in k},
             "chunks": len(gens), "raw_questions": n_raw_q, "parse_fail_chunks": n_parse_fail,
             "with_corp_and_year": int((cdf.has_corp & cdf.has_year).sum()),
             "missing_corp": int((~cdf.has_corp).sum()), "missing_year": int((~cdf.has_year).sum()),
             "deictic_in_kept": int(pdf.has_deictic.sum()), "pairs": len(pairs),
             "numeric_share": round(float(pdf.is_numeric.mean()), 3),
             "with_hard_negative": int(pdf.negative_id.notna().sum()),
             "negative_year_offset": (pdf.negative_fiscal_year - pdf.fiscal_year).abs().value_counts().to_dict(),
             "lexical_jaccard_pilot": describe(pdf.lexical_jaccard),
             "lexical_jaccard_findartbench_ref": describe(fdb_j),
             "by_industry": pdf.industry.value_counts().to_dict()}
    (OUT / "pairs_pilot_stats.json").write_text(json.dumps(stats, ensure_ascii=False, indent=1, default=str))

    sample = random.Random(SEED).sample(pairs, 30)
    lines = ["# 학습 쌍 파일럿 검수 30쌍", "", f"`data/train/pairs_pilot.jsonl` {len(pairs)}쌍 중 seed {SEED}로 30쌍.", ""]
    for n, p in enumerate(sample, 1):
        lines += [f"## {n}. `{p['pair_id']}` {p['corp_name']} FY{p['fiscal_year']} {p['section']}", "",
                  f"- 질문: {p['anchor']}", f"- 모델 답: {p['answer']}",
                  f"- 겹침(Jaccard): {p['lexical_jaccard']}, 숫자 질문: {p['is_numeric']}",
                  f"- 본문: {p['positive'][:600].replace(chr(10), ' ')}",
                  f"- hard negative (FY{p['negative_fiscal_year']}): {(p['negative'] or '')[:200].replace(chr(10), ' ')}",
                  "- 판정: ", "- 메모: ", ""]
    (OUT / "pairs_pilot_review_30.md").write_text("\n".join(lines))
    print(json.dumps(stats, ensure_ascii=False, indent=1, default=str))


if __name__ == "__main__":
    main()

"""R2 step 3: filter generations into training pairs with year hard negatives (no training).

Filters, in order (counts kept for each):
  1. JSON parse
  2. question names the company and the fiscal year (finsight.query_meta with the training-company aliases)
  3. answer says the text has no answer ("제공되지 않", "명시되어 있지 않" …)
  4. deictic words ("해당", "위 표", "본문" …)
  5. numeric questions: every number in the answer must be in the chunk, or the answer's KRW amount
     (finsight.numbers.find_amounts) must equal a chunk value times the chunk's "(단위: …)" within 0.5%;
     years (2000-2035) are ignored; a numeric answer without any number fails
Hard negative: same company, same top-level section, *another* fiscal year, best BM25 (Kiwi content) for the
question, skipping candidates with char 5-gram Jaccard >= 0.8 to the positive or that contain the answer
(the answer text, or for numeric answers any of its 3+ digit numbers — a later report repeats this year's
figures in its prior-year column).
Then 300 pairs at random (seed 42) and a 30-pair review sheet.

Usage: python scripts/pairs_r2_build.py
Outputs: data/train/pairs_r2.jsonl, outputs/proposal/pairs_r2_stats.json, pairs_r2_review_30.md
"""
from __future__ import annotations

import json
import random
import re
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
from rank_bm25 import BM25Okapi

sys.path.insert(0, str(Path(__file__).resolve().parent))
from eval_corpus_a_retrieval import grams  # noqa: E402
from eval_findartbench_retrieval import load  # noqa: E402
from pairs_pilot_build import DEICTIC, describe, jaccard, parse  # noqa: E402

from finsight.numbers import UNIT_MULTIPLIER, find_amounts, rel_close
from finsight.query_meta import extract
from finsight.tokenize import content_morphs

ROOT = Path(__file__).resolve().parents[1]
TRAIN = ROOT / "data" / "train"
OUT = ROOT / "outputs" / "proposal"
N_PAIRS = 300
SEED = 42
NOANS = re.compile(r"(제공되지 않|명시되어 있지 않|명시되지 않|언급하고 있지 않|언급되지 않|알 수 없|공개되어 있지 않|"
                   r"제시되어 있지 않|정보가 없|포함되어 있지 않|나와 있지 않|확인할 수 없)")
NUM = re.compile(r"\d[\d,]*(?:\.\d+)?")
UNIT = re.compile(r"단위\s*[:：]\s*([^\s\),%]+)")
EXTRA_ALIASES = {"에이치디현대오일뱅크": ["HD현대오일뱅크", "현대오일뱅크"], "삼성E&A": ["삼성이앤에이"],
                 "비바리퍼블리카": ["토스"], "DB손해보험": ["DB손보"]}


def numbers_ok(answer: str, chunk: str) -> tuple[bool, str]:
    toks = [t.replace(",", "") for t in NUM.findall(answer)]
    toks = [t for t in toks if not (len(t) == 4 and t.isdigit() and 2000 <= int(t) <= 2035)]
    if not toks:
        return False, "no_number"
    chunk_nums = {t.replace(",", "") for t in NUM.findall(chunk)}
    if all(t in chunk_nums for t in toks):
        return True, "tokens_in_chunk"
    amounts = find_amounts(answer)
    units = [UNIT_MULTIPLIER.get(u) for u in UNIT.findall(chunk)] or [1]
    vals = [float(t) for t in chunk_nums if t.replace(".", "").isdigit()]
    for a in amounts:
        for mult in {m for m in units if m}:
            if any(rel_close(a, v * mult, 0.005) for v in vals):
                return True, "krw_match"
    return False, "number_not_in_chunk"


def norm(s: str) -> str:
    return re.sub(r"[\s|#*\-:·ㆍ,.()]+", "", s)


def main() -> None:
    src = {c["chunk_id"]: c for c in map(json.loads, open(TRAIN / "r2_source_chunks.jsonl", encoding="utf-8"))}
    pool = pd.read_json(ROOT / "data/processed/corpus_train.jsonl", lines=True, dtype={"rcept_no": str, "chunk_id": str})
    pool["sec1"] = pool.section_path.str.split(" > ").str[0]
    gens = [json.loads(l) for l in open(TRAIN / "r2_generations.jsonl", encoding="utf-8")]
    cfg = json.loads((TRAIN / "r2_generation_config.json").read_text())
    aliases = {c: [c] + EXTRA_ALIASES.get(c, []) for c in pool.corp_name.unique()}

    stage = Counter()
    kept, dropped = [], []
    for g in gens:
        c = src[g["chunk_id"]]
        qs = parse(g["raw"])
        if not qs:
            stage["parse_fail_chunks"] += 1
        for q in qs:
            stage["generated"] += 1
            text, ans = q["question"].strip(), str(q.get("answer", ""))
            meta = extract(text, aliases=aliases)
            rec = {"chunk_id": c["chunk_id"], "question": text, "answer": ans, "is_numeric": bool(q.get("is_numeric")),
                   "value_table": c["value_table"]}
            reason = None
            if c["corp_name"] not in meta["companies"]:
                reason = "missing_company"
            elif c["fiscal_year"] not in meta["fiscal_years"]:
                reason = "missing_year"
            elif NOANS.search(ans):
                reason = "answer_says_not_in_text"
            elif DEICTIC.search(text):
                reason = "deictic"
            elif rec["is_numeric"]:
                ok, how = numbers_ok(ans, c["text"])
                rec["number_check"] = how
                if not ok:
                    reason = f"numeric_{how}"
            if reason:
                stage[f"drop_{reason}"] += 1
                dropped.append({**rec, "reason": reason})
            else:
                kept.append(rec)
    stage["kept_after_filters"] = len(kept)

    idx = list(range(len(kept)))
    random.Random(SEED).shuffle(idx)
    sample = [kept[i] for i in sorted(idx[:N_PAIRS])]

    pool_tok = {cid: content_morphs(t) for cid, t in zip(pool.chunk_id, pool.text)}
    pairs, skipped_neg = [], Counter()
    for i, r in enumerate(sample):
        c = src[r["chunk_id"]]
        sec1 = c["section_path"].split(" > ")[0]
        cand = pool[(pool.corp_name == c["corp_name"]) & (pool.sec1 == sec1) & (pool.fiscal_year != c["fiscal_year"])]
        neg = None
        if len(cand):
            bm = BM25Okapi([pool_tok[x] for x in cand.chunk_id])
            order = np.argsort(-bm.get_scores(content_morphs(r["question"])))
            pg, an = grams(c["text"]), norm(r["answer"])
            ans_nums = {t.replace(",", "") for t in NUM.findall(r["answer"])
                        if len(t.replace(",", "").replace(".", "")) >= 3
                        and not (len(t) == 4 and t.isdigit() and 2000 <= int(t) <= 2035)}
            for j in order[:50]:
                cnd = cand.iloc[int(j)]
                cg = grams(cnd.text)
                if pg and cg and len(pg & cg) / len(pg | cg) >= 0.8:
                    skipped_neg["near_duplicate"] += 1
                    continue
                if len(an) >= 6 and an in norm(cnd.text):
                    skipped_neg["contains_answer"] += 1
                    continue
                if ans_nums and ans_nums & {t.replace(",", "") for t in NUM.findall(cnd.text)}:
                    skipped_neg["contains_answer_number"] += 1
                    continue
                neg = cnd
                break
        qt, pt = set(content_morphs(r["question"])), set(pool_tok[c["chunk_id"]])
        pairs.append({"pair_id": f"r2-{i + 1:04d}", "anchor": r["question"], "positive": c["text"],
                      "positive_id": c["chunk_id"], "negative": None if neg is None else neg.text,
                      "negative_id": None if neg is None else neg.chunk_id,
                      "negative_fiscal_year": None if neg is None else int(neg.fiscal_year),
                      "corp_name": c["corp_name"], "industry": c["industry"], "fiscal_year": c["fiscal_year"],
                      "section_path": c["section_path"], "value_table": c["value_table"], "answer": r["answer"],
                      "is_numeric": r["is_numeric"], "number_check": r.get("number_check"),
                      "lexical_jaccard": round(jaccard(qt, pt), 4),
                      "generated_by": "llm:" + cfg["model"].split("/")[-1], "review": None})
    with open(TRAIN / "pairs_r2.jsonl", "w", encoding="utf-8") as f:
        for p in pairs:
            f.write(json.dumps(p, ensure_ascii=False) + "\n")
    with open(TRAIN / "pairs_r2_dropped.jsonl", "w", encoding="utf-8") as f:
        for d in dropped:
            f.write(json.dumps(d, ensure_ascii=False) + "\n")

    fdb = load(ROOT / "data/raw/findartbench/data.jsonl").drop_duplicates("ctx_id")
    fdb_j = [jaccard(set(content_morphs(q)), set(content_morphs(c))) for q, c in zip(fdb.question, fdb.context)]
    pdf = pd.DataFrame(pairs)
    stats = {"generation": {k: v for k, v in cfg.items() if "prompt" not in k and k != "rule4_variants"},
             "chunks": len(gens), "stages": dict(stage),
             "keep_rate": round(len(kept) / stage["generated"], 3),
             "pairs": len(pairs), "numeric_share": round(float(pdf.is_numeric.mean()), 3),
             "from_value_table_share": round(float(pdf.value_table.mean()), 3),
             "number_check_routes": pdf.number_check.value_counts().to_dict(),
             "with_hard_negative": int(pdf.negative_id.notna().sum()), "hard_negative_skips": dict(skipped_neg),
             "negative_year_offset": (pdf.negative_fiscal_year - pdf.fiscal_year).abs().value_counts().to_dict(),
             "lexical_jaccard_r2": describe(pdf.lexical_jaccard),
             "lexical_jaccard_findartbench_ref": describe(fdb_j),
             "by_industry": pdf.industry.value_counts().to_dict()}
    (OUT / "pairs_r2_stats.json").write_text(json.dumps(stats, ensure_ascii=False, indent=1, default=str))

    review = random.Random(SEED).sample(pairs, 30)
    lines = ["# 학습 쌍 재파일럿(R2, OpenDART 원문) 검수 30쌍", "",
             f"`data/train/pairs_r2.jsonl` {len(pairs)}쌍 중 seed {SEED}로 30쌍.", ""]
    for n, p in enumerate(review, 1):
        lines += [f"## {n}. `{p['pair_id']}` {p['corp_name']} FY{p['fiscal_year']} · {p['section_path']}", "",
                  f"- 질문: {p['anchor']}", f"- 모델 답: {p['answer']}",
                  f"- 숫자 질문: {p['is_numeric']} ({p['number_check']}), 값 있는 표 청크: {p['value_table']}, "
                  f"겹침(Jaccard): {p['lexical_jaccard']}",
                  f"- 본문: {p['positive'][:900].replace(chr(10), ' ')}",
                  f"- hard negative (FY{p['negative_fiscal_year']}): {(p['negative'] or '')[:250].replace(chr(10), ' ')}",
                  "- 판정: ", "- 메모: ", ""]
    (OUT / "pairs_r2_review_30.md").write_text("\n".join(lines))
    print(json.dumps(stats, ensure_ascii=False, indent=1, default=str))


if __name__ == "__main__":
    main()

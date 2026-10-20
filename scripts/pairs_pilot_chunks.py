"""Q6 step 1: pick source chunks for the training-pair pilot from Dartdoc (no LLM).

- 20 companies with FY2022-2024 사업보고서 in Dartdoc, mixed industries, none of the 10 evaluation
  companies (so training and evaluation companies are disjoint).
- Latest filing per company x fiscal year (same rule as dartdoc_stats.py), sections II and IV.
- Dartdoc chunks (<=2,000 chars) are re-split to <=512 KURE tokens (finsight.chunking, overlap 50);
  sub-chunks shorter than 150 tokens are skipped. 4 sub-chunks per company x year (seed 42).

Usage: python scripts/pairs_pilot_chunks.py
Outputs: data/train/pilot_source_chunks.jsonl (picked), data/train/pilot_pool_chunks.jsonl (all
re-split II/IV sub-chunks of these companies, used later for hard negatives)
"""
from __future__ import annotations

import json
import random
from pathlib import Path

import pandas as pd

from finsight.chunking import chunk_text
from finsight.metadata import parse_report_name

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw" / "dartdoc" / "data"
TRAIN = ROOT / "data" / "train"
YEARS = (2022, 2023, 2024)
EVAL_COMPANIES = {"삼성전자", "SK하이닉스", "LG전자", "현대자동차", "기아", "SK텔레콤",
                  "한국전력공사", "국민은행", "HMM", "두나무"}
PILOT = {
    "은행·금융지주": ["JB금융지주", "경남은행"],
    "카드·캐피탈": ["롯데카드", "현대캐피탈"],
    "보험": ["교보생명보험", "DB손해보험"],
    "에너지·화학": ["한국수력원자력", "한화에너지", "여천NCC", "에이치디현대오일뱅크", "코스모화학"],
    "운송·유통·플랫폼": ["아시아나항공", "컬리", "비바리퍼블리카"],
    "제조·전선": ["LS", "SK실트론"],
    "건설·엔지니어링": ["롯데건설", "삼성E&A", "현대엔지니어링"],
    "소프트웨어": ["셀바스AI"],
}
PER_COMPANY_YEAR = 4  # 3 gave 274 pairs with company+year (< 300 target)
MIN_TOKENS = 150
SEED = 42


def main() -> None:
    companies = {c: ind for ind, cs in PILOT.items() for c in cs}
    assert len(companies) == 20 and not set(companies) & EVAL_COMPANIES
    rows = []
    for f in sorted(RAW.glob("*.jsonl")):
        with f.open(encoding="utf-8") as fh:
            for line in fh:
                r = json.loads(line)
                if r["corp_name"] not in companies:
                    continue
                kind, fy, _, amended = parse_report_name(r["report_type"])
                if kind == "사업보고서" and fy in YEARS:
                    rows.append({**r, "fiscal_year": fy, "is_amended": amended, "chunk": int(r["chunk"])})
    df = pd.DataFrame(rows)
    latest = df.groupby(["corp_name", "fiscal_year"]).rcept_no.max().reset_index()
    df = df.merge(latest, on=["corp_name", "fiscal_year", "rcept_no"])

    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained("nlpai-lab/KURE-v1")

    def ntok(s: str) -> int:
        return len(tok(s, add_special_tokens=False)["input_ids"])

    pool = []
    for r in df.sort_values(["corp_name", "fiscal_year", "section", "chunk"]).itertuples():
        for j, piece in enumerate(chunk_text(r.text, ntok, 512, 50)):
            n = ntok(piece) + 2
            pool.append({"chunk_id": f"dd_{r.rcept_no}_{r.section}_{r.chunk:03d}_{j}", "corp_name": r.corp_name,
                         "industry": companies[r.corp_name], "fiscal_year": int(r.fiscal_year),
                         "rcept_no": r.rcept_no, "report_type": r.report_type, "section": r.section,
                         "n_tokens": n, "text": piece})
    TRAIN.mkdir(parents=True, exist_ok=True)
    with open(TRAIN / "pilot_pool_chunks.jsonl", "w", encoding="utf-8") as f:
        for p in pool:
            f.write(json.dumps(p, ensure_ascii=False) + "\n")

    rng = random.Random(SEED)
    picked = []
    pdf = pd.DataFrame(pool)
    for (corp, fy), sub in pdf[pdf.n_tokens >= MIN_TOKENS].groupby(["corp_name", "fiscal_year"]):
        idx = sorted(sub.index)
        rng.shuffle(idx)
        picked += [pool[i] for i in idx[:PER_COMPANY_YEAR]]
    with open(TRAIN / "pilot_source_chunks.jsonl", "w", encoding="utf-8") as f:
        for p in picked:
            f.write(json.dumps(p, ensure_ascii=False) + "\n")
    pk = pd.DataFrame(picked)
    print({"pool_subchunks": len(pool), "picked": len(picked), "companies": pk.corp_name.nunique(),
           "by_section": pk.section.value_counts().to_dict(), "by_year": pk.fiscal_year.value_counts().to_dict(),
           "tokens_mean": round(pk.n_tokens.mean(), 1)})


if __name__ == "__main__":
    main()

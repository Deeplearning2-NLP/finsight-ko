"""Template-generated numeric evaluation questions from the P2 ground-truth table (P3). No LLM.

Types:
  1 단일 수치     — one company, one fiscal year, one account
  2 연도 비교     — same company & account, two fiscal years; answer = change in KRW (+ rate when base > 0)
  3 기업 비교     — same-industry pair, same year & account, same fs_div; answer = difference (A - B)
Answers are computed in Python from data/processed/financials_a.csv (원 단위). split is left empty
(dev/test is assigned with the full evaluation set in week 9).
Year comparisons use each year's own report ("as originally reported"). When the later report restates
the earlier year (frmtrm_amount differs), the restated value and change are kept in answer_extra.

Usage: python scripts/make_numeric_questions.py
Outputs: data/eval/numeric_auto.jsonl, outputs/proposal/numeric_auto_stats.json
"""
from __future__ import annotations

import itertools
import json
from pathlib import Path

import pandas as pd

from finsight import opendart as od

ROOT = Path(__file__).resolve().parents[1]
PROC = ROOT / "data" / "processed"
EVAL = ROOT / "data" / "eval"
OUT = ROOT / "outputs" / "proposal"
PAIRS = [("삼성전자", "SK하이닉스"), ("현대자동차", "기아")]  # same industry; others are not compared
FS_LABEL = {"CFS": "연결", "OFS": "별도"}


def josa(word: str, with_batchim: str, without: str) -> str:
    """Korean particle by final consonant of the last Hangul syllable (은/는, 이/가 ...)."""
    ch = word[-1]
    if "가" <= ch <= "힣":
        return with_batchim if (ord(ch) - 0xAC00) % 28 else without
    return without


def src(r) -> dict:
    return {"corp": r.corp, "fiscal_year": int(r.fiscal_year), "rcept_no": str(r.rcept_no),
            "account_id": r.account_id, "account_nm_raw": r.account_nm_raw, "fs_div": r.fs_div}


def restated_prior(r_later) -> float | None:
    """Earlier-year value as shown (frmtrm) in the later year's report, from the cached API response."""
    js = od.fin_statements(str(r_later.corp_code), int(r_later.fiscal_year), "11011", r_later.fs_div)
    a = next((a for a in js.get("list", []) if a["account_id"] == r_later.account_id
              and a["sj_div"] == r_later.sj_div), None)
    if not a or not (a.get("frmtrm_amount") or "").strip():
        return None
    return float(a["frmtrm_amount"].replace(",", ""))


def main() -> None:
    EVAL.mkdir(parents=True, exist_ok=True)
    fin = pd.read_csv(PROC / "financials_a.csv", dtype={"rcept_no": str, "corp_code": str}).dropna(subset=["value_krw"])
    key = {(r.corp, int(r.fiscal_year), r.account_std): r for r in fin.itertuples()}
    qs = []

    def add(qtype, template, question, value, extra, sources):
        qs.append({"qid": f"num-{len(qs) + 1:04d}", "type": qtype, "template": template, "question": question,
                   "answer_value": value, "answer_unit": "원", "answer_extra": extra,
                   "source": sources, "generated_by": "template", "split": None})

    # 1. single value
    for (corp, y, acc), r in sorted(key.items()):
        fs = FS_LABEL[r.fs_div]
        add(1, "single_value",
            f"{corp}의 {y} 사업연도 {fs} 기준 {acc}{josa(acc, '은', '는')} 얼마인가?",
            r.value_krw, {}, [src(r)])

    # 2. year comparison (consecutive years and 2022 -> 2024)
    for corp, acc in sorted({(c, a) for c, _, a in key}):
        for y1, y2 in [(2022, 2023), (2023, 2024), (2022, 2024)]:
            a, b = key.get((corp, y1, acc)), key.get((corp, y2, acc))
            if a is None or b is None or a.fs_div != b.fs_div:
                continue
            delta = b.value_krw - a.value_krw
            rate = round(delta / a.value_krw * 100, 2) if a.value_krw > 0 else None
            restated = restated_prior(b) if y2 == y1 + 1 and b.account_id != "-표준계정코드 미사용-" else None
            if restated is not None and abs(restated - a.value_krw) < 0.5:
                restated = None
            add(2, "year_change",
                f"{corp}의 {FS_LABEL[a.fs_div]} 기준 {acc}{josa(acc, '은', '는')} {y1} 사업연도 대비 "
                f"{y2} 사업연도에 얼마나 변했는가? (증감액과 증감률)",
                delta, {"value_from": a.value_krw, "value_to": b.value_krw, "change_rate_pct": rate,
                        "direction": "증가" if delta > 0 else "감소" if delta < 0 else "변동 없음",
                        "note": None if rate is not None else "기준 연도 값이 0 이하라 증감률 없음 (적자→흑자 등)",
                        "restated_value_from": restated,
                        "restated_change": None if restated is None else b.value_krw - restated},
                [src(a), src(b)])

    # 3. company comparison within an industry pair
    for (c1, c2), y, acc in itertools.product(PAIRS, [2022, 2023, 2024], sorted({a for *_, a in key})):
        a, b = key.get((c1, y, acc)), key.get((c2, y, acc))
        if a is None or b is None or a.fs_div != b.fs_div:
            continue
        diff = a.value_krw - b.value_krw
        add(3, "company_compare",
            f"{y} 사업연도 {FS_LABEL[a.fs_div]} 기준 {acc}{josa(acc, '은', '는')} {c1}{josa(c1, '과', '와')} "
            f"{c2} 중 어디가 더 크며, 차이는 얼마인가?",
            abs(diff), {"value_a": a.value_krw, "value_b": b.value_krw, "larger": c1 if diff > 0 else c2,
                        "signed_diff_a_minus_b": diff},
            [src(a), src(b)])

    with open(EVAL / "numeric_auto.jsonl", "w", encoding="utf-8") as f:
        for q in qs:
            f.write(json.dumps(q, ensure_ascii=False) + "\n")
    df = pd.DataFrame(qs)
    stats = {"n": len(qs), "by_type": df.type.value_counts().sort_index().to_dict(),
             "by_template": df.template.value_counts().to_dict(),
             "no_rate_year_change": int(sum(1 for q in qs if q["type"] == 2 and q["answer_extra"]["change_rate_pct"] is None)),
             "year_change_with_restated_prior": int(sum(1 for q in qs if q["type"] == 2 and q["answer_extra"]["restated_value_from"] is not None)),
             "examples": [next(q for q in qs if q["type"] == t) for t in (1, 2, 3)]}
    (OUT / "numeric_auto_stats.json").write_text(json.dumps(stats, ensure_ascii=False, indent=1))
    print(json.dumps(stats, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()

"""Template-generated numeric evaluation questions from the P2 ground-truth table (P3). No LLM.

Types:
  1 단일 수치     — one company, one fiscal year, one account
  2 연도 비교     — same company & account, two fiscal years; answer = change in KRW (+ rate when base > 0)
  3 기업 비교     — same-industry pair, same year & account, same fs_div; answer = difference (A - B)
Answers are computed in Python from data/processed/financials_a.csv (원 단위). split is left empty
(dev/test is assigned with the full evaluation set in week 9).
Restatements (docs/decisions.md 2026-10-20):
  - 연도 비교: default answer = the later year's report pair (thstrm vs frmtrm, or bfefrmtrm for a
    two-year gap) — the company's own like-for-like comparison. The change computed from the
    originally reported values goes to answer_alt when it differs.
  - 단일 수치: default = that year's own report; restated values shown in later reports go to answer_alt.
  - graded by finsight.grading (default or any answer_alt within 0.5%).

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


def later_report_value(r_later, years_back: int) -> float | None:
    """Value of the year `years_back` before r_later's year, as shown in r_later's own report
    (frmtrm for 1, bfefrmtrm for 2), from the cached fnlttSinglAcntAll response."""
    if r_later.account_id == "-표준계정코드 미사용-":
        return None
    js = od.fin_statements(str(r_later.corp_code), int(r_later.fiscal_year), "11011", r_later.fs_div)
    a = next((a for a in js.get("list", []) if a["account_id"] == r_later.account_id
              and a["sj_div"] == r_later.sj_div), None)
    field = {1: "frmtrm_amount", 2: "bfefrmtrm_amount"}[years_back]
    if not a or not (a.get(field) or "").strip():
        return None
    return float(a[field].replace(",", ""))


def differs(a: float | None, b: float) -> bool:
    return a is not None and abs(a - b) >= 0.5


def change_rate(frm: float, to: float) -> float | None:
    return round((to - frm) / frm * 100, 2) if frm > 0 else None


def main() -> None:
    EVAL.mkdir(parents=True, exist_ok=True)
    fin = pd.read_csv(PROC / "financials_a.csv", dtype={"rcept_no": str, "corp_code": str}).dropna(subset=["value_krw"])
    key = {(r.corp, int(r.fiscal_year), r.account_std): r for r in fin.itertuples()}
    qs = []

    def add(qtype, template, question, value, extra, sources, answer_alt=None):
        qs.append({"qid": f"num-{len(qs) + 1:04d}", "type": qtype, "template": template, "question": question,
                   "answer_value": value, "answer_alt": answer_alt, "answer_unit": "원", "answer_extra": extra,
                   "source": sources, "generated_by": "template", "split": None})

    # 1. single value: own report; restated values from later reports are accepted too
    for (corp, y, acc), r in sorted(key.items()):
        fs = FS_LABEL[r.fs_div]
        alts = []
        for back in (1, 2):
            later = key.get((corp, y + back, acc))
            if later is not None and later.fs_div == r.fs_div:
                v = later_report_value(later, back)
                if differs(v, r.value_krw) and all(differs(v, x) for x in alts):
                    alts.append(v)
        add(1, "single_value",
            f"{corp}의 {y} 사업연도 {fs} 기준 {acc}{josa(acc, '은', '는')} 얼마인가?",
            r.value_krw, {"restated_in_later_report": bool(alts)}, [src(r)], answer_alt=alts or None)

    # 2. year comparison: default = the later report's own pair
    for corp, acc in sorted({(c, a) for c, _, a in key}):
        for y1, y2 in [(2022, 2023), (2023, 2024), (2022, 2024)]:
            a, b = key.get((corp, y1, acc)), key.get((corp, y2, acc))
            if a is None or b is None or a.fs_div != b.fs_div:
                continue
            prior = later_report_value(b, y2 - y1)
            basis = "later_report_pair" if prior is not None else "original_values"
            frm = prior if prior is not None else a.value_krw
            delta = b.value_krw - frm
            orig_delta = b.value_krw - a.value_krw
            restated = differs(prior, a.value_krw)
            rate = change_rate(frm, b.value_krw)
            add(2, "year_change",
                f"{corp}의 {FS_LABEL[a.fs_div]} 기준 {acc}{josa(acc, '은', '는')} {y1} 사업연도 대비 "
                f"{y2} 사업연도에 얼마나 변했는가? (증감액과 증감률)",
                delta, {"basis": basis, "value_from": frm, "value_to": b.value_krw, "change_rate_pct": rate,
                        "direction": "증가" if delta > 0 else "감소" if delta < 0 else "변동 없음",
                        "note": None if rate is not None else "기준 연도 값이 0 이하라 증감률 없음 (적자→흑자 등)",
                        "restated": restated,
                        "original_value_from": a.value_krw,
                        "original_change_rate_pct": change_rate(a.value_krw, b.value_krw) if restated else None},
                [src(a), src(b)], answer_alt=[orig_delta] if restated else None)

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
             "with_answer_alt": {t: int(sum(1 for q in qs if q["type"] == t and q["answer_alt"])) for t in (1, 2, 3)},
             "year_change_basis": df[df.type == 2].answer_extra.map(lambda e: e["basis"]).value_counts().to_dict(),
             "restated_questions": [q for q in qs if q["answer_alt"]],
             "examples": [next(q for q in qs if q["type"] == t) for t in (1, 2, 3)]}
    (OUT / "numeric_auto_stats.json").write_text(json.dumps(stats, ensure_ascii=False, indent=1))
    print(json.dumps(stats, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()

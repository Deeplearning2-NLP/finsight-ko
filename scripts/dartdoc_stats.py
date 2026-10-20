"""Dartdoc coverage / length statistics for scoping (T2).

Steps: filter 사업보고서 -> parse fiscal_year / is_amended / filing_date ->
keep the latest filing per (company, fiscal_year) -> company x year pivots ->
candidate coverage -> scope proposal -> char/token length stats -> samples.

Usage: python scripts/dartdoc_stats.py [--tok-sample 0]
Outputs: outputs/proposal/dartdoc_coverage.csv, dartdoc_stats.json, dartdoc_scope.csv
"""
from __future__ import annotations

import argparse
import glob
import json
from pathlib import Path

import numpy as np
import pandas as pd

from finsight.metadata import filing_date_from_rcept_no, parse_report_name

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw" / "dartdoc" / "data"
OUT = ROOT / "outputs" / "proposal"
YEARS = list(range(2020, 2025))
SCOPE_YEARS = [2022, 2023, 2024]

# FinDartBench company -> Dartdoc corp_name
FINDARTBENCH = {
    "삼성전자": "삼성전자", "SK하이닉스": "SK하이닉스", "LG전자": "LG전자",
    "현대자동차": "현대자동차", "기아": "기아", "SK텔레콤": "SK텔레콤",
    "한국전력": "한국전력공사", "국민은행": "국민은행", "HMM": "HMM", "두나무": "두나무",
}

# Hand-made industry map for the scope proposal (Dartdoc corp_name).
# Chosen from companies that have FY2022-2024 사업보고서 in Dartdoc (the FinDartBench
# large caps mostly do not; see findartbench_coverage).
INDUSTRY = {
    "은행·금융지주": ["JB금융지주", "농협금융지주", "한국금융지주", "경남은행", "한국씨티은행"],
    "카드·캐피탈": ["롯데카드", "현대카드", "하나카드", "현대캐피탈"],
    "보험": ["교보생명보험", "DB손해보험", "푸본현대생명보험"],
    "에너지·유틸리티": ["한국전력공사", "한국수력원자력", "한화에너지", "여천NCC", "에이치디현대오일뱅크"],
    "운송·플랫폼": ["HMM", "아시아나항공", "두나무", "컬리", "비바리퍼블리카"],
}


def load() -> pd.DataFrame:
    rows = []
    for f in sorted(glob.glob(str(RAW / "*.jsonl"))):
        with open(f, encoding="utf-8") as fh:
            for line in fh:
                r = json.loads(line)
                rows.append((r["rcept_no"], r["corp_name"], r["report_type"], r["section"],
                             int(r["chunk"]), r["text"]))
    return pd.DataFrame(rows, columns=["rcept_no", "corp_name", "report_type", "section", "chunk", "text"])


def describe(x: np.ndarray) -> dict:
    return {"mean": round(float(x.mean()), 1), "p50": int(np.percentile(x, 50)),
            "p90": int(np.percentile(x, 90)), "max": int(x.max())}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tok-sample", type=int, default=0, help="0 = tokenize all scope chunks")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    stats: dict = {}

    df = load()
    stats["total_chunks"] = len(df)
    parsed = df.report_type.map(parse_report_name)
    df["report_kind"] = parsed.map(lambda p: p[0])
    df["fiscal_year"] = parsed.map(lambda p: p[1])
    df["is_amended"] = parsed.map(lambda p: p[3])
    df["filing_date"] = df.rcept_no.map(filing_date_from_rcept_no)

    # 1. 사업보고서 only (excludes 유동화전문회사 / 회계법인 variants and 증권신고서)
    biz = df[df.report_kind == "사업보고서"].copy()
    stats["kind_counts_top"] = df.report_kind.value_counts().head(8).to_dict()
    stats["biz_chunks"] = len(biz)
    stats["biz_reports"] = biz.rcept_no.nunique()
    stats["biz_companies"] = biz.corp_name.nunique()
    stats["biz_amended_reports"] = int(biz.drop_duplicates("rcept_no").is_amended.sum())
    stats["filing_year_vs_fiscal_year"] = (
        biz.drop_duplicates("rcept_no")
        .assign(filing_year=lambda d: d.filing_date.map(lambda x: x.year))
        .groupby(["fiscal_year", "filing_year"]).size()
        .loc[lambda s: s.index.get_level_values(0) >= 2020].to_dict()
    )
    stats["filing_year_vs_fiscal_year"] = {f"FY{int(k[0])}->filed{k[1]}": int(v)
                                           for k, v in stats["filing_year_vs_fiscal_year"].items()}

    # 2. keep latest filing per (company, fiscal_year)
    reports = biz.drop_duplicates("rcept_no")[["rcept_no", "corp_name", "fiscal_year", "is_amended"]]
    dup = reports.groupby(["corp_name", "fiscal_year"]).size()
    stats["company_years_with_multiple_reports"] = int((dup > 1).sum())
    latest = reports.sort_values("rcept_no").groupby(["corp_name", "fiscal_year"]).tail(1)
    biz_d = biz[biz.rcept_no.isin(latest.rcept_no)].copy()
    stats["dedup"] = {"reports_before": len(reports), "reports_after": len(latest),
                      "chunks_before": len(biz), "chunks_after": len(biz_d),
                      "latest_is_amended": int(latest.is_amended.sum())}

    # 3. company x fiscal_year pivot (+ by section)
    b = biz_d[biz_d.fiscal_year.isin(YEARS)]
    piv = b.groupby(["corp_name", "fiscal_year"]).size().unstack(fill_value=0)
    piv.columns = [int(c) for c in piv.columns]
    sec = b.groupby(["corp_name", "section", "fiscal_year"]).size().unstack(["section", "fiscal_year"], fill_value=0)
    sec.columns = [f"{s}_{int(y)}" for s, y in sec.columns]
    cov = piv.join(sec).fillna(0).astype(int)
    cov["all_2022_2024"] = (piv[SCOPE_YEARS] > 0).all(axis=1) if set(SCOPE_YEARS) <= set(piv.columns) else False
    cov = cov.sort_values([2022, 2023, 2024], ascending=False)
    cov.to_csv(OUT / "dartdoc_coverage.csv", encoding="utf-8-sig")
    stats["companies_by_year"] = {int(y): int((piv[y] > 0).sum()) for y in piv.columns}
    stats["companies_all_2022_2024"] = int(cov.all_2022_2024.sum())
    stats["chunks_by_year"] = {int(y): int(piv[y].sum()) for y in piv.columns}
    stats["chunks_by_section"] = b.section.value_counts().to_dict()

    # 4. FinDartBench candidates
    cand = {}
    for fdb_name, dd in FINDARTBENCH.items():
        row = cov.loc[dd] if dd in cov.index else None
        cand[fdb_name] = {"dartdoc_name": dd,
                          **{str(y): int(row[y]) if row is not None and y in row else 0 for y in YEARS}}
    stats["findartbench_coverage"] = cand

    # 5. scope proposal
    scope = []
    for ind, names in INDUSTRY.items():
        for n in names:
            r = cov.loc[n] if n in cov.index else None
            counts = {y: (int(r[y]) if r is not None and y in r else 0) for y in SCOPE_YEARS}
            scope.append({"industry": ind, "corp_name": n, **counts,
                          "all_3_years": all(v > 0 for v in counts.values()),
                          "in_findartbench": n in FINDARTBENCH.values()})
    scope_df = pd.DataFrame(scope)
    scope_df.to_csv(OUT / "dartdoc_scope.csv", index=False, encoding="utf-8-sig")
    ok = scope_df[scope_df.all_3_years]
    stats["scope"] = {"companies": len(ok), "industries": ok.industry.nunique(),
                      "chunks": int(ok[SCOPE_YEARS].to_numpy().sum()),
                      "missing": scope_df[~scope_df.all_3_years].corp_name.tolist()}

    # 6. length distribution (scope = FY2022-2024 deduplicated 사업보고서)
    s = biz_d[biz_d.fiscal_year.isin(SCOPE_YEARS)]
    stats["scope_all_companies_chunks"] = len(s)
    stats["chars_all_biz_2022_2024"] = describe(s.text.str.len().to_numpy())
    sel = s[s.corp_name.isin(ok.corp_name)]
    stats["chars_scope"] = describe(sel.text.str.len().to_numpy())
    texts_all = s.text if not args.tok_sample else s.text.sample(args.tok_sample, random_state=args.seed)
    from transformers import AutoTokenizer
    stats["tokens"] = {}
    for tok_name in ["nlpai-lab/KURE-v1", "klue/bert-base"]:
        tok = AutoTokenizer.from_pretrained(tok_name)
        for label, texts in [("all_biz_2022_2024", texts_all), ("scope", sel.text)]:
            lens = np.array([len(x) for x in tok(texts.tolist(), add_special_tokens=True)["input_ids"]])
            stats["tokens"][f"{tok_name}|{label}"] = {**describe(lens), "n": len(lens),
                                                      "over_512": round(float((lens > 512).mean()), 4)}
        print(tok_name, "done", flush=True)

    # 7. samples
    rng = np.random.default_rng(args.seed)
    samples = []
    for n in ["한국전력공사", "JB금융지주", "교보생명보험", "컬리", "HMM"]:
        sub = sel[(sel.corp_name == n)]
        if len(sub) == 0:
            continue
        r = sub.iloc[int(rng.integers(len(sub)))]
        samples.append({"corp_name": n, "fiscal_year": int(r.fiscal_year), "section": r.section,
                        "rcept_no": r.rcept_no, "report_type": r.report_type, "chunk": int(r.chunk),
                        "chars": len(r.text), "head": r.text[:200]})
    stats["samples"] = samples

    with open(OUT / "dartdoc_stats.json", "w", encoding="utf-8") as f:
        json.dump(stats, f, ensure_ascii=False, indent=1, default=str)
    print(json.dumps({k: v for k, v in stats.items() if k != "samples"}, ensure_ascii=False, indent=1, default=str))


if __name__ == "__main__":
    main()

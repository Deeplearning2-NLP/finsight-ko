"""Numeric ground-truth table for scope A (P2).

For each company x fiscal year: fnlttSinglAcntAll (사업보고서 11011), CFS first; status 013 -> OFS
and fs_div recorded. Six standard accounts are mapped by account_id (IFRS / DART taxonomy) only;
an account that does not map stays empty (e.g. banks have no ifrs-full_Revenue) — never forced.

Usage: python scripts/build_financials.py
Outputs: data/processed/financials_a.csv, outputs/proposal/financials_a_stats.json
"""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from finsight import opendart as od

ROOT = Path(__file__).resolve().parents[1]
PROC = ROOT / "data" / "processed"
OUT = ROOT / "outputs" / "proposal"

# account_std -> (account_ids in priority order, statements in priority order, exact-name fallback)
# The name fallback is used only for rows that carry no standard code ("-표준계정코드 미사용-").
ACCOUNTS = {
    "매출액": (("ifrs-full_Revenue",), ("IS", "CIS"), ()),
    "영업이익": (("dart_OperatingIncomeLoss", "ifrs-full_ProfitLossFromOperatingActivities"), ("IS", "CIS"), ()),
    "당기순이익": (("ifrs-full_ProfitLoss",), ("IS", "CIS"), ("당기순이익", "당기순이익(손실)", "연결당기순이익")),
    "자산총계": (("ifrs-full_Assets",), ("BS",), ()),
    "부채총계": (("ifrs-full_Liabilities",), ("BS",), ()),
    "자본총계": (("ifrs-full_Equity",), ("BS",), ()),
}
NO_CODE = "-표준계정코드 미사용-"


def find_account(items: list[dict], ids, sjs, names) -> tuple[dict | None, str | None]:
    for acc_id in ids:
        for sj in sjs:
            hit = next((a for a in items if a["sj_div"] == sj and a["account_id"] == acc_id), None)
            if hit:
                return hit, "account_id"
    for sj in sjs:
        hit = next((a for a in items if a["sj_div"] == sj and a["account_id"] == NO_CODE
                    and a["account_nm"] in names), None)
        if hit:
            return hit, "name (no standard code)"
    return None, None


def to_krw(s: str) -> float | None:
    s = (s or "").replace(",", "").strip()
    if s in ("", "-"):
        return None
    return float(s)


def main() -> None:
    filings = pd.read_csv(PROC / "filings_a.csv", dtype={"corp_code": str, "rcept_no": str})
    rows, calls, log = [], [], []
    for f in filings.itertuples():
        fs_div = "CFS"
        js = od.fin_statements(f.corp_code, int(f.fiscal_year), "11011", "CFS")
        if js.get("status") == "013":
            fs_div = "OFS"
            js = od.fin_statements(f.corp_code, int(f.fiscal_year), "11011", "OFS")
        calls.append({"corp_name": f.corp_name, "fiscal_year": int(f.fiscal_year), "fs_div": fs_div,
                      "status": js.get("status"), "n_accounts": len(js.get("list", []))})
        items = js.get("list", [])
        api_rcept = {a["rcept_no"] for a in items}
        if items and api_rcept != {f.rcept_no}:
            log.append(f"{f.corp_name} FY{f.fiscal_year}: API rcept_no {sorted(api_rcept)} != corpus {f.rcept_no}")
        for std, (ids, sjs, names) in ACCOUNTS.items():
            hit, method = find_account(items, ids, sjs, names)
            rows.append({"corp": f.corp_name, "corp_code": f.corp_code, "fiscal_year": int(f.fiscal_year),
                         "fs_div": fs_div, "account_std": std,
                         "account_nm_raw": hit["account_nm"] if hit else None,
                         "account_id": hit["account_id"] if hit else None,
                         "sj_div": hit["sj_div"] if hit else None, "match_method": method,
                         "value_krw": to_krw(hit["thstrm_amount"]) if hit else None,
                         "rcept_no": hit["rcept_no"] if hit else f.rcept_no})
    df = pd.DataFrame(rows)
    df.to_csv(PROC / "financials_a.csv", index=False, encoding="utf-8-sig")

    filled = df.assign(ok=df.value_krw.notna())
    cov = filled.pivot_table(index="corp", columns="account_std", values="ok", aggfunc="sum")[list(ACCOUNTS)]
    raw_names = (df.dropna(subset=["account_nm_raw"]).groupby("account_std").account_nm_raw
                 .agg(lambda s: sorted(set(s))).to_dict())
    stats = {"cells": len(df), "filled": int(filled.ok.sum()),
             "coverage_corp_x_account_years": cov.astype(int).to_dict(orient="index"),
             "raw_account_names": raw_names, "calls": calls, "rcept_mismatch": log,
             "match_method": df.match_method.value_counts().to_dict(),
             "name_matched": df[df.match_method == "name (no standard code)"][["corp", "fiscal_year", "account_std", "account_nm_raw"]].to_dict(orient="records"),
             "missing": filled[~filled.ok][["corp", "fiscal_year", "account_std"]].to_dict(orient="records")}
    (OUT / "financials_a_stats.json").write_text(json.dumps(stats, ensure_ascii=False, indent=1))
    print(cov.astype(int).to_string())
    print(json.dumps({k: stats[k] for k in ("cells", "filled", "match_method", "name_matched", "missing", "rcept_mismatch")},
                     ensure_ascii=False, indent=1))
    print("fs_div:", pd.DataFrame(calls).fs_div.value_counts().to_dict())


if __name__ == "__main__":
    main()

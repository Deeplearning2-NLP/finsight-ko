"""Build the scope-A evaluation corpus from OpenDART originals (P1).

10 FinDartBench companies x fiscal years 2022-2024 사업보고서:
  list.json -> fiscal year from the report name "(YYYY.MM)" (finsight.metadata), latest filing per year
  -> document.xml -> finsight.dart_parse blocks -> Markdown tables (header repeated on split)
  -> <=512-token chunks (KURE-v1 tokenizer, overlap 50, final re-count).
All API calls are cached by finsight.opendart.

Training companies (R2): `--set train` builds the same corpus for the 20 non-evaluation companies of the
pair pilot (scripts/pairs_pilot_chunks.PILOT); the script asserts that the two company sets are disjoint.

Usage: python scripts/build_corpus.py [--set eval|train]
Outputs: data/processed/corpus_<a|train>.jsonl, filings_<a|train>.csv, outputs/proposal/corpus_<a|train>_stats.json
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd

from finsight import opendart as od
from finsight.dart_parse import chunk_units, parse_document, units_from_blocks
from finsight.metadata import filing_date_from_rcept_no, parse_report_name

ROOT = Path(__file__).resolve().parents[1]
PROC = ROOT / "data" / "processed"
OUT = ROOT / "outputs" / "proposal"
COMPANIES = ["삼성전자", "SK하이닉스", "LG전자", "현대자동차", "기아", "SK텔레콤",
             "한국전력공사", "국민은행", "HMM", "두나무"]
YEARS = [2022, 2023, 2024]


def train_companies() -> list[str]:
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from pairs_pilot_chunks import PILOT
    comps = [c for cs in PILOT.values() for c in cs]
    assert len(comps) == 20 and not set(comps) & set(COMPANIES), "training companies overlap evaluation companies"
    return comps
CORE_SECTIONS = ("II.", "IV.")


def pick_filings(corp: str, corp_code: str) -> tuple[list[dict], list[dict]]:
    """Latest 사업보고서 per fiscal year in YEARS, plus every candidate seen (for the log)."""
    seen = od.list_filings(corp_code, f"{YEARS[0]}0101", f"{YEARS[-1] + 1}1231")
    cands = []
    for f in seen:
        kind, fy, month, amended = parse_report_name(f["report_nm"])
        if kind != "사업보고서" or fy not in YEARS:
            continue
        cands.append({"corp_name": corp, "corp_code": corp_code, "rcept_no": f["rcept_no"],
                      "report_nm": f["report_nm"], "fiscal_year": fy, "period_month": month,
                      "is_amended": amended, "filing_date": str(filing_date_from_rcept_no(f["rcept_no"]))})
    latest = {}
    for c in sorted(cands, key=lambda c: c["rcept_no"]):
        latest[c["fiscal_year"]] = c
    return [latest[y] for y in YEARS if y in latest], cands


def describe(x) -> dict:
    x = np.asarray(x)
    return {"n": int(len(x)), "mean": round(float(x.mean()), 1), "p50": int(np.percentile(x, 50)),
            "p90": int(np.percentile(x, 90)), "max": int(x.max()), "over_512": int((x > 512).sum())}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tokenizer", default="nlpai-lab/KURE-v1")
    ap.add_argument("--set", default="eval", choices=["eval", "train"])
    args = ap.parse_args()
    PROC.mkdir(parents=True, exist_ok=True)
    OUT.mkdir(parents=True, exist_ok=True)
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(args.tokenizer)

    def ntok(s: str) -> int:
        return len(tok(s, add_special_tokens=False)["input_ids"])

    codes = od.corp_codes()
    filings, all_cands, failures = [], [], []
    companies = COMPANIES if args.set == "eval" else train_companies()
    tag = "a" if args.set == "eval" else "train"
    for corp in companies:
        if corp not in codes:
            failures.append({"corp_name": corp, "reason": "corpCode에 없음"})
            continue
        picked, cands = pick_filings(corp, codes[corp]["corp_code"])
        if not picked:  # same-name entries: try the other corp codes (현대엔지니어링)
            for alt in od.corp_code_candidates(corp)[1:]:
                picked, cands = pick_filings(corp, alt["corp_code"])
                if picked:
                    break
        all_cands += cands
        filings += picked
        for y in sorted(set(YEARS) - {p["fiscal_year"] for p in picked}):
            failures.append({"corp_name": corp, "fiscal_year": y, "reason": "list.json에 사업보고서 없음"})

    rows = []
    per_report = []
    with open(PROC / f"corpus_{tag}.jsonl", "w", encoding="utf-8") as fout:
        for f in filings:
            try:
                files = od.document(f["rcept_no"])
            except Exception as e:  # keep going; record the failure
                failures.append({**{k: f[k] for k in ("corp_name", "fiscal_year", "rcept_no")}, "reason": str(e)[:200]})
                continue
            main_xml = files.get(f"{f['rcept_no']}.xml") or max(files.values(), key=len)
            blocks = parse_document(main_xml)
            # front/back matter (【 대표이사 등의 확인 】 etc.) is often image-only, so compare main sections
            raw_titles = {re.sub(r"\s+", " ", t.decode("utf-8", "replace").strip()) for t in
                          re.findall(rb"<SECTION-1[^>]*>\s*<TITLE[^>]*>(.*?)</TITLE>", main_xml, re.S)}
            missing = sorted(t for t in raw_titles - {b["sec1"] for b in blocks} if not t.startswith("【"))
            chunks = chunk_units(units_from_blocks(blocks, "markdown", ntok), ntok, 512, 50)
            n_core = 0
            for i, c in enumerate(chunks):
                core = c["sec"].startswith(CORE_SECTIONS)
                n_core += core
                rec = {"chunk_id": f"{f['rcept_no']}_{i:04d}", "corp_code": f["corp_code"],
                       "corp_name": f["corp_name"], "fiscal_year": f["fiscal_year"], "rcept_no": f["rcept_no"],
                       "filing_date": f["filing_date"], "is_amended": f["is_amended"],
                       "section_path": c["sec"], "is_core_section": core, "is_table": c["is_table"],
                       "n_tokens": c["ntok"], "text": c["text"]}
                fout.write(json.dumps(rec, ensure_ascii=False) + "\n")
                rows.append({k: v for k, v in rec.items() if k != "text"})
            per_report.append({**{k: f[k] for k in ("corp_name", "fiscal_year", "rcept_no", "report_nm")},
                               "blocks_text": sum(b["type"] == "text" for b in blocks),
                               "tables": sum(b["type"] == "table" for b in blocks),
                               "chunks": len(chunks), "chunks_II_IV": n_core,
                               "missing_sections": missing})
            print(per_report[-1], flush=True)

    pd.DataFrame(filings).to_csv(PROC / f"filings_{tag}.csv", index=False, encoding="utf-8-sig")
    df = pd.DataFrame(rows)
    core = df[df.is_core_section]
    stats = {
        "reports": len(per_report), "expected": len(companies) * len(YEARS),
        "failures": failures,
        "amended_latest": [f"{f['corp_name']} FY{f['fiscal_year']} {f['report_nm']}" for f in filings if f["is_amended"]],
        "superseded_filings": len(all_cands) - len(filings),
        "missing_sections": {f"{r['corp_name']} FY{r['fiscal_year']}": r["missing_sections"]
                             for r in per_report if r["missing_sections"]},
        "non_december": sorted({f"{c['corp_name']} ({c['fiscal_year']}.{c['period_month']:02d})"
                                for c in all_cands if c["period_month"] != 12}),
        "chunks_all": len(df), "chunks_II_IV": len(core),
        "table_chunk_ratio_all": round(float(df.is_table.mean()), 4),
        "table_chunk_ratio_II_IV": round(float(core.is_table.mean()), 4),
        "tokens_all": describe(df.n_tokens), "tokens_II_IV": describe(core.n_tokens),
        "chunks_by_corp_year_all": {f"{k[0]}|{k[1]}": int(v) for k, v in df.groupby(["corp_name", "fiscal_year"]).size().items()},
        "chunks_by_corp_year_II_IV": {f"{k[0]}|{k[1]}": int(v) for k, v in core.groupby(["corp_name", "fiscal_year"]).size().items()},
        "per_report": per_report,
    }
    (OUT / f"corpus_{tag}_stats.json").write_text(json.dumps(stats, ensure_ascii=False, indent=1))
    print(json.dumps({k: v for k, v in stats.items() if not k.startswith(("chunks_by", "per_report"))},
                     ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()

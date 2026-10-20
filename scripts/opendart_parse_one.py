"""Parse and chunk one OpenDART 사업보고서 (tables included) (T3).

1. corpCode.xml -> corp_code, list.json -> FY rcept_no (latest filing)
2. document.xml -> sections (SECTION-1/2 titles), body text vs tables
3. table -> text: (a) Markdown (b) "row | column | value" lines; token counts
4. chunk to <=512 tokens (paragraph / sentence / table-row boundaries, overlap 50)
5. fnlttSinglAcntAll (CFS, 11011) 매출액·영업이익 vs the 요약재무정보 table parsed with finsight.numbers

Usage: python scripts/opendart_parse_one.py [--corp 삼성전자 --year 2024]
Outputs: outputs/proposal/opendart_<corp>_<year>.json (+ _chunks.jsonl)
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from finsight import opendart as od
from finsight.dart_parse import (chunk_units, find_row_value, parse_document, table_markdown,
                                 table_triples, units_from_blocks)
from finsight.numbers import parse_krw, rel_close

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs" / "proposal"


def describe(x) -> dict:
    x = np.asarray(x)
    return {"n": int(len(x)), "mean": round(float(x.mean()), 1), "p50": int(np.percentile(x, 50)),
            "p90": int(np.percentile(x, 90)), "max": int(x.max()), "over_512": int((x > 512).sum())}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--corp", default="삼성전자")
    ap.add_argument("--year", type=int, default=2024)
    ap.add_argument("--tokenizer", default="nlpai-lab/KURE-v1")
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(args.tokenizer)

    def ntok(s: str) -> int:
        return len(tok(s, add_special_tokens=False)["input_ids"])

    res: dict = {"corp": args.corp, "fiscal_year": args.year, "tokenizer": args.tokenizer}
    cc = od.corp_codes()[args.corp]
    res["corp_code"] = cc["corp_code"]
    filings = od.list_filings(cc["corp_code"], f"{args.year + 1}0101", f"{args.year + 1}1231")
    filings = [f for f in filings if f"({args.year}.12)" in f["report_nm"]]
    filing = max(filings, key=lambda f: f["rcept_no"])  # latest (handles 정정)
    res["rcept_no"], res["report_nm"] = filing["rcept_no"], filing["report_nm"]

    files = od.document(filing["rcept_no"])
    main_xml = files.get(f"{filing['rcept_no']}.xml") or max(files.values(), key=len)
    res["document_files"] = {k: len(v) for k, v in files.items()}
    blocks = parse_document(main_xml)
    res["blocks"] = {"text": sum(b["type"] == "text" for b in blocks),
                     "table": sum(b["type"] == "table" for b in blocks)}
    by_sec: dict[str, dict] = {}
    for b in blocks:
        d = by_sec.setdefault(b["sec1"], {"text_blocks": 0, "tables": 0, "chars_text": 0})
        if b["type"] == "text":
            d["text_blocks"] += 1
            d["chars_text"] += len(b["text"])
        else:
            d["tables"] += 1
    res["sections"] = by_sec

    # table -> text comparison on the II/IV sections (same sections as Dartdoc)
    target = [b for b in blocks if b["sec1"].startswith(("II.", "IV."))]
    tables = [b for b in target if b["type"] == "table"]
    md_tok = [ntok(table_markdown(b)) for b in tables]
    tr_tok = [ntok("\n".join(table_triples(b))) for b in tables]
    res["table_text_tokens_II_IV"] = {"n_tables": len(tables), "markdown": describe(md_tok),
                                      "triples": describe(tr_tok),
                                      "markdown_total": int(sum(md_tok)), "triples_total": int(sum(tr_tok))}
    body_tok = sum(ntok(b["text"]) for b in target if b["type"] == "text")
    res["body_text_tokens_II_IV"] = int(body_tok)

    # chunking, both table modes, II/IV and whole document
    res["chunks"] = {}
    for mode in ("markdown", "triples"):
        for scope, bl in (("II_IV", target), ("all", blocks)):
            ch = chunk_units(units_from_blocks(bl, mode, ntok), ntok)
            res["chunks"][f"{mode}|{scope}"] = describe([c["ntok"] for c in ch])
            if scope == "II_IV":
                with open(OUT / f"opendart_{args.corp}_{args.year}_{mode}_chunks.jsonl", "w", encoding="utf-8") as f:
                    for i, c in enumerate(ch):
                        f.write(json.dumps({"chunk_id": f"{filing['rcept_no']}_{mode}_{i}", **c},
                                           ensure_ascii=False) + "\n")

    # examples for slides: one text section and one table, original vs converted
    ex_tab = next((b for b in tables if 4 <= len(b["rows"]) <= 8 and len(b["rows"][0]) >= 3), tables[0])
    res["example_table"] = {"sec": f"{ex_tab['sec1']} > {ex_tab['sec2']}", "unit": ex_tab.get("unit"),
                            "rows": ex_tab["rows"], "markdown": table_markdown(ex_tab),
                            "triples": table_triples(ex_tab)}
    ex_txt = next(b for b in target if b["type"] == "text" and len(b["text"]) > 150)
    res["example_text"] = {"sec": f"{ex_txt['sec1']} > {ex_txt['sec2']}", "text": ex_txt["text"][:400]}

    # OpenDART official values vs parsed 요약재무정보
    fs = od.fin_statements(cc["corp_code"], args.year, "11011", "CFS")
    res["fnltt_status"] = fs.get("status")
    checks = []
    targets = {"매출액": ["매출액", "수익(매출액)", "영업수익"], "영업이익": ["영업이익", "영업이익(손실)"]}
    for label, names in targets.items():
        acc = next((a for a in fs.get("list", []) if a["sj_div"] in ("IS", "CIS") and a["account_nm"] in names), None)
        raw, unit, head = None, None, None
        for nm in names:
            raw, unit, head = find_row_value(blocks, "요약재무정보", nm)
            if raw:
                break
        api_val = float(acc["thstrm_amount"].replace(",", "")) if acc else None
        parsed = parse_krw(raw, unit=unit) if raw else None
        checks.append({"item": label, "api_account_nm": acc and acc["account_nm"], "api_value_krw": api_val,
                       "table_raw": raw, "table_unit": unit, "table_header": head, "parsed_krw": parsed,
                       "match": bool(api_val and parsed and rel_close(api_val, parsed, tol=0.0005))})
    res["verification"] = checks

    with open(OUT / f"opendart_{args.corp}_{args.year}.json", "w", encoding="utf-8") as f:
        json.dump(res, f, ensure_ascii=False, indent=1)
    show = {k: v for k, v in res.items() if k not in ("example_table", "example_text")}
    print(json.dumps(show, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()

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
import re
from pathlib import Path

import numpy as np
from lxml import etree

from finsight import opendart as od
from finsight.numbers import parse_krw, rel_close

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs" / "proposal"
CELL_TAGS = {"TD", "TH", "TE", "TU"}
SENT_SPLIT = re.compile(r"(?<=[다요음함됨임\.])\s+(?=\S)")
UNIT_RE = re.compile(r"단위\s*[:：]\s*([^\s\),%]+)")


def clean(s: str) -> str:
    return re.sub(r"\s+", " ", s.replace("\xa0", " ")).strip()


def text_of(el) -> str:
    return clean("".join(el.itertext()))


# ---------------------------------------------------------------- parsing
def own_rows(table):
    """TR elements of this table, excluding rows of nested tables."""
    return [tr for tr in table.iter("TR")
            if next(a for a in tr.iterancestors() if a.tag == "TABLE") is table]


def is_layout_table(table) -> bool:
    """Tables used as page layout (e.g. financial notes): nested tables or long prose cells."""
    if table.find(".//TABLE") is not None:
        return True
    return any(len(text_of(c)) > 400 for tr in own_rows(table) for c in tr if c.tag in CELL_TAGS)


def table_rows(table) -> tuple[list[list[str]], int]:
    """Expand rowspan/colspan into a rectangular grid. Returns (rows, n_header_rows)."""
    grid: list[list[str]] = []
    pending: dict[tuple[int, int], str] = {}
    n_head = 0
    for r, tr in enumerate(own_rows(table)):
        row: list[str] = []
        c = 0
        cells = [x for x in tr if x.tag in CELL_TAGS]
        if tr.getparent() is not None and tr.getparent().tag == "THEAD":
            n_head = r + 1
        for cell in cells:
            while (r, c) in pending:
                row.append(pending.pop((r, c)))
                c += 1
            t = text_of(cell)
            cs = int(cell.get("COLSPAN", 1) or 1)
            rs = int(cell.get("ROWSPAN", 1) or 1)
            for dc in range(cs):
                row.append(t)
                for dr in range(1, rs):
                    pending[(r + dr, c + dc)] = t
            c += cs
        while (r, c) in pending:
            row.append(pending.pop((r, c)))
            c += 1
        grid.append(row)
    width = max((len(r) for r in grid), default=0)
    grid = [r + [""] * (width - len(r)) for r in grid]
    if n_head == 0 and grid:
        n_head = 1
    return grid, n_head


def parse_document(xml: bytes) -> list[dict]:
    """Flat list of blocks: {sec1, sec2, type: text|table, text | rows, n_head, unit}."""
    root = etree.fromstring(xml, etree.XMLParser(recover=True, huge_tree=True))
    blocks: list[dict] = []

    def walk(el, sec1: str, sec2: str, last_unit: list[str | None]):
        for ch in el:
            tag = ch.tag if isinstance(ch.tag, str) else ""
            if tag == "SECTION-1":
                walk(ch, text_of(ch.find("TITLE")) if ch.find("TITLE") is not None else sec1, "", [None])
            elif tag in ("SECTION-2", "SECTION-3"):
                t = text_of(ch.find("TITLE")) if ch.find("TITLE") is not None else sec2
                walk(ch, sec1, t if tag == "SECTION-2" else f"{sec2} > {t}", [None])
            elif tag == "TITLE":
                continue
            elif tag == "P":
                t = text_of(ch)
                if t:
                    m = UNIT_RE.search(t)
                    if m:
                        last_unit[0] = m.group(1)
                    blocks.append({"sec1": sec1, "sec2": sec2, "type": "text", "text": t})
            elif tag == "TABLE" and is_layout_table(ch):
                for tr in own_rows(ch):
                    for cell in (c for c in tr if c.tag in CELL_TAGS):
                        if cell.find(".//TABLE") is not None or cell.find("P") is not None:
                            if (cell.text or "").strip():
                                blocks.append({"sec1": sec1, "sec2": sec2, "type": "text", "text": clean(cell.text)})
                            walk(cell, sec1, sec2, last_unit)
                        elif text_of(cell):
                            blocks.append({"sec1": sec1, "sec2": sec2, "type": "text", "text": text_of(cell)})
            elif tag == "TABLE":
                rows, n_head = table_rows(ch)
                flat = " ".join(" ".join(r) for r in rows)
                m = UNIT_RE.search(flat)
                # small unit-only tables ("(단위 : 백만원)") set the unit for the next table
                if m and len(rows) <= 1:
                    last_unit[0] = m.group(1)
                    continue
                if rows and any(any(c for c in r) for r in rows):
                    blocks.append({"sec1": sec1, "sec2": sec2, "type": "table", "rows": rows,
                                   # a preceding "(단위 : 백만원)" wins; in-table matches are often row notes
                                   "n_head": n_head, "unit": last_unit[0] or (m.group(1) if m else None)})
                    last_unit[0] = None
            else:
                walk(ch, sec1, sec2, last_unit)

    body = root.find(".//BODY")
    walk(body if body is not None else root, "", "", [None])
    return blocks


# ---------------------------------------------------------------- table -> text
def dedup_row(r: list[str]) -> list[str]:
    """Collapse colspan duplicates (adjacent equal cells)."""
    out: list[str] = []
    for c in r:
        if not out or c != out[-1]:
            out.append(c)
    return out


def table_markdown(b: dict) -> str:
    rows = b["rows"]
    h = b["n_head"]
    head = [" / ".join(dict.fromkeys(x for x in col if x)) for col in zip(*rows[:h])] if h else rows[0]
    lines = ([f"(단위: {b['unit']})"] if b.get("unit") else [])
    lines.append("| " + " | ".join(head) + " |")
    lines.append("|" + "---|" * len(head))
    lines += ["| " + " | ".join(r) + " |" for r in rows[h:]]
    return "\n".join(lines)


def table_triples(b: dict) -> list[str]:
    """One line per value cell: '행이름 | 열이름 | 값'. Row name = leading label cells."""
    rows = b["rows"]
    h = b["n_head"]
    cols = [" ".join(dict.fromkeys(x for x in col if x)) for col in zip(*rows[:h])]
    unit = f" ({b['unit']})" if b.get("unit") else ""
    lines = []
    for r in rows[h:]:
        # label cells = leading cells that do not look numeric
        k = 0
        while k < len(r) - 1 and not re.search(r"\d", r[k].replace("제", "")):
            k += 1
        k = max(k, 1)
        row_name = " ".join(dict.fromkeys(x for x in r[:k] if x))
        seen = set()
        for j in range(k, len(r)):
            v = r[j]
            if not v or v in ("-",) or (cols[j], v) in seen:
                continue
            seen.add((cols[j], v))
            lines.append(f"{row_name} | {cols[j]} | {v}{'' if v.endswith('%') else unit}")
    return lines


# ---------------------------------------------------------------- chunking
def units_from_blocks(blocks: list[dict], table_mode: str, ntok) -> list[dict]:
    """Atomic units (sentence or table line) with token counts; tables carry a header for repeats."""
    units = []
    for b in blocks:
        sec = f"{b['sec1']} > {b['sec2']}" if b["sec2"] else b["sec1"]
        if b["type"] == "text":
            parts = [b["text"]] if ntok(b["text"]) <= 200 else SENT_SPLIT.split(b["text"])
            units += [{"sec": sec, "text": p, "table_head": None} for p in parts if p]
        else:
            if table_mode == "markdown":
                md = table_markdown(b).split("\n")
                n = 3 if md[0].startswith("(단위") else 2
                head = "\n".join(md[:n])
                if ntok(head) > 150:  # do not repeat very wide headers in every chunk
                    head = None
                units += [{"sec": sec, "text": line, "table_head": head} for line in md[n:]]
                if head is None or len(md) == n:
                    units.insert(len(units) - (len(md) - n), {"sec": sec, "text": "\n".join(md[:n]), "table_head": None})
            else:
                units += [{"sec": sec, "text": line, "table_head": None} for line in table_triples(b)]
    for u in units:
        u["ntok"] = ntok(u["text"])
    return units


def chunk_units(units: list[dict], ntok, max_tok: int = 512, overlap: int = 50) -> list[dict]:
    """Greedy pack within a section; carry the last <=overlap tokens of units into the next chunk."""
    budget = max_tok - 2  # [CLS]/[SEP]
    chunks, cur, cur_tok, cur_sec = [], [], 0, None

    def flush():
        if cur:
            body = "\n".join(u["text"] for u in cur)
            text = f"[{cur_sec}]\n{body}"
            chunks.append({"sec": cur_sec, "text": text, "ntok": ntok(text) + 2})

    for u in units:
        head_tok = ntok(u["table_head"]) if u["table_head"] else 0
        if u["sec"] != cur_sec:
            flush()
            cur, cur_tok, cur_sec = [], ntok(f"[{u['sec']}]"), u["sec"]
        if cur and cur_tok + u["ntok"] > budget:
            flush()
            carry, t = [], 0
            for prev in reversed(cur):
                if t + prev["ntok"] > overlap:
                    break
                carry.insert(0, prev)
                t += prev["ntok"]
            cur = carry
            cur_tok = ntok(f"[{cur_sec}]") + t
            if u["table_head"] and not any(c["text"] == u["table_head"] for c in cur):
                cur.insert(0, {"text": u["table_head"], "ntok": head_tok})
                cur_tok += head_tok
            if cur_tok + u["ntok"] > budget:  # overlap would not fit: start clean
                cur = [{"text": u["table_head"], "ntok": head_tok}] if u["table_head"] else []
                cur_tok = ntok(f"[{cur_sec}]") + (head_tok if u["table_head"] else 0)
        if not cur and u["table_head"]:
            cur.append({"text": u["table_head"], "ntok": head_tok})
            cur_tok += head_tok
        if u["ntok"] > budget:  # very long sentence: hard split by characters
            step = max(50, int(len(u["text"]) * budget / u["ntok"] * 0.9))
            for i in range(0, len(u["text"]), step):
                piece = u["text"][i:i + step]
                cur.append({"text": piece, "ntok": ntok(piece)})
                flush()
                cur, cur_tok = [], ntok(f"[{cur_sec}]")
            continue
        cur.append(u)
        cur_tok += u["ntok"]
    flush()
    return chunks


def describe(x) -> dict:
    x = np.asarray(x)
    return {"n": int(len(x)), "mean": round(float(x.mean()), 1), "p50": int(np.percentile(x, 50)),
            "p90": int(np.percentile(x, 90)), "max": int(x.max()), "over_512": int((x > 512).sum())}


# ---------------------------------------------------------------- verification
def find_row_value(blocks: list[dict], sec2_kw: str, row_kw: str):
    """First table in a sub-section whose first cell matches row_kw -> (raw current-year value, unit)."""
    for b in blocks:
        if b["type"] != "table" or sec2_kw not in b["sec2"]:
            continue
        for r in b["rows"][b["n_head"]:]:
            if re.sub(r"\s", "", r[0]) == row_kw:
                vals = [c for c in r[1:] if re.search(r"\d", c)]
                if vals:
                    return vals[0], b.get("unit"), b["rows"][: b["n_head"]]
    return None, None, None


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

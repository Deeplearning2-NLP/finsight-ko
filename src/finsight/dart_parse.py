"""Parse OpenDART document.xml (사업보고서 원문) into text/table blocks and token-bounded chunks.

- Sections: SECTION-1/2/3 + TITLE. Body: P. Tables: TABLE/TR/TD·TH·TE·TU (ROWSPAN/COLSPAN expanded).
- Layout tables (financial notes: nested tables or long prose cells) are unpacked into text.
- The unit comes from the preceding "(단위 : 백만원)" line/table; an in-table "단위" note is
  only a fallback (it is often a per-row note such as 주당이익 "(단위 : 원)").
- Chunks: <= max_tok tokens incl. 2 special tokens, overlap, section path prefix, table header
  repeated when a table is split, and a final re-count so no chunk exceeds max_tok.
"""
from __future__ import annotations

import re
from typing import Callable

from lxml import etree

CELL_TAGS = {"TD", "TH", "TE", "TU"}
SENT_SPLIT = re.compile(r"(?<=[다요음함됨임\.])\s+(?=\S)")
UNIT_RE = re.compile(r"단위\s*[:：]\s*([^\s\),%]+)")

NTok = Callable[[str], int]


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


# "<금융자산>" etc. appear unescaped in some filings; lxml's recovery then nests every later
# SECTION inside that bogus element. Escape "<" unless it starts a real (ASCII) tag.
# A real tag name is ASCII and ends with whitespace, "/" or ">"; "<JB금융지주>" (JB금융지주 FY2022-2024)
# starts with ASCII letters but is text, so the name has to be checked to its end.
_BARE_LT = re.compile(rb"<(?![?!]|/?[A-Za-z][A-Za-z0-9_.:\-]*[\s/>])")
_BARE_AMP = re.compile(rb"&(?!#?\w+;)")
# English-translation attributes sometimes contain stray quotes (현대자동차·기아 FY2024:
# ENG="Other receivables"" / ENG="　"Proceeds from ..."") which break the tag; we never use them.
_ENG_ATTR = re.compile(rb'\sENG="[^<>]*?"(?=\s+[A-Z][A-Z0-9_-]*=|\s*/?>)')


def sanitize_xml(xml: bytes) -> bytes:
    """Repair the malformations seen in 사업보고서 XML before lxml's recovery drops whole sections."""
    xml = _BARE_LT.sub(b"&lt;", xml)
    xml = _BARE_AMP.sub(b"&amp;", xml)
    return _ENG_ATTR.sub(b"", xml)


def parse_document(xml: bytes) -> list[dict]:
    """Flat list of blocks: {sec1, sec2, type: text|table, text | rows, n_head, unit}."""
    root = etree.fromstring(sanitize_xml(xml), etree.XMLParser(recover=True, huge_tree=True))
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
    """One line per value cell: '행이름 | 열이름 | 값'. Row name = first column only."""
    rows = b["rows"]
    h = b["n_head"]
    cols = [" ".join(dict.fromkeys(x for x in col if x)) for col in zip(*rows[:h])]
    unit = f" ({b['unit']})" if b.get("unit") else ""
    lines = []
    for r in rows[h:]:
        row_name = r[0]
        seen = set()
        for j in range(1, len(r)):
            v = r[j]
            if not v or v == "-" or v == row_name or (cols[j], v) in seen:
                continue
            seen.add((cols[j], v))
            lines.append(f"{row_name} | {cols[j]} | {v}{'' if v.endswith('%') else unit}")
    return lines


# ---------------------------------------------------------------- chunking
def section_path(b: dict) -> str:
    return f"{b['sec1']} > {b['sec2']}" if b["sec2"] else b["sec1"]


def units_from_blocks(blocks: list[dict], table_mode: str, ntok: NTok) -> list[dict]:
    """Atomic units (sentence or table line) with token counts; tables carry a header for repeats."""
    units = []
    for b in blocks:
        sec = section_path(b)
        if b["type"] == "text":
            parts = [b["text"]] if ntok(b["text"]) <= 200 else SENT_SPLIT.split(b["text"])
            units += [{"sec": sec, "text": p, "table_head": None, "is_table": False} for p in parts if p]
        elif table_mode == "markdown":
            md = table_markdown(b).split("\n")
            n = 3 if md[0].startswith("(단위") else 2
            head = "\n".join(md[:n])
            if ntok(head) > 150:  # do not repeat very wide headers in every chunk
                head = None
            body = [{"sec": sec, "text": line, "table_head": head, "is_table": True} for line in md[n:]]
            if head is None or not body:
                body.insert(0, {"sec": sec, "text": "\n".join(md[:n]), "table_head": None, "is_table": True})
            units += body
        else:
            units += [{"sec": sec, "text": line, "table_head": None, "is_table": True} for line in table_triples(b)]
    for u in units:
        u["ntok"] = ntok(u["text"])
    return units


def _render(sec: str, parts: list[dict]) -> str:
    return f"[{sec}]\n" + "\n".join(p["text"] for p in parts)


def _fit(sec: str, parts: list[dict], ntok: NTok, max_tok: int) -> list[list[dict]]:
    """Final re-count: split a chunk in halves until every piece fits (token counts are not additive)."""
    if ntok(_render(sec, parts)) + 2 <= max_tok:
        return [parts]
    if len(parts) == 1:
        t = parts[0]["text"]
        mid = len(t) // 2
        return (_fit(sec, [{**parts[0], "text": t[:mid]}], ntok, max_tok)
                + _fit(sec, [{**parts[0], "text": t[mid:]}], ntok, max_tok))
    mid = len(parts) // 2
    return _fit(sec, parts[:mid], ntok, max_tok) + _fit(sec, parts[mid:], ntok, max_tok)


def chunk_units(units: list[dict], ntok: NTok, max_tok: int = 512, overlap: int = 50) -> list[dict]:
    """Greedy pack within a section; carry the last <=overlap tokens of units into the next chunk.

    Returns [{sec, text, ntok, is_table}] with ntok counted on the final text incl. 2 special tokens.
    """
    budget = max_tok - 2
    groups: list[tuple[str, list[dict]]] = []
    cur: list[dict] = []
    cur_tok, cur_sec = 0, None

    def flush():
        if cur:
            groups.append((cur_sec, list(cur)))

    for u in units:
        head_tok = ntok(u["table_head"]) if u["table_head"] else 0
        head_unit = {"text": u["table_head"], "ntok": head_tok, "is_table": True} if u["table_head"] else None
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
            if head_unit and not any(c["text"] == u["table_head"] for c in cur):
                cur.insert(0, head_unit)
                cur_tok += head_tok
            if cur_tok + u["ntok"] > budget:  # overlap would not fit: start clean
                cur = [head_unit] if head_unit else []
                cur_tok = ntok(f"[{cur_sec}]") + (head_tok if head_unit else 0)
        if not cur and head_unit:
            cur.append(head_unit)
            cur_tok += head_tok
        if u["ntok"] > budget:  # very long sentence: hard split by characters
            step = max(50, int(len(u["text"]) * budget / u["ntok"] * 0.9))
            for i in range(0, len(u["text"]), step):
                cur.append({**u, "text": u["text"][i:i + step]})
                flush()
                cur = []
            cur_tok = ntok(f"[{cur_sec}]")
            continue
        cur.append(u)
        cur_tok += u["ntok"]
    flush()

    chunks = []
    for sec, parts in groups:
        for piece in _fit(sec, parts, ntok, max_tok):
            text = _render(sec, piece)
            chunks.append({"sec": sec, "text": text, "ntok": ntok(text) + 2,
                           "is_table": any(p.get("is_table") for p in piece)})
    return chunks


# ---------------------------------------------------------------- verification helper
def find_row_value(blocks: list[dict], sec2_kw: str, row_kw: str):
    """First table in a sub-section whose first cell matches row_kw -> (raw current-year value, unit, header)."""
    for b in blocks:
        if b["type"] != "table" or sec2_kw not in b["sec2"]:
            continue
        for r in b["rows"][b["n_head"]:]:
            if re.sub(r"\s", "", r[0]) == row_kw:
                vals = [c for c in r[1:] if re.search(r"\d", c)]
                if vals:
                    return vals[0], b.get("unit"), b["rows"][: b["n_head"]]
    return None, None, None

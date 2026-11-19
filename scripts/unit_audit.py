"""G1: table-unit audit of corpus_a and a unit-fixed copy with the SAME chunk boundaries and ids.

Parses the 30 cached document.xml files twice: as built (parse_document) and with unit_fix=True (R1 statement
header -> next table, R2 ignore body-row unit notes in big tables; src/finsight/dart_parse.py). Chunk boundaries are
taken from the original parse (re-run of chunk_units, checked to reproduce corpus_a text exactly), then each chunk is
re-rendered with the fixed unit line for every table it contains:
  A  unit changed   : the chunk showed "(단위: X)" but the table's unit is Y (e.g. 원 -> 백만원)
  B  unit was missing and is now known (no "(단위" line; the fix finds it)
  C  unit known but not shown in this chunk (continuation of a table whose header was too wide to repeat) -> inserted
  D  numeric table with no unit even after the fix (count only)
Chunk ids, order and boundaries do not change, so gold labels, splits and FinDartBench mapping stay valid.

Usage: python scripts/unit_audit.py
Two versions: AB (fix A and B only) and ABC (also insert the known unit for C).
Outputs: data/processed/corpus_a_unitfix_{AB,ABC}.jsonl, outputs/midterm/unit_audit.json (counts + affected chunk ids)
"""
import json
import re
import sys
from collections import Counter
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from finsight import opendart as od  # noqa: E402
from finsight.dart_parse import _fit, _render, parse_document, units_from_blocks  # noqa: E402

NUM = re.compile(r"\d{1,3}(?:,\d{3})+")


def chunk_pieces(units, ntok, max_tok=512, overlap=50):
    """Copy of dart_parse.chunk_units that returns the parts of each chunk (head units keep their block index)."""
    budget = max_tok - 2
    groups, cur, cur_tok, cur_sec = [], [], 0, None

    def flush():
        if cur:
            groups.append((cur_sec, list(cur)))

    for u in units:
        head_tok = ntok(u["table_head"]) if u["table_head"] else 0
        head_unit = {"text": u["table_head"], "ntok": head_tok, "is_table": True, "bi": u["bi"], "is_head": True} \
            if u["table_head"] else None
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
            if cur_tok + u["ntok"] > budget:
                cur = [head_unit] if head_unit else []
                cur_tok = ntok(f"[{cur_sec}]") + (head_tok if head_unit else 0)
        if not cur and head_unit:
            cur.append(head_unit)
            cur_tok += head_tok
        if u["ntok"] > budget:
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
    out = []
    for sec, parts in groups:
        out += [(sec, piece) for piece in _fit(sec, parts, ntok, max_tok)]
    return out


def tagged_units(blocks, ntok):
    us = []
    for bi, b in enumerate(blocks):
        for u in units_from_blocks([b], "markdown", ntok):
            u["bi"] = bi
            us.append(u)
    return us


def main():
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained("nlpai-lab/KURE-v1")

    def ntok(s):
        return len(tok(s, add_special_tokens=False)["input_ids"])

    corpus = pd.read_json(ROOT / "data/processed/corpus_a.jsonl", lines=True, dtype={"chunk_id": str, "rcept_no": str})
    by_r = {r: g.sort_values("chunk_id") for r, g in corpus.groupby("rcept_no")}
    cats, affected, examples = Counter(), {k: [] for k in "ABCD"}, []
    new_rows, blocks_changed, mismatch = {"AB": [], "ABC": []}, Counter(), 0
    for rcept, g in by_r.items():
        files = od.document(rcept)  # cached; no API call
        xml = files.get(f"{rcept}.xml") or max(files.values(), key=len)
        old_b, new_b = parse_document(xml), parse_document(xml, unit_fix=True)
        assert len(old_b) == len(new_b)
        for a, b in zip(old_b, new_b):
            if a["type"] == "table" and a["unit"] != b["unit"]:
                blocks_changed[f"{a['unit']} -> {b['unit']}"] += 1
        pieces = chunk_pieces(tagged_units(old_b, ntok), ntok)
        texts = [_render(sec, p) for sec, p in pieces]
        if len(texts) != len(g) or any(t != o for t, o in zip(texts, g.text)):
            mismatch += 1
            print("cannot reproduce", rcept, len(texts), len(g), flush=True)
            for rec in g.to_dict("records"):
                new_rows["AB"].append(rec)
                new_rows["ABC"].append(rec)
            continue
        for (sec, parts), rec in zip(pieces, g.to_dict("records")):
          for mode in ("AB", "ABC"):
            lines, seen, flags = [], set(), set()
            for p in parts:
                bi = p.get("bi")
                if bi is None or old_b[bi]["type"] != "table":
                    lines.append(p["text"])
                    continue
                ou, nu = old_b[bi]["unit"], new_b[bi]["unit"]
                t = p["text"]
                has_line = t.startswith("(단위")
                if has_line:  # head part (or a wide head shown once): replace the unit line
                    body = t.split("\n", 1)[1] if "\n" in t else ""
                    t = (f"(단위: {nu})\n{body}" if nu else body) if body else (f"(단위: {nu})" if nu else "")
                    if ou != nu:
                        flags.add("A" if ou and nu else ("B" if nu else "A"))
                    seen.add(bi)
                elif bi not in seen:  # first part of this table in the chunk, no unit line yet
                    seen.add(bi)
                    if nu:
                        flags.add("B" if not ou else "C")
                        if not ou or mode == "ABC":
                            t = f"(단위: {nu})\n{t}"
                    elif NUM.search(t):
                        flags.add("D")
                if t:
                    lines.append(t)
            new_text = f"[{sec}]\n" + "\n".join(lines)
            if mode == "ABC":
                for f in flags:
                    cats[f] += 1
                    affected[f].append(rec["chunk_id"])
                if flags & {"A", "B"} and len(examples) < 8:
                    examples.append({"chunk_id": rec["chunk_id"], "flags": sorted(flags), "old": rec["text"][:300], "new": new_text[:300]})
            new_rows[mode].append({**rec, "text": new_text, "n_tokens": ntok(new_text) + 2})
    old_text = dict(zip(corpus.chunk_id, corpus.text))
    stats = {}
    pos = {cid: i for i, cid in enumerate(corpus.chunk_id)}  # keep the original file order (embedding rows)
    for mode, rows in new_rows.items():
        rows.sort(key=lambda r: pos[r["chunk_id"]])
        with open(ROOT / f"data/processed/corpus_a_unitfix_{mode}.jsonl", "w", encoding="utf-8") as f:
            for r in rows:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        changed = [r["chunk_id"] for r in rows if r["text"] != old_text[r["chunk_id"]]]
        stats[mode] = {"chunks_text_changed": len(changed),
                       "chunks_over_512_after": sum(r["n_tokens"] > 512 for r in rows),
                       "changed_ids": changed}
    stats["before"] = {"chunks_over_512": int((corpus.n_tokens > 512).sum())}
    out = {"chunks": len(corpus), "table_chunks": int(corpus.is_table.sum()), "reports_not_reproduced": mismatch,
           "chunk_counts": {"A_unit_changed": cats["A"], "B_unit_was_missing": cats["B"],
                            "C_unit_not_shown": cats["C"], "D_numeric_no_unit_after_fix": cats["D"]},
           "versions": stats, "table_blocks_unit_changed": dict(blocks_changed), "examples": examples,
           "affected": affected}
    (ROOT / "outputs/midterm/unit_audit.json").write_text(json.dumps(out, ensure_ascii=False, indent=1))
    print(json.dumps({k: (v if k != "versions" else {m: {kk: vv for kk, vv in x.items() if kk != "changed_ids"} for m, x in v.items()})
                      for k, v in out.items() if k not in ("affected", "examples")}, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()

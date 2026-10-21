"""R2 step 1: source chunks for the training-pair re-pilot from OpenDART originals of the 20 training companies.

- data/processed/corpus_train.jsonl (build_corpus.py --set train), sections II and IV (same as the Q6 pilot).
- "value table" chunk = has table rows and >= 5 numbers with 3+ digits; only these may get a numeric question
  (Q6: 9 of 12 errors were numeric questions on chunks whose numbers had been stripped).
- Chunks under 150 tokens are skipped; N per company x fiscal year at random (seed 42).
- Asserts the training companies are disjoint from the 10 evaluation companies.

Usage: python scripts/pairs_r2_chunks.py [--per 6]
Output: data/train/r2_source_chunks.jsonl
"""
from __future__ import annotations

import argparse
import json
import random
import re
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from build_corpus import COMPANIES as EVAL_COMPANIES  # noqa: E402
from pairs_pilot_chunks import PILOT  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
TRAIN = ROOT / "data" / "train"
NUM3 = re.compile(r"\d{1,3}(?:,\d{3})+|\d{3,}(?:\.\d+)?")
SEED = 42


def value_table(text: str, is_table: bool) -> bool:
    return bool(is_table) and len(NUM3.findall(text)) >= 5


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--per", type=int, default=6)
    args = ap.parse_args()
    industry = {c: ind for ind, cs in PILOT.items() for c in cs}
    assert len(industry) == 20 and not set(industry) & set(EVAL_COMPANIES), "train/eval company overlap"
    c = pd.read_json(ROOT / "data/processed/corpus_train.jsonl", lines=True, dtype={"rcept_no": str, "chunk_id": str})
    assert set(c.corp_name) <= set(industry) and not set(c.corp_name) & set(EVAL_COMPANIES)
    c = c[c.is_core_section & (c.n_tokens >= 150)].copy()
    c["value_table"] = [value_table(t, it) for t, it in zip(c.text, c.is_table)]
    rng = random.Random(SEED)
    picked = []
    for (corp, fy), sub in c.groupby(["corp_name", "fiscal_year"]):
        idx = sorted(sub.index)
        rng.shuffle(idx)
        for i in idx[:args.per]:
            r = c.loc[i]
            picked.append({"chunk_id": r.chunk_id, "corp_name": corp, "industry": industry[corp], "fiscal_year": int(fy),
                           "rcept_no": r.rcept_no, "section_path": r.section_path, "is_table": bool(r.is_table),
                           "value_table": bool(r.value_table), "n_tokens": int(r.n_tokens), "text": r.text})
    TRAIN.mkdir(parents=True, exist_ok=True)
    with open(TRAIN / "r2_source_chunks.jsonl", "w", encoding="utf-8") as f:
        for p in picked:
            f.write(json.dumps(p, ensure_ascii=False) + "\n")
    pk = pd.DataFrame(picked)
    print({"core_chunks": len(c), "core_value_table_share": round(float(c.value_table.mean()), 3), "picked": len(pk),
           "picked_value_table": int(pk.value_table.sum()), "companies": pk.corp_name.nunique(),
           "by_year": pk.fiscal_year.value_counts().sort_index().to_dict()})


if __name__ == "__main__":
    main()

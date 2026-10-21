"""Derive company -> (fiscal_year - 기수) offsets from the covers of the scope-A originals (R1).

Every report cover says "(제 N 기)"; fiscal_year - N is constant per company. The result is
copied into finsight.query_meta.PERIOD_OFFSET (checked by tests/test_query_meta.py).

Usage: python scripts/build_period_map.py
"""
from __future__ import annotations

import re
from pathlib import Path

import pandas as pd

from finsight import opendart as od

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    filings = pd.read_csv(ROOT / "data/processed/filings_a.csv", dtype=str)
    offsets: dict[str, set[int]] = {}
    for r in filings.itertuples():
        files = od.document(r.rcept_no)
        xml = (files.get(f"{r.rcept_no}.xml") or max(files.values(), key=len)).decode("utf-8", "replace")
        cover = xml[xml.find("<COVER"):xml.find("</COVER>")]
        m = re.search(r"제\s*(\d+)\s*기", cover)
        if not m:
            print("no period on cover:", r.corp_name, r.fiscal_year)
            continue
        offsets.setdefault(r.corp_name, set()).add(int(r.fiscal_year) - int(m.group(1)))
    for corp, offs in offsets.items():
        assert len(offs) == 1, (corp, offs)
        print(f'    "{corp}": {offs.pop()},')


if __name__ == "__main__":
    main()

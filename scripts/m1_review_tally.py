"""Tally the human review sheet of the M1 training pairs (outputs/midterm/pairs_m1_review_200.md).

Counts the "- 판정:" labels (답가능 / 베끼기 / 모호 / 오류), overall and by question format, and the share of
unjudged items. Usage: python scripts/m1_review_tally.py
"""
from __future__ import annotations

import json
import re
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SHEET = ROOT / "outputs/midterm/pairs_m1_review_200.md"
LABELS = ("답가능", "베끼기", "모호", "오류")


def main() -> None:
    text = SHEET.read_text()
    items = re.split(r"(?m)^## \d+\. ", text)[1:]
    total, by_fmt = Counter(), defaultdict(Counter)
    for it in items:
        fmt = re.search(r"형식 (\S+)", it).group(1)
        m = re.search(r"- 판정:\s*\**([^\s*]*)", it)
        lab = next((l for l in LABELS if m and l in m.group(1).replace(" ", "")), "미판정")
        total[lab] += 1
        by_fmt[fmt][lab] += 1
    judged = sum(v for k, v in total.items() if k != "미판정")
    usable = total["답가능"] + total["베끼기"]
    out = {"items": len(items), "judged": judged, "counts": dict(total),
           "usable_share_of_judged": round(usable / judged, 3) if judged else None,
           "error_share_of_judged": round(total["오류"] / judged, 3) if judged else None,
           "by_format": {k: dict(v) for k, v in by_fmt.items()}}
    (ROOT / "outputs/midterm/pairs_m1_review_tally.json").write_text(json.dumps(out, ensure_ascii=False, indent=1))
    print(json.dumps(out, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()

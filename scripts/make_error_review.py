"""G5: team-review copy of the 30-error analysis (outputs/midterm/error_analysis_30.md -> docs/deliverables/).
FinDartBench is CC BY-NC with no redistribution, so its question text (the "## N." headings) and the quoted
passages are cut to the first 80 characters. Adds "팀 판정(동의/수정)" and "메모" lines to every item.
Usage: python scripts/make_error_review.py"""
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
src = (ROOT / "outputs/midterm/error_analysis_30.md").read_text(encoding="utf-8")


def cut(s, n=80):
    return s if len(s) <= n else s[:n] + "…"


out, n_items = [], 0
for line in src.split("\n"):
    m = re.match(r"^## (\d+)\. (.*)$", line)
    if m:
        if n_items:
            out += ["- **팀 판정(동의/수정):** ", "- **메모:** ", ""]
        n_items += 1
        out.append(f"## {m.group(1)}. {cut(m.group(2))}")
        continue
    q = re.match(r"^(\s*> )(.*)$", line)
    if q:
        out.append(q.group(1) + cut(q.group(2)))
        continue
    out.append(line)
out += ["- **팀 판정(동의/수정):** ", "- **메모:** ", ""]
head = ["> 팀 검토용 사본 (G5). FinDartBench(CC BY-NC 4.0, 재배포 금지) 질문과 인용 문단은 앞 80자만 남겼다. "
        "전체는 서버 `outputs/midterm/error_analysis_30.md`. 각 항목의 '유형'과 '왜 틀렸나'를 읽고 "
        "**팀 판정**에 동의 또는 수정(수정이면 메모에 맞는 유형·이유)을 적는다.", ""]
lines = out[:2] + head + out[2:] if out and out[0].startswith("# ") else head + out
(ROOT / "docs/deliverables/error_analysis_30.md").write_text("\n".join(lines), encoding="utf-8")
print("items", n_items)

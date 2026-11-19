"""F2: tally the filled human review sheets (scripts/make_review_sheets.py) -> markdown for the report / slides.

Training pairs (99 unique, 15 judged twice): label shares with Wilson 95% CI. An overlap pair counts once, with the
label of its "own" reviewer. "사용 가능" = 답가능 + 베끼기 (the question is answerable; 베끼기 = copies the chunk's
wording). Cohen's kappa (4 labels) and observed agreement on the 15 overlap pairs. Breakdown by question format.
Eval set (30 items x 2 reviewers): verdict shares per reviewer, kappa / agreement on the verdict (3 labels) and on
each Y/N check, breakdown by type and by origin (LLM draft vs Claude proxy item; joined from evalset_v1.jsonl).
Unfilled rows are reported as 미판정 and left out of rates.

--consensus: the sheets were filled by joint (consensus) judgement, so overlapping sheets are identical by
construction; kappa / observed agreement are not printed (they would be 1.00 and are not an independent agreement).

Usage: python scripts/tally_human_review.py [--dir docs/deliverables] [--out docs/deliverables/review_tally.md] [--consensus]
"""
import argparse
import csv
import json
import math
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PAIR_LABELS = ["답가능", "베끼기", "모호", "오류"]
EVAL_LABELS = ["통과", "수정", "폐기"]


def read(path):
    with open(path, encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def norm(v, labels):
    v = (v or "").replace(" ", "").strip()
    for l in labels:
        if l in v:
            return l
    if labels == ["Y", "N"]:
        u = v.upper()
        if u in ("Y", "YES", "O", "예", "네", "1", "TRUE"):
            return "Y"
        if u in ("N", "NO", "X", "아니오", "아니요", "0", "FALSE"):
            return "N"
    return None


def wilson(k, n, z=1.96):
    if n == 0:
        return (float("nan"),) * 3
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return p, max(0, c - h), min(1, c + h)


def kappa(a, b, labels):
    pairs = [(x, y) for x, y in zip(a, b) if x and y]
    n = len(pairs)
    if n == 0:
        return None, None, 0
    po = sum(x == y for x, y in pairs) / n
    pe = sum((sum(x == l for x, _ in pairs) / n) * (sum(y == l for _, y in pairs) / n) for l in labels)
    k = (po - pe) / (1 - pe) if pe < 1 else 1.0
    return k, po, n


def fmt_rate(k, n):
    p, lo, hi = wilson(k, n)
    return f"{k}/{n} = {p:.1%} [{lo:.1%}, {hi:.1%}]" if n else "–"


CONSENSUS = False
CONS_NOTE = "3인 합의 판정이라 독립 κ가 아니다 (같은 판정을 두 시트에 적었으므로 κ = 1.00은 의미 없음). 독립 κ는 13주차 새 문항에서 잰다."


def tally_pairs(d: Path) -> list[str]:
    key = json.loads((d / "review_key_pairs.json").read_text())
    lab, fmt = {}, {}
    for r in key["own"]:
        f = d / f"review_pairs_{r}.csv"
        if not f.exists():
            continue
        for row in read(f):
            lab[(r, row["pair_id"])] = norm(row["판정(답가능/베끼기/모호/오류)"], PAIR_LABELS)
            fmt[row["pair_id"]] = row["질문 형식"]
    out = ["## 학습 쌍 사람 검수 (99쌍, " + ("3인 합의 판정" if CONSENSUS else "3명") + ")", ""]
    final = {p: lab.get((r, p)) for r, ps in key["own"].items() for p in ps}
    judged = {p: v for p, v in final.items() if v}
    n = len(judged)
    c = Counter(judged.values())
    out += [f"- 판정 완료 {n}/99쌍 (미판정 {99 - n})", "", "| 판정 | 비율 [95% CI] |", "|---|---|"]
    for l in PAIR_LABELS:
        out.append(f"| {l} | {fmt_rate(c[l], n)} |")
    out.append(f"| **사용 가능 (답가능+베끼기)** | {fmt_rate(c['답가능'] + c['베끼기'], n)} |")
    out.append(f"| **오류** | {fmt_rate(c['오류'], n)} |")
    a, b = [], []
    for r, ps in key["overlap"].items():  # r judged these too; owner is the next reviewer
        owners = {p: o for o, qs in key["own"].items() for p in qs}
        for p in ps:
            a.append(lab.get((owners[p], p)))
            b.append(lab.get((r, p)))
    k, po, m = kappa(a, b, PAIR_LABELS)
    if CONSENSUS:
        out += ["", f"- 일치도: {CONS_NOTE}"]
    else:
        out += ["", f"- 겹친 15쌍 일치도: " + (f"Cohen's κ = {k:.2f}, 일치율 p_o = {po:.1%} (두 명 모두 판정한 {m}쌍)" if m else "아직 없음")]
    groups = Counter()
    for p, v in judged.items():
        kind = "숫자 질문" if "숫자" in fmt.get(p, "") else "서술 질문"
        groups[(kind, v)] += 1
    out += ["", "| 질문 종류 | n | 사용 가능 | 오류 |", "|---|---|---|---|"]
    for kind in ("숫자 질문", "서술 질문"):
        nn = sum(v for (g, _), v in groups.items() if g == kind)
        out.append(f"| {kind} | {nn} | {fmt_rate(groups[(kind, '답가능')] + groups[(kind, '베끼기')], nn)} | {fmt_rate(groups[(kind, '오류')], nn)} |")
    return out


def tally_eval(d: Path) -> list[str]:
    key = json.loads((d / "review_key_eval.json").read_text())
    meta = {}
    ev = ROOT / "data/eval/evalset_v1.jsonl"
    if ev.exists():
        meta = {x["qid"]: x for x in map(json.loads, open(ev, encoding="utf-8"))}
    sheets = {}
    for r in ("백윤우", "박채영"):
        f = d / f"review_eval_{r}.csv"
        if f.exists():
            sheets[r] = {row["qid"]: row for row in read(f)}
    out = ["", "## 평가셋 사람 검수 (30문항, " + ("3인 합의 판정" if CONSENSUS else "2명 동일 문항") + ")", ""]
    if not meta:
        out += ["> 경고: `data/eval/evalset_v1.jsonl`이 없어 유형·출처별 표를 만들 수 없다 (데이터가 있는 서버에서 실행).", ""]
    cols = {"판정": ("판정(통과/수정/폐기)", EVAL_LABELS), "질문 명확": ("질문 명확(Y/N)", ["Y", "N"]),
            "정답 하나": ("정답 하나(Y/N)", ["Y", "N"]), "근거 맞음": ("근거 맞음(Y/N)", ["Y", "N"])}
    vals = {r: {q: {c: norm(row.get(col, ""), labs) for c, (col, labs) in cols.items()} for q, row in s.items()}
            for r, s in sheets.items()}
    out += ["| " + ("시트 (같은 합의 판정)" if CONSENSUS else "검수자") + " | 판정 완료 | 통과 | 수정 | 폐기 |", "|---|---|---|---|---|"]
    for r, v in vals.items():
        js = [x["판정"] for x in v.values() if x["판정"]]
        c = Counter(js)
        out.append(f"| {r} | {len(js)}/30 | " + " | ".join(fmt_rate(c[l], len(js)) for l in EVAL_LABELS) + " |")
    if CONSENSUS:
        out += ["", f"- 일치도: {CONS_NOTE}"]
    elif len(vals) == 2:
        A, B = vals["백윤우"], vals["박채영"]
        qs = key["qids"]
        out += ["", "| 항목 | Cohen's κ | 일치율 p_o | n |", "|---|---|---|---|"]
        for c, (_, labs) in cols.items():
            k, po, n = kappa([A.get(q, {}).get(c) for q in qs], [B.get(q, {}).get(c) for q in qs], labs)
            out.append(f"| {c} | {k:.2f} | {po:.1%} | {n} |" if n else f"| {c} | – | – | 0 |")
        out.append("")
        out.append("- κ ≥ 0.6이면 나머지 문항은 1명 검수로 충분(가이드라인 5절). 낮으면 불일치 문항을 같이 보고 기준을 고친다.")
    # verdict by type / origin: an item "passes" if every reviewer who judged it said 통과
    def agg(group_of):
        g = Counter()
        for q in key["qids"]:
            vs = [vals[r][q]["판정"] for r in vals if q in vals[r] and vals[r][q]["판정"]]
            if not vs:
                continue
            grp = group_of(q)
            g[(grp, "n")] += 1
            g[(grp, "통과")] += all(v == "통과" for v in vs)
            g[(grp, "폐기")] += any(v == "폐기" for v in vs)
        return g
    tname = {1: "단일 수치", 2: "연도 비교", 3: "기업 비교", 4: "서술 요약", 5: "답 없는 질문"}
    for title, fn, keys in (("유형별", lambda q: tname.get(meta.get(q, {}).get("type"), "?"), list(tname.values())),
                            ("출처별 (LLM 초안 vs Claude 대행 문항)",
                             lambda q: {"llm": "LLM 초안", "human_proxy": "Claude 대행"}.get(meta.get(q, {}).get("origin"), "?"),
                             ["LLM 초안", "Claude 대행"])):
        g = agg(fn)
        cap = "(통과 / 폐기)" if CONSENSUS else "(두 명 모두 통과 / 한 명이라도 폐기)"
        hdr = "| 그룹 | n | 통과 | 폐기 |" if CONSENSUS else "| 그룹 | n | 모두 통과 | 폐기 있음 |"
        out += ["", f"**{title}** {cap}", "", hdr, "|---|---|---|---|"]
        for k in keys:
            n = g[(k, "n")]
            out.append(f"| {k} | {n} | {fmt_rate(g[(k, '통과')], n)} | {fmt_rate(g[(k, '폐기')], n)} |")
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default=str(ROOT / "docs/deliverables"))
    ap.add_argument("--out", default=str(ROOT / "docs/deliverables/review_tally.md"))
    ap.add_argument("--consensus", action="store_true", help="joint judgement: do not report kappa")
    a = ap.parse_args()
    global CONSENSUS
    CONSENSUS = a.consensus
    d = Path(a.dir)
    lines = ["# 사람 검수 집계", "", "`scripts/tally_human_review.py" + (" --consensus" if CONSENSUS else "") +
             "`로 생성. 비율 옆 괄호는 Wilson 95% 신뢰구간." + (" **3인(임병건·백윤우·박채영) 합의 판정이며, 독립 κ가 아니다.**" if CONSENSUS else ""), ""]
    lines += tally_pairs(d) + tally_eval(d)
    Path(a.out).write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()

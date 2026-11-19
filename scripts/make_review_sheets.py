"""F2: human review sheets (CSV, UTF-8 with BOM, paste into Google Sheets) for the team, docs/midterm-tasks-final.md.

1) Training pairs: 99 of the 200 pairs in outputs/midterm/pairs_m1_review_200.md (seed 42), 33 per reviewer
   (임병건 / 백윤우 / 박채영). For Cohen's kappa, 15 pairs are judged twice: each sheet also gets 5 pairs from the next
   reviewer's 33 (A<-B, B<-C, C<-A), shuffled in, so each sheet has 38 rows. Label: 답가능 / 베끼기 / 모호 / 오류 + reason.
2) Eval set: 30 items of evalset v1 (dev+test, not discarded), 6 per type 1-5 (seed 42), the same items for
   백윤우 and 박채영 (shuffled differently). Checklist of docs/eval_guideline.md section 5.
Texts come from our OpenDART-parsed corpora only (no FinDartBench text: CC BY-NC, no redistribution).
The item origin (LLM / Claude proxy) is not shown on the sheet (blind); scripts/tally_human_review.py joins it.

Usage: python scripts/make_review_sheets.py   -> docs/deliverables/review_pairs_<name>.csv, review_eval_<name>.csv
"""
import csv
import json
import random
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "docs/deliverables"
SEED = 42
PAIR_REVIEWERS = ["임병건", "백윤우", "박채영"]
EVAL_REVIEWERS = ["백윤우", "박채영"]
TYPE = {1: "단일 수치", 2: "연도 비교", 3: "기업 비교", 4: "서술 요약", 5: "답 없는 질문"}
SPEC_KO = {"explicit": "연도 명시", "omitted": "연도 생략", "relative": "상대 표현(올해 등)",
           "name": "회사명", "alias": "회사 별칭", }


def krw(v):
    if v is None:
        return ""
    neg, a = v < 0, abs(round(v))
    jo, r = divmod(a, 10**12)
    eok, r = divmod(r, 10**8)
    s = " ".join(p for p in (f"{jo:,}조" if jo else "", f"{eok:,}억" if eok else "") if p) or f"{a:,}"
    return ("-" if neg else "") + s + "원"


def write(path, header, rows):
    with open(path, "w", encoding="utf-8-sig", newline="") as f:
        w = csv.writer(f)
        w.writerow(header)
        w.writerows(rows)
    print("wrote", path.relative_to(ROOT), len(rows), "rows")


def pairs_sheets():
    ids = re.findall(r"^## \d+\. `(m1-\d{5})`", (ROOT / "outputs/midterm/pairs_m1_review_200.md").read_text(), re.M)
    assert len(ids) == 200
    pairs = {p["pair_id"]: p for p in map(json.loads, open(ROOT / "data/train/pairs_m1.jsonl", encoding="utf-8"))}
    rng = random.Random(SEED)
    pick = rng.sample(ids, 99)
    own = {r: pick[i * 33:(i + 1) * 33] for i, r in enumerate(PAIR_REVIEWERS)}
    extra = {r: own[PAIR_REVIEWERS[(i + 1) % 3]][:5] for i, r in enumerate(PAIR_REVIEWERS)}  # A<-B, B<-C, C<-A
    header = ["번호", "pair_id", "회사", "사업연도", "섹션", "질문 형식", "질문", "생성 모델 답", "정답 청크(본문)",
              "판정(답가능/베끼기/모호/오류)", "이유(한 줄)"]
    for r in PAIR_REVIEWERS:
        items = own[r] + extra[r]
        random.Random(f"{SEED}-{r}").shuffle(items)
        rows = []
        for n, pid in enumerate(items, 1):
            p = pairs[pid]
            sp = p.get("spec") or {}
            fmt = " / ".join(x for x in (SPEC_KO.get(sp.get("year"), sp.get("year")), SPEC_KO.get(sp.get("company"), sp.get("company")),
                                          "숫자 질문" if p.get("is_numeric") else "서술 질문") if x)
            rows.append([n, pid, p["corp_name"], p["fiscal_year"], p["section_path"], fmt, p["anchor"], p.get("answer") or "",
                         p["positive"], "", ""])
        write(OUT / f"review_pairs_{r}.csv", header, rows)
    key = {"seed": SEED, "own": own, "overlap": {r: extra[r] for r in PAIR_REVIEWERS},
           "overlap_rule": "each sheet = own 33 + first 5 of the next reviewer's 33 (A<-B, B<-C, C<-A)"}
    (OUT / "review_key_pairs.json").write_text(json.dumps(key, ensure_ascii=False, indent=1))


def eval_sheets():
    import pandas as pd
    corpus = pd.read_json(ROOT / "data/processed/corpus_a.jsonl", lines=True, dtype={"chunk_id": str}).set_index("chunk_id")
    items = [x for x in map(json.loads, open(ROOT / "data/eval/evalset_v1.jsonl", encoding="utf-8"))
             if x["review"]["status"] != "discarded" and x.get("split") in ("dev", "test")]
    rng = random.Random(SEED)
    pick = []
    for t in TYPE:
        pool = sorted((x for x in items if x["type"] == t), key=lambda x: x["qid"])
        pick += rng.sample(pool, 6)

    def answer(x):
        if x.get("answer_value") is not None:
            e = x.get("answer_extra") or {}
            s = krw(x["answer_value"])
            if x["type"] == 2 and "value_from" in e:
                s = f"증감액 {s} ({krw(e['value_from'])} → {krw(e['value_to'])}, {e.get('change_rate_pct')}%)"
            if x["type"] == 3 and "value_a" in e:
                s = f"차이 {s}, 큰 쪽 {e.get('larger')} ({krw(e['value_a'])} vs {krw(e['value_b'])})"
            if x.get("answer_alt"):
                s += " / 재작성 값 허용: " + ", ".join(krw(v) for v in x["answer_alt"])
            return s
        return x.get("answer_text") or ""

    def evidence(x):
        g = x["gold_chunk_ids"]
        if not g:
            return "(근거 없음: 답 없는 질문)"
        parts = [f"[{c}]\n{corpus.loc[c, 'text']}" for c in g[:3]]
        if len(g) > 3:
            parts.append(f"(외 {len(g) - 3}개 청크: {', '.join(g[3:])})")
        return "\n\n".join(parts)

    header = ["번호", "qid", "유형", "질문", "정답", "근거 청크 ID", "근거 청크 텍스트(최대 3개)",
              "질문 명확(Y/N)", "정답 하나(Y/N)", "근거 맞음(Y/N)", "판정(통과/수정/폐기)", "메모"]
    for r in EVAL_REVIEWERS:
        order = list(pick)
        random.Random(f"{SEED}-{r}").shuffle(order)
        rows = [[n, x["qid"], TYPE[x["type"]], x["question"], answer(x), ", ".join(x["gold_chunk_ids"]), evidence(x),
                 "", "", "", "", ""] for n, x in enumerate(order, 1)]
        write(OUT / f"review_eval_{r}.csv", header, rows)
    (OUT / "review_key_eval.json").write_text(json.dumps({"seed": SEED, "qids": [x["qid"] for x in pick]}, indent=1))


if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    pairs_sheets()
    eval_sheets()

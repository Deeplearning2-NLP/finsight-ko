"""G2: evalset v1.1 = v1 + the team's consensus review of 30 items (docs/deliverables/review_eval_*.csv), and an
automatic scan of the other 167 live items for the same kinds of problems (list only; the team decides).

Edits (review.history entries with by = "team-consensus"; the memo that motivated each edit is quoted):
  - 6 unanswerable items: name the target report (2024 사업연도, filed 2025-03) and the reference date / scope;
    a keyword scan of that report is recorded as evidence of absence (to be confirmed by a person).
  - 6 company-compare items: gold list reordered so both companies appear among the first chunks (the sheet showed
    only the first 3, all of one company). No gold id added or removed.
  - 6 unit-conflict items: the corpus fix is G1 (data/processed/corpus_a_unitfix_AB.jsonl, same chunk ids); here the
    representative evidence chunk is moved first and the fix is noted.
  - 2 ambiguous wordings, 2 truncated narrative answers (rewritten from the gold chunk), 1 restated-value evidence
    (FY2023 report chunks carrying the restated FY2022 figure added to the gold list).
Splits are unchanged (data/eval/splits.json). No retrieval or test run happens here.

Usage: python scripts/evalset_v1_1.py -> data/eval/evalset_v1_1.jsonl, docs/evalset_v1_1_changes.md
"""
import csv
import json
import re
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
BY = "team-consensus"
DATE = "2026-11-19"

Q_EDIT = {
    "draft-u001": "HMM의 2024 사업연도 사업보고서(2025년 3월 공시)에 2025년 2분기 확정 배당금(주당)이 나와 있나요?",
    "draft-u009": "기아의 2024 사업연도 사업보고서(2025년 3월 공시)에 회사가 제시한 향후 12개월(2025년 3월 기준) 목표 주가가 있나요?",
    "draft-u010": "두나무의 2024 사업연도 사업보고서(2025년 3월 공시)에 경쟁 거래소(빗썸·코인원)의 비공개 내부 영업 전략이 나와 있나요?",
    "hum-0053": "삼성전자의 2024 사업연도 사업보고서(2025년 3월 공시)에 경영진이 밝힌 2026년 말 목표 주가가 있나요?",
    "hum-0054": "한국전력공사의 2024 사업연도 사업보고서(2025년 3월 공시)에 2027년 1월 주택용 전기요금의 확정 인상액(원/kWh)이 나와 있나요?",
    "hum-0055": "SK하이닉스의 2024 사업연도 사업보고서(2025년 3월 공시)에 경쟁사 마이크론의 HBM3E 생산 수율(%)이 나와 있나요?",
    "hum-0011": "하이닉스의 2024 사업연도 연결 매출은 2023 사업연도의 2배를 넘었는데, 증가액은 얼마인가?",
    "hum-0042": "KB국민은행의 유동성커버리지비율(LCR, 은행업감독업무시행세칙 기준, 외화 LCR 제외)은 2023년 4분기와 2024년 4분기에 각각 몇 %였고, 금융감독원 규제비율은 어떻게 바뀌었나?",
    "draft-s042": "국민은행의 2023 사업연도 사업보고서에서 2023년 4분기 유동성 지표(LCR·외화 LCR)와 유동성 위기관리 체계는 어떻게 설명되나요?",
}
UNANS_SCAN = {  # qid -> (company, fiscal year, regex for the absence check)
    "draft-u001": ("HMM", 2024, r"분기\s*배당|중간\s*배당"),
    "draft-u009": ("기아", 2024, r"목표\s*주가"),
    "draft-u010": ("두나무", 2024, r"빗썸|코인원"),
    "hum-0053": ("삼성전자", 2024, r"목표\s*주가"),
    "hum-0054": ("한국전력공사", 2024, r"2027.{0,40}(요금|인상)|(요금|인상).{0,40}2027"),
    "hum-0055": ("SK하이닉스", 2024, r"마이크론|Micron"),
}
A_EDIT = {
    "draft-s004": ("중동 항로는 공급 감소와 코로나 영향이 해소되어 2023년 수요·공급에 따라 시황이 오르내렸다(운임 하락 시 선사 임시 결항으로 반등, "
                   "인상 시 임시 운항으로 선복이 늘어 다시 하락). 춘절·국경절·라마단 등 특수기 물량 효과는 약해졌다. 2023년 말 홍해 지정학적 이슈로 "
                   "운임이 일시 급반등했으나 2024년 1월 임시 운항 증가로 다시 하락했다. 인도 항로는 초강세가 끝나 팬데믹 이전 수준으로 돌아왔고, "
                   "2022년 4분기부터 하락하다 홍해 이슈로 2024년 초 일시 반등했다. 회사는 얼라이언스 차원의 항로 합리화(공급 조절, 시장 모니터링)로 "
                   "수익성 극대화를 추진하고, 인도 항로는 지중해 시황에 따라 격주 운영·선복 축소 등 가변적으로 운영하며 영업력을 확대한다.",
                   ["중동: 수급에 따른 시황 변동, 임시 결항·임시 운항에 따른 운임 반등·하락", "특수기(춘절·국경절·라마단) 물량 효과 약화",
                    "2023년 말 홍해 이슈로 운임 일시 급반등, 2024년 1월 다시 하락", "인도: 팬데믹 이전 수준으로 회복, 2024년 초 일시 반등",
                    "전략: 얼라이언스 항로 합리화로 공급 조절·모니터링·수익성 극대화", "인도 항로는 지중해 시황에 따라 격주 운영·선복 축소 등 가변 운영, 영업력 확대"]),
    "draft-s042": ("2023년 4분기 원화 LCR은 102.90%로 금융감독원 경영지도비율 95.0%를 웃돈다. 2023년 3분기부터 상향된 규제비율(92.5%→95.0%)에 "
                   "대응해 고유동성자산(HQLA)을 크게 늘려 전년 동기 대비 3.17%p 올랐다. 외화 LCR은 135.26%로 경영지도비율 80%를 웃돌며, "
                   "만기 HQLA 재투자를 일부 줄여 4.33%p 하락했다. 위기관리는 '유동성상황진단'·'유동성 조기경보'로 일별·월별 모니터링을 하고, "
                   "위기 판단 시 재무전략협의회 산하 비상대책위원회가 비상자금조달계획을 실행하며, 이 절차는 연 1회 이상 리스크관리위원회 결의로 갱신한다.",
                   ["LCR 102.90% (경영지도비율 95.0%), 전년 동기 대비 +3.17%p", "규제비율 92.5%→95.0% 상향에 대응해 HQLA 확대",
                    "외화 LCR 135.26% (경영지도비율 80%), -4.33%p", "유동성상황진단·조기경보로 일별/월별 모니터링",
                    "위기 시 비상대책위원회가 비상자금조달계획 실행, 연 1회 이상 리스크관리위원회 결의로 갱신"]),
}
GOLD_ADD = {"draft-n0004": ["20240328001188_0058", "20240328001188_0067", "20240328001188_0500"]}
GOLD_FIRST = {"draft-n0034": "_0102", "draft-n0037": "_0086", "draft-n0206": "_0089"}
UNIT_ITEMS = {"draft-n0034", "draft-n0037", "draft-n0066", "draft-n0203", "draft-n0206", "hum-0014"}
COMPARE_ITEMS = {"draft-n0335", "draft-n0336", "draft-n0339", "draft-n0347", "draft-n0348", "draft-n0365"}


def interleave(ids, corp_of):
    groups = {}
    for c in ids:
        groups.setdefault(corp_of[c], []).append(c)
    out, lists = [], list(groups.values())
    for i in range(max(map(len, lists))):
        out += [l[i] for l in lists if i < len(l)]
    return out


def main():
    corpus = pd.read_json(ROOT / "data/processed/corpus_a.jsonl", lines=True, dtype={"chunk_id": str})
    corp_of = dict(zip(corpus.chunk_id, corpus.corp_name))
    text_of = dict(zip(corpus.chunk_id, corpus.text))
    items = [json.loads(l) for l in open(ROOT / "data/eval/evalset_v1.jsonl", encoding="utf-8")]
    memo = {r["qid"]: r for r in csv.DictReader(open(ROOT / "docs/deliverables/review_eval_백윤우.csv", encoding="utf-8-sig"))}
    reviewed = set(memo)
    log = []
    for x in items:
        q = x["qid"]
        if q not in reviewed:
            continue
        m = memo[q]
        verdict = m["판정(통과/수정/폐기)"].strip()
        hist = []
        if q in Q_EDIT:
            hist.append(f"question: '{x['question']}' -> '{Q_EDIT[q]}'")
            x["question"] = Q_EDIT[q]
        if q in UNANS_SCAN:
            corp, fy, pat = UNANS_SCAN[q]
            g = corpus[(corpus.corp_name == corp) & (corpus.fiscal_year == fy)]
            hits = g[g.text.str.contains(pat, regex=True)].chunk_id.tolist()
            x["fiscal_year"] = [fy]
            x["absence_check"] = {"report": f"{corp} FY{fy}", "pattern": pat, "hits": hits,
                                  "status": "키워드 검색만 함 — 사람이 hits를 읽고 부재를 확정해야 함"}
            hist.append(f"absence scan in {corp} FY{fy} /{pat}/: {len(hits)} chunk(s) to read")
        if q in A_EDIT:
            hist.append("answer_text rewritten from the gold chunk (was truncated)")
            x["answer_text"], x["answer_points"] = A_EDIT[q]
        if q in GOLD_ADD:
            add = [c for c in GOLD_ADD[q] if c not in x["gold_chunk_ids"]]
            assert all(c in text_of and "9,949,357" in text_of[c] for c in add)
            x["gold_chunk_ids"] += add
            hist.append(f"gold + {add} (restated FY2022 value 9,949,357백만원 in the FY2023 report)")
        if q in GOLD_FIRST:
            first = next(c for c in x["gold_chunk_ids"] if c.endswith(GOLD_FIRST[q]))
            x["gold_chunk_ids"] = [first] + [c for c in x["gold_chunk_ids"] if c != first]
            hist.append(f"representative evidence {first} moved first")
        if q in UNIT_ITEMS:
            hist.append("unit header conflict/missing in evidence chunks: fixed in the corpus by G1 (corpus_a_unitfix_AB, same ids)")
        if q in COMPARE_ITEMS:
            before = x["gold_chunk_ids"][:3]
            x["gold_chunk_ids"] = interleave(x["gold_chunk_ids"], corp_of)
            hist.append(f"gold order interleaved by company (first 3 were {sorted({corp_of[c] for c in before})})")
            if q == "draft-n0365":
                vb = f"{round(x['answer_extra']['value_b'] / 1e6):,}"
                kia = [c for c in x["gold_chunk_ids"] if corp_of[c] == "기아" and vb in text_of[c]]
                hist.append(f"기아 value {vb}백만원 found in {kia}")
        hist.append(f"team memo: {m['메모']}")
        old = x.get("review", {})
        x["review"] = {"status": {"통과": "pass", "수정": "revise", "폐기": "discarded"}[verdict], "by": BY,
                       "date": DATE, "history": old.get("history", []) + [f"[{BY} {DATE}] " + h for h in hist],
                       "previous": {k: old.get(k) for k in ("status", "by")}}
        x["version"] = "v1.1"
        log.append((q, verdict, hist[:-1]))
    with open(ROOT / "data/eval/evalset_v1_1.jsonl", "w", encoding="utf-8") as f:
        for x in items:
            f.write(json.dumps(x, ensure_ascii=False) + "\n")

    # automatic scan of the other live items (list only)
    unit_ab = set()
    audit = ROOT / "outputs/midterm/unit_audit.json"
    if audit.exists():
        a = json.loads(audit.read_text())["affected"]
        unit_ab = set(a["A"]) | set(a["B"])
    rest = [x for x in items if x["qid"] not in reviewed and x["review"]["status"] != "discarded"]
    scan = {"truncated answer": [], "answerless question without report/year": [],
            "company compare: first 3 gold chunks from one company": [], "numeric gold chunk with wrong/missing unit (G1 A/B)": []}
    for x in rest:
        a = x.get("answer_text") or ""
        if a and ("생성이 잘려" in a or '"points"' in a or '", "' in a or a.rstrip().endswith(":")):
            scan["truncated answer"].append(x["qid"])
        if x["type"] == 5 and not re.search(r"20\d\d\s*사업\s*연도|20\d\d\s*사업보고서", x["question"]):
            scan["answerless question without report/year"].append(x["qid"])
        if x["type"] == 3 and len({corp_of[c] for c in x["gold_chunk_ids"][:3]}) == 1 and len(x["corp"]) > 1:
            scan["company compare: first 3 gold chunks from one company"].append(x["qid"])
        if x["type"] in (1, 2, 3) and any(c in unit_ab for c in x["gold_chunk_ids"]):
            scan["numeric gold chunk with wrong/missing unit (G1 A/B)"].append(x["qid"])
    L = ["# 평가셋 v1.1 변경 내역 (G2)", "",
         f"`scripts/evalset_v1_1.py` → `data/eval/evalset_v1_1.jsonl` (git-ignored). 분할은 v1 그대로(`data/eval/splits.json`). "
         f"검수한 30문항에 3인 합의 판정과 메모를 반영했다(`review.by = {BY}`). 검색·test 실행 없음.", "",
         "| qid | 판정 | 반영 |", "|---|---|---|"]
    for q, v, h in sorted(log):
        L.append(f"| {q} | {v} | {'; '.join(h) if h else '그대로 (통과)'} |")
    L += ["", "## 나머지 167문항 자동 점검 (목록만, 고칠지는 팀이 정함)", ""]
    for k, v in scan.items():
        L.append(f"- **{k}:** {len(v)}문항" + (f" — {', '.join(v)}" if v else ""))
    L += ["", f"점검 대상 {len(rest)}문항. 단위 항목은 G1 코퍼스 수정본에서 이미 고쳐진다(같은 청크 ID)."]
    (ROOT / "docs/evalset_v1_1_changes.md").write_text("\n".join(L) + "\n", encoding="utf-8")
    print("\n".join(L))


if __name__ == "__main__":
    main()

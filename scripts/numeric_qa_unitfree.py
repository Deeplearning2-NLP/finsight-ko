"""F5: re-grade the saved numeric answers (outputs/midterm/qa_rows_<split>.jsonl from numeric_qa.py) — analysis only.

Two criteria for the same stored key value (no new generation):
  strict    : within +-0.5% of answer_value or answer_alt (finsight.grading, as in numeric_qa.py)
  unit-free : within +-0.5% after multiplying by 10^k for some integer k in [-12, 12] (any power-of-ten scale error)
For company comparison (type 3) the absolute difference is accepted, as in numeric_qa.py.
Purpose: split RAG errors into "right digits, wrong unit/scale" vs "wrong number".

Usage: python scripts/numeric_qa_unitfree.py   -> outputs/midterm/qa_unitfree.json, docs/f5_numeric_unitfree.md
"""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))
from finsight.grading import grade_value  # noqa: E402

M = ROOT / "outputs/midterm"


def ok(v, x, unitfree):
    if v is None:
        return False
    cands = [v] + ([-v] if x["type"] == 3 else [])
    scales = [10.0 ** k for k in range(-12, 13)] if unitfree else [1.0]
    return any(grade_value(c * s, x) for c in cands for s in scales)


def main():
    items = {x["qid"]: x for x in map(json.loads, open(ROOT / "data/eval/evalset_v1.jsonl", encoding="utf-8"))}
    res = {}
    for split in ("dev", "test"):
        rows = [r for r in map(json.loads, open(M / f"qa_rows_{split}.jsonl", encoding="utf-8")) if r["type"] != 5]
        out = {"n": len(rows)}
        for s in ("s0", "s1", "s2"):
            st = [ok(r[f"{s}_value"], items[r["qid"]], False) for r in rows]
            uf = [ok(r[f"{s}_value"], items[r["qid"]], True) for r in rows]
            answered = sum(r[f"{s}_value"] is not None for r in rows)
            by_type = {}
            for t in (1, 2, 3):
                ii = [i for i, r in enumerate(rows) if r["type"] == t]
                by_type[t] = {"n": len(ii), "strict": sum(st[i] for i in ii), "unitfree": sum(uf[i] for i in ii)}
            out[s] = {"answered": answered, "strict": sum(st), "unitfree": sum(uf),
                      "scale_only_errors": sum(u and not a for a, u in zip(st, uf)),
                      "wrong_number": answered - sum(uf), "by_type": by_type}
        res[split] = out
    (M / "qa_unitfree.json").write_text(json.dumps(res, ensure_ascii=False, indent=1))

    name = {"s0": "S0 LLM 단독", "s1": "S1 RAG", "s2": "S2 RAG + OpenDART"}
    L = ["# F5. 수치 검증 보조 채점: 단위 무시 vs ±0.5% (분석만)", "",
         "`scripts/numeric_qa_unitfree.py`. `numeric_qa.py`가 저장한 핵심 수치를 그대로 다시 채점했다(새 생성 없음). "
         "**단위 무시** = 10^k배(k = −12…12) 차이를 허용하고 ±0.5% 안이면 정답. "
         "RAG 오답이 단위·자릿수 문제인지 숫자 자체 문제인지 나누려는 것이다. 숫자는 저장된 출력에서만 가져왔다.", ""]
    for split, title in (("test", "자체 평가셋 v1 test"), ("dev", "자체 평가셋 v1 dev")):
        o = res[split]
        n = o["n"]
        L += [f"**{title}, 수치 {n}문항 (EXAONE-3.5-7.8B, greedy)**", "",
              "| 시스템 | 답함 | 정답 ±0.5% | 정답 단위 무시 | 자릿수만 틀림 | 숫자 자체가 틀림 |", "|---|---|---|---|---|---|"]
        for s in ("s0", "s1", "s2"):
            r = o[s]
            L.append(f"| {name[s]} | {r['answered']} | {r['strict']}/{n} = {r['strict'] / n:.1%} | "
                     f"{r['unitfree']}/{n} = {r['unitfree'] / n:.1%} | {r['scale_only_errors']} | {r['wrong_number']} |")
        L += ["", "| 시스템 | 단일 수치 (±0.5% → 단위 무시) | 연도 비교 | 기업 비교 |", "|---|---|---|---|"]
        for s in ("s0", "s1", "s2"):
            bt = o[s]["by_type"]
            L.append(f"| {name[s]} | " + " | ".join(f"{bt[t]['strict']}/{bt[t]['n']} → {bt[t]['unitfree']}/{bt[t]['n']}" for t in (1, 2, 3)) + " |")
        L.append("")
    t1 = res["test"]["s1"]
    L += ["**해석 (test)**", "",
          f"- RAG가 답한 {t1['answered']}개 중 {t1['strict']}개만 맞았다. {t1['scale_only_errors']}개는 자릿수만 틀렸고(단위 무시로 맞음), "
          f"{t1['wrong_number']}개는 숫자 자체가 틀렸다. 즉 RAG 오답은 표 단위를 반영하지 않은 것과, 증감·차이 계산 또는 다른 셀을 읽은 것이 섞여 있다.",
          "- 보고서 15절의 \"단위 오류 29건\"(정답 대비 비율을 10의 거듭제곱으로 어림한 분류)과 숫자가 다르다. 이 표는 answer_alt와 부호까지 맞춰 정확히 다시 판정한 값이라, 이쪽을 쓴다.",
          "- 단위 문제를 빼고 봐도 RAG(S1)는 OpenDART 조회(S2)에 크게 못 미친다. 특히 계산이 필요한 연도 비교·기업 비교에서 그렇다(유형별 표).",
          "- **주의:** S2 정답률 95.8%는 문항이 모두 OpenDART 6개 계정(매출·영업이익·순이익·자산·부채·자본)의 템플릿형 수치라서, "
          "조회기가 다루는 범위와 정확히 겹치는 쉬운 조건이다. 6개 계정 밖의 수치나 주석·본문 숫자는 S2가 조회하지 못한다(\"조회 불가\" → RAG 답 사용).", ""]
    (ROOT / "docs/f5_numeric_unitfree.md").write_text("\n".join(L), encoding="utf-8")
    print("\n".join(L))


if __name__ == "__main__":
    main()

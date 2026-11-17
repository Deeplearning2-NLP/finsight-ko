"""M1 step 1: source chunks + a question-format spec per chunk, fixed before any evaluation set is seen.

Proposal feedback (10/26): diversify training questions by criteria independent of the evaluation set.
The format mix below was written into docs/midterm-plan.md before generation and is not tuned:
  year     : explicit "20XX 사업연도" 40% / omitted 45% / relative ("당기", "전기" …) 15%
  company  : official name 55% / alias 15% (name if the company has no alias) / omitted 30%
  content  : numeric (only on value-table chunks) with prob. 0.5, otherwise descriptive
Same chunk pool as S1 (20 non-evaluation companies, sections II·IV, >= 150 tokens, up to 65 per company-year).

Usage: python scripts/m1_chunks.py
Output: data/train/m1_source_chunks.jsonl
"""
from __future__ import annotations

import json
import random
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from build_corpus import COMPANIES as EVAL_COMPANIES  # noqa: E402
from pairs_pilot_chunks import PILOT  # noqa: E402
from pairs_r2_chunks import value_table  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
TRAIN = ROOT / "data" / "train"
SEED = 42
PER = 65
YEAR_MIX = [("explicit", 0.40), ("omitted", 0.45), ("relative", 0.15)]
COMPANY_MIX = [("name", 0.55), ("alias", 0.15), ("omitted", 0.30)]
P_NUMERIC = 0.5
ALIASES = {"에이치디현대오일뱅크": "HD현대오일뱅크", "삼성E&A": "삼성이앤에이", "비바리퍼블리카": "토스",
           "DB손해보험": "DB손보", "한국수력원자력": "한수원", "교보생명보험": "교보생명",
           "현대캐피탈": "현대캐피탈㈜", "여천NCC": "여천엔씨씨", "아시아나항공": "아시아나",
           "JB금융지주": "JB금융", "롯데카드": "롯데카드㈜", "경남은행": "BNK경남은행"}


def draw(rng: random.Random, mix):
    x, acc = rng.random(), 0.0
    for name, p in mix:
        acc += p
        if x < acc:
            return name
    return mix[-1][0]


def rule1(corp: str, year: int, ym: str, cm: str) -> tuple[str, str | None]:
    alias = ALIASES.get(corp)
    if cm == "alias" and not alias:
        cm = "name"
    comp = {"name": f'회사명 "{corp}"을 질문에 넣습니다.',
            "alias": f'회사는 "{alias}"라는 짧은 이름으로 부릅니다 ("{corp}"이라고 쓰지 않습니다).',
            "omitted": "회사 이름을 질문에 쓰지 않습니다(\"당사\", \"이 회사\"도 쓰지 않습니다)."}[cm]
    yr = {"explicit": f'"{year} 사업연도"를 질문에 넣습니다.',
          "omitted": "연도나 사업연도를 질문에 쓰지 않습니다.",
          "relative": '연도 숫자 대신 "당기", "전기", "올해 보고서 기준" 같은 상대 표현을 하나 씁니다.'}[ym]
    return f"1. {comp} {yr}", cm


def main() -> None:
    industry = {c: ind for ind, cs in PILOT.items() for c in cs}
    assert len(industry) == 20 and not set(industry) & set(EVAL_COMPANIES)
    c = pd.read_json(ROOT / "data/processed/corpus_train.jsonl", lines=True, dtype={"rcept_no": str, "chunk_id": str})
    assert not set(c.corp_name) & set(EVAL_COMPANIES)
    c = c[c.is_core_section & (c.n_tokens >= 150)].copy()
    c["value_table"] = [value_table(t, it) for t, it in zip(c.text, c.is_table)]
    rng = random.Random(SEED)
    out = []
    for (corp, fy), sub in c.groupby(["corp_name", "fiscal_year"]):
        idx = sorted(sub.index)
        rng.shuffle(idx)
        for i in idx[:PER]:
            r = c.loc[i]
            ym, cm = draw(rng, YEAR_MIX), draw(rng, COMPANY_MIX)
            numeric = bool(r.value_table) and rng.random() < P_NUMERIC
            r1, cm = rule1(corp, int(fy), ym, cm)
            out.append({"chunk_id": r.chunk_id, "corp_name": corp, "industry": industry[corp], "fiscal_year": int(fy),
                        "rcept_no": r.rcept_no, "section_path": r.section_path, "is_table": bool(r.is_table),
                        "value_table": bool(r.value_table), "n_tokens": int(r.n_tokens), "text": r.text,
                        "spec": {"year": ym, "company": cm, "numeric": numeric, "alias": ALIASES.get(corp)},
                        "rule1": r1, "rule4_key": numeric})
    TRAIN.mkdir(parents=True, exist_ok=True)
    with open(TRAIN / "m1_source_chunks.jsonl", "w", encoding="utf-8") as f:
        for o in out:
            f.write(json.dumps(o, ensure_ascii=False) + "\n")
    df = pd.DataFrame([o["spec"] for o in out])
    print(len(out), df.year.value_counts(normalize=True).round(3).to_dict(),
          df.company.value_counts(normalize=True).round(3).to_dict(), round(df.numeric.mean(), 3))


if __name__ == "__main__":
    main()

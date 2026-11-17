"""M1 step 3: filter format-diversified generations into training pairs with three negative types.

Format filters (spec fixed in m1_chunks.py):
  company name    -> the official name appears        alias   -> the alias appears, the official name does not
  company omitted -> no name/alias, no "당사"/"이 회사"
  year explicit   -> query_meta finds the chunk's fiscal year
  year omitted    -> no year and no relative expression    relative -> a relative expression, no explicit year
Other filters as in pairs_r2_build.py: answer says "not in text", deictic words, numeric answers whose numbers are
not in the chunk (finsight.numbers incl. units).
Negatives stored per pair (training picks via --neg-mode):
  neg_year    : same company & top-level section, other fiscal year (only for year-explicit questions; for year-less
                questions such a passage is a correct answer too) — skip candidates that contain the answer / its numbers
                or have 5-gram Jaccard >= 0.8 with the positive
  neg_section : same company, a different section path (any year), best BM25, same exclusions
  neg_easy    : a random chunk of another training company
Then 5,000 pairs (seed 42).

Usage: python scripts/m1_build.py
Outputs: data/train/pairs_m1.jsonl, outputs/midterm/pairs_m1_stats.json, pairs_m1_review_200.md (human review sheet)
"""
from __future__ import annotations

import json
import random
import re
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
from rank_bm25 import BM25Okapi

sys.path.insert(0, str(Path(__file__).resolve().parent))
from eval_corpus_a_retrieval import grams  # noqa: E402
from pairs_pilot_build import DEICTIC, jaccard, parse  # noqa: E402
from pairs_r2_build import NOANS, NUM, norm, numbers_ok  # noqa: E402

from finsight.query_meta import _RELATIVE, extract
from finsight.tokenize import content_morphs

ROOT = Path(__file__).resolve().parents[1]
TRAIN = ROOT / "data" / "train"
OUT = ROOT / "outputs" / "midterm"
N_PAIRS = 5000
SEED = 42
SELF_REF = re.compile(r"당사|이 회사|해당 회사|동사")


def ans_numbers(ans: str) -> set[str]:
    return {t.replace(",", "") for t in NUM.findall(ans) if len(t.replace(",", "").replace(".", "")) >= 3
            and not (len(t) == 4 and t.isdigit() and 2000 <= int(t) <= 2035)}


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    src = {c["chunk_id"]: c for c in map(json.loads, open(TRAIN / "m1_source_chunks.jsonl", encoding="utf-8"))}
    gens = [json.loads(l) for l in open(TRAIN / "m1_generations.jsonl", encoding="utf-8")]
    cfg = json.loads((TRAIN / "m1_generation_config.json").read_text())
    pool = pd.read_json(ROOT / "data/processed/corpus_train.jsonl", lines=True, dtype={"rcept_no": str, "chunk_id": str})
    pool = pool[pool.is_core_section].reset_index(drop=True)
    pool["sec1"] = pool.section_path.str.split(" > ").str[0]
    companies = sorted(pool.corp_name.unique())
    all_names = {c for c in companies} | {s["spec"]["alias"] for s in src.values() if s["spec"]["alias"]}

    stage, kept = Counter(), []
    for g in gens:
        c = src[g["chunk_id"]]
        sp = c["spec"]
        qs = parse(g["raw"])
        if not qs:
            stage["parse_fail_chunks"] += 1
        for q in qs:
            stage["generated"] += 1
            text, ans = q["question"].strip(), str(q.get("answer", ""))
            meta = extract(text, aliases={c["corp_name"]: [c["corp_name"]]})
            years = set(meta["fiscal_years"])
            has_rel = bool(_RELATIVE.search(text))
            reason = None
            if sp["company"] == "name" and c["corp_name"] not in text:
                reason = "company_name_missing"
            elif sp["company"] == "alias" and (sp["alias"] not in text or c["corp_name"] in text):
                reason = "alias_rule_broken"
            elif sp["company"] == "omitted" and (any(n in text for n in all_names) or SELF_REF.search(text)):
                reason = "company_not_omitted"
            elif sp["year"] == "explicit" and c["fiscal_year"] not in years:
                reason = "year_missing"
            elif sp["year"] == "omitted" and (years or has_rel):
                reason = "year_not_omitted"
            elif sp["year"] == "relative" and (years or not has_rel):
                reason = "relative_rule_broken"
            elif NOANS.search(ans):
                reason = "answer_says_not_in_text"
            elif DEICTIC.search(text):
                reason = "deictic"
            elif q.get("is_numeric"):
                ok, how = numbers_ok(ans, c["text"])
                if not ok:
                    reason = f"numeric_{how}"
            if reason:
                stage[f"drop_{reason}"] += 1
            else:
                kept.append({"chunk_id": c["chunk_id"], "question": text, "answer": ans,
                             "is_numeric": bool(q.get("is_numeric")), "spec": sp})
    stage["kept_after_filters"] = len(kept)
    idx = list(range(len(kept)))
    random.Random(SEED).shuffle(idx)
    sample = [kept[i] for i in sorted(idx[:N_PAIRS])]

    toks = {cid: content_morphs(t) for cid, t in zip(pool.chunk_id, pool.text)}
    by_corp = {k: g for k, g in pool.groupby("corp_name")}
    bm_cache: dict = {}
    rng = random.Random(SEED)

    def best_neg(key, cand, q, pos_text, ans, skips):
        if not len(cand):
            return None
        if key not in bm_cache:
            bm_cache[key] = BM25Okapi([toks[x] for x in cand.chunk_id])
        order = np.argsort(-bm_cache[key].get_scores(content_morphs(q)))
        pg, an, nums = grams(pos_text), norm(ans), ans_numbers(ans)
        for j in order[:50]:
            r = cand.iloc[int(j)]
            cg = grams(r.text)
            if pg and cg and len(pg & cg) / len(pg | cg) >= 0.8:
                skips["near_duplicate"] += 1
                continue
            if len(an) >= 6 and an in norm(r.text):
                skips["contains_answer"] += 1
                continue
            if nums and nums & {t.replace(",", "") for t in NUM.findall(r.text)}:
                skips["contains_answer_number"] += 1
                continue
            return r
        return None

    pairs, skips = [], Counter()
    for i, r in enumerate(sample):
        c = src[r["chunk_id"]]
        g = by_corp[c["corp_name"]]
        sec1 = c["section_path"].split(" > ")[0]
        ny = None
        if r["spec"]["year"] == "explicit":
            cand = g[(g.sec1 == sec1) & (g.fiscal_year != c["fiscal_year"])]
            ny = best_neg(("y", c["corp_name"], sec1, c["fiscal_year"]), cand, r["question"], c["text"], r["answer"], skips)
        cand = g[g.section_path != c["section_path"]]
        if r["spec"]["year"] != "explicit":  # other-year copies of the same section would be correct answers
            cand = cand[cand.sec1 != sec1]
        ns = best_neg(("s", c["corp_name"], c["section_path"], r["spec"]["year"] != "explicit"), cand,
                      r["question"], c["text"], r["answer"], skips)
        others = pool[pool.corp_name != c["corp_name"]]
        ne = others.iloc[rng.randrange(len(others))]
        qt, pt = set(content_morphs(r["question"])), set(toks[c["chunk_id"]])
        pairs.append({"pair_id": f"m1-{i + 1:05d}", "anchor": r["question"], "positive": c["text"],
                      "positive_id": c["chunk_id"], "corp_name": c["corp_name"], "fiscal_year": c["fiscal_year"],
                      "section_path": c["section_path"], "spec": r["spec"], "answer": r["answer"],
                      "is_numeric": r["is_numeric"],
                      "neg_year": None if ny is None else ny.text, "neg_year_id": None if ny is None else ny.chunk_id,
                      "neg_section": None if ns is None else ns.text, "neg_section_id": None if ns is None else ns.chunk_id,
                      "neg_easy": ne.text, "neg_easy_id": ne.chunk_id,
                      "lexical_jaccard": round(jaccard(qt, pt), 4),
                      "generated_by": "llm:" + cfg["model"].split("/")[-1], "review": "none (0% human review)"})
    with open(TRAIN / "pairs_m1.jsonl", "w", encoding="utf-8") as f:
        for p in pairs:
            f.write(json.dumps(p, ensure_ascii=False) + "\n")
    pdf = pd.DataFrame(pairs)
    spec = pd.DataFrame(list(pdf.spec))
    stats = {"generation": {k: v for k, v in cfg.items() if "prompt" not in k and k != "rule4_variants"},
             "chunks": len(gens), "stages": dict(stage), "keep_rate": round(len(kept) / stage["generated"], 3),
             "pairs": len(pairs),
             "format_mix_pairs": {"year": spec.year.value_counts(normalize=True).round(3).to_dict(),
                                  "company": spec.company.value_counts(normalize=True).round(3).to_dict(),
                                  "numeric": round(float(pdf.is_numeric.mean()), 3)},
             "with_neg_year": int(pdf.neg_year_id.notna().sum()), "with_neg_section": int(pdf.neg_section_id.notna().sum()),
             "neg_skips": dict(skips),
             "lexical_jaccard_p50": round(float(pdf.lexical_jaccard.median()), 4)}
    (OUT / "pairs_m1_stats.json").write_text(json.dumps(stats, ensure_ascii=False, indent=1, default=str))

    review = random.Random(SEED).sample(pairs, 200)  # proposal feedback: ~200 pairs reviewed by people
    lines = ["# 학습 쌍 M1 사람 검수표 (200쌍)", "",
             "판정: 답가능 / 베끼기 / 모호 / 오류. 기준 = 질문이 그 청크만으로 분명하게 답할 수 있는가(회사·연도를 생략한 형식이면 "
             "생략은 오류가 아님). 판정 칸을 채운 뒤 `scripts/m1_review_tally.py`로 집계.", ""]
    for n, p in enumerate(review, 1):
        lines += [f"## {n}. `{p['pair_id']}` {p['corp_name']} FY{p['fiscal_year']} · {p['section_path']} · "
                  f"형식 {p['spec']['year']}/{p['spec']['company']}/{'숫자' if p['is_numeric'] else '서술'}", "",
                  f"- 질문: {p['anchor']}", f"- 모델 답: {p['answer']}",
                  f"- 본문: {p['positive'][:700].replace(chr(10), ' ')}", "- 판정: ", "- 메모: ", ""]
    (OUT / "pairs_m1_review_200.md").write_text("\n".join(lines))
    print(json.dumps(stats, ensure_ascii=False, indent=1, default=str))


if __name__ == "__main__":
    main()

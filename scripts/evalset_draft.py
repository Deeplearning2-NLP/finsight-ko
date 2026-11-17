"""Draft of the in-house evaluation set (LLM/template part, ~70%), following docs/eval_guideline.md.

Generator LLM = EEVE-Korean-Instruct-10.8B (local, vLLM, fp8). It differs from the training-pair generator (Qwen2.5-7B)
and from the planned answer LLM (EXAONE), so neither the fine-tuned retriever nor the system LLM gets questions in
its own style. Human-written items (~30%, mostly comparisons) and the review are left to the team; every item here
is marked review = "draft (not reviewed)".

Types (guideline 2): 1 single value / 2 year change / 3 company compare  -> from data/eval/numeric_auto.jsonl
(answers = OpenDART values, restatement alternatives kept); the question is paraphrased by the LLM and the
paraphrase must keep the company, fiscal year(s) and account word. Gold chunk ids: chunks of that company-year
in corpus_a whose text contains the reported figure (in the table's unit) — found automatically.
4 narrative summary -> LLM question + 2-4 sentence answer from a body-text chunk of an evaluation company;
the source chunk is the gold (plus same-report chunks containing the answer's key sentence).
5 unanswerable -> categories a 사업보고서 never contains x company, phrased by the LLM; gold = [].

Usage: CUDA_VISIBLE_DEVICES=3 /mnt/urp_jwl_shared/bk/vllm-venv/bin/python scripts/evalset_draft.py --stage generate
       python scripts/evalset_draft.py --stage build       (project venv: query_meta checks, gold search)
Outputs: data/eval/evalset_draft_candidates.jsonl, data/eval/evalset_draft.jsonl (140 items), outputs/midterm/evalset_draft_stats.json
"""
from __future__ import annotations

import argparse
import json
import random
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
EVAL = ROOT / "data" / "eval"
OUT = ROOT / "outputs" / "midterm"
MODEL = "yanolja/EEVE-Korean-Instruct-10.8B-v1.0"
SEED = 42
TARGET = {1: 32, 2: 32, 3: 32, 4: 32, 5: 12}  # 140 = 70% of 200; the team adds ~60 human-written items
UNANSWERABLE = ["향후 주가 전망이나 목표 주가", "대표이사 개인의 자산이나 가족 관계", "아직 끝나지 않은 2026 사업연도의 연간 실적",
                "경쟁사 내부의 비공개 전략", "특정 직원의 연봉", "다음 분기 배당 확정 금액"]
# dropped at build time: average employee pay IS in 사업보고서 (VIII), so these questions can be answerable
UNANSWERABLE_DROP = {"특정 직원의 연봉"}
SYS = "A chat between a curious user and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the user's questions."


def prompt(user: str) -> str:
    return f"{SYS}\nHuman: {user}\nAssistant: "


def paraphrase_prompt(q: str) -> str:
    return prompt(f"""다음 질문을 투자자가 실제로 물을 법한 자연스러운 한국어 질문 한 문장으로 바꿔 쓰세요.
회사명, 사업연도(예: "2024 사업연도"), 연결/별도, 재무 계정 이름은 그대로 남기고, 숫자 답은 쓰지 마세요.
질문: {q}
바꾼 질문만 출력하세요.""")


def narrative_prompt(corp: str, year: int, sec: str, text: str) -> str:
    return prompt(f"""아래는 {corp}의 {year} 사업연도 사업보고서({sec}) 일부입니다.

{text}

이 본문으로 답할 수 있는 서술형 질문(사업 내용, 위험 요인, 전략, 시장 상황 요약) 1개와 답을 만드세요.
규칙: 질문에 "{corp}"와 "{year} 사업연도"를 넣는다. 본문 문장을 그대로 베끼지 않는다. 숫자 하나만 묻는 질문은 만들지 않는다.
답은 본문 근거로 2~4문장, 핵심 사실 2~3개를 목록으로.
JSON만 출력: {{"question": "...", "answer": "...", "points": ["...", "..."]}}""")


def unanswerable_prompt(corp: str, cat: str) -> str:
    return prompt(f"""{corp}의 사업보고서에는 원래 나오지 않는 정보인 "{cat}"를 묻는 자연스러운 질문 한 문장을 만드세요.
회사명 "{corp}"를 넣으세요. 질문만 출력하세요.""")


def generate() -> None:
    import pandas as pd
    from vllm import LLM, SamplingParams

    rng = random.Random(SEED)
    num = [json.loads(l) for l in open(EVAL / "numeric_auto.jsonl", encoding="utf-8")]
    corpus = pd.read_json(ROOT / "data/processed/corpus_a.jsonl", lines=True, dtype={"rcept_no": str, "chunk_id": str})
    body = corpus[corpus.is_core_section & ~corpus.is_table & (corpus.n_tokens >= 250)]
    narr = []
    for (corp, fy), sub in body.groupby(["corp_name", "fiscal_year"]):
        narr += sub.sample(min(3, len(sub)), random_state=SEED).to_dict("records")
    comps = sorted(corpus.corp_name.unique())
    unans = [(c, cat) for c in comps for cat in rng.sample(UNANSWERABLE, 2)]

    jobs = ([("num", q["qid"], paraphrase_prompt(q["question"])) for q in num]
            + [("narr", r["chunk_id"], narrative_prompt(r["corp_name"], r["fiscal_year"], r["section_path"], r["text"]))
               for r in narr]
            + [("unans", f"{c}|{cat}", unanswerable_prompt(c, cat)) for c, cat in unans])
    llm = LLM(model=MODEL, dtype="bfloat16", quantization="fp8", max_model_len=4096, gpu_memory_utilization=0.9,
              seed=SEED)
    sp = SamplingParams(temperature=0.6, top_p=0.9, max_tokens=400, seed=SEED)
    outs = llm.generate([p for _, _, p in jobs], sp)
    EVAL.mkdir(parents=True, exist_ok=True)
    with open(EVAL / "evalset_draft_raw.jsonl", "w", encoding="utf-8") as f:
        for (kind, key, _), o in zip(jobs, outs):
            f.write(json.dumps({"kind": kind, "key": key, "raw": o.outputs[0].text}, ensure_ascii=False) + "\n")
    (OUT / "evalset_draft_gen_config.json").write_text(json.dumps(
        {"model": MODEL, "quantization": "fp8", "temperature": 0.6, "top_p": 0.9, "seed": SEED, "jobs": len(jobs),
         "narrative_chunks": len(narr), "unanswerable": len(unans)}, ensure_ascii=False, indent=1))
    print("generated", len(jobs))


def build() -> None:
    import sys
    import pandas as pd
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from pairs_pilot_build import DEICTIC
    from finsight.query_meta import extract

    rng = random.Random(SEED)
    raw = [json.loads(l) for l in open(EVAL / "evalset_draft_raw.jsonl", encoding="utf-8")]
    num = {q["qid"]: q for q in map(json.loads, open(EVAL / "numeric_auto.jsonl", encoding="utf-8"))}
    corpus = pd.read_json(ROOT / "data/processed/corpus_a.jsonl", lines=True, dtype={"rcept_no": str, "chunk_id": str})
    by_cy = {k: g for k, g in corpus.groupby(["corp_name", "fiscal_year"])}
    cands, drops = [], {}

    def drop(r):
        drops[r] = drops.get(r, 0) + 1

    def figure_strings(v: float) -> list[str]:
        out = []
        for div in (1e6, 1e3, 1, 1e8):  # 백만원, 천원, 원, 억원
            x = abs(v) / div
            if x >= 1 and abs(x - round(x)) < 1e-6:
                out.append(f"{int(round(x)):,}")
        return out

    def find_gold(corp: str, fy: int, value: float) -> list[str]:
        g = by_cy.get((corp, fy))
        if g is None:
            return []
        figs = figure_strings(value)
        hit = g[g.text.apply(lambda t: any(f in t for f in figs))]
        core = hit[hit.section_path.str.contains("요약재무정보|재무제표|경영진단|손익|재무상태")]
        return (core if len(core) else hit).chunk_id.tolist()[:8]

    for r in raw:
        text = r["raw"].strip().split("\n")[0].strip().strip('"')
        if r["kind"] == "num":
            q = num[r["key"]]
            src = q["source"]
            corps = sorted({s["corp"] for s in src})
            years = sorted({s["fiscal_year"] for s in src})
            meta = extract(text)
            acc = re.sub(r"[은는이가]$", "", q["question"].split(" 기준 ")[1].split(" ")[0]) if " 기준 " in q["question"] else ""
            if not set(corps) <= set(meta["companies"]) or not set(years) <= set(meta["fiscal_years"]):
                drop("num_lost_company_or_year")
                text = q["question"]  # keep the template wording rather than a broken paraphrase
                paraphrased = False
            elif acc and acc not in text:
                drop("num_lost_account")
                text, paraphrased = q["question"], False
            else:
                paraphrased = True
            gold = []
            if q["type"] == 1:
                gold = find_gold(src[0]["corp"], src[0]["fiscal_year"], q["answer_value"])
            elif q["type"] == 2:  # later report shows both years
                later = max(src, key=lambda s: s["fiscal_year"])
                gold = find_gold(later["corp"], later["fiscal_year"], q["answer_extra"]["value_to"])
            else:
                gold = find_gold(src[0]["corp"], src[0]["fiscal_year"], q["answer_extra"]["value_a"]) + \
                       find_gold(src[1]["corp"], src[1]["fiscal_year"], q["answer_extra"]["value_b"])
            if not gold:
                drop("num_no_gold_chunk")
                continue
            cands.append({"qid": q["qid"].replace("num-", "draft-n"), "type": q["type"], "question": text,
                          "question_template": q["question"], "paraphrased": paraphrased, "corp": corps,
                          "fiscal_year": years, "answer_value": q["answer_value"], "answer_alt": q.get("answer_alt"),
                          "answer_unit": "원", "answer_extra": q["answer_extra"], "gold_chunk_ids": gold,
                          "source": src, "generated_by": f"template+llm:{MODEL.split('/')[1]}" if paraphrased else "template"})
        elif r["kind"] == "narr":
            # EEVE often ran past max_tokens inside the JSON or answered as "질문: … 답변: …";
            # only the question (+ gold chunk) is needed for retrieval, so recover it tolerantly.
            js = None
            m = re.search(r"\{.*\}", r["raw"], re.S)
            if m:
                try:
                    js = json.loads(m.group(0))
                except json.JSONDecodeError:
                    js = None
            if not js:
                mq = re.search(r'"question"\s*:\s*"(.+?)"', r["raw"]) or re.search(r"질문\s*[:：]\s*(.+)", r["raw"])
                ma = re.search(r'"answer"\s*:\s*"(.+)', r["raw"], re.S) or re.search(r"답변?\s*[:：]\s*(.+)", r["raw"], re.S)
                js = {"question": mq.group(1).strip()} if mq else None
                if js and ma:
                    js["answer"] = ma.group(1).strip()[:600] + " (생성이 잘려 부분만 저장)"
                if js:
                    drop("narr_recovered_without_json")
            if not js or not js.get("question"):
                drop("narr_parse")
                continue
            c = corpus.set_index("chunk_id").loc[r["key"]]
            meta = extract(js["question"])
            if c.corp_name not in meta["companies"] or c.fiscal_year not in meta["fiscal_years"]:
                drop("narr_missing_company_or_year")
                continue
            if DEICTIC.search(js["question"]):
                drop("narr_deictic")
                continue
            cands.append({"qid": f"draft-s{len([x for x in cands if x['type'] == 4]) + 1:03d}", "type": 4,
                          "question": js["question"], "corp": [c.corp_name], "fiscal_year": [int(c.fiscal_year)],
                          "answer_text": js.get("answer", ""), "answer_points": js.get("points", []),
                          "gold_chunk_ids": [r["key"]], "source_section": c.section_path,
                          "generated_by": f"llm:{MODEL.split('/')[1]}"})
        else:
            corp, cat = r["key"].split("|")
            if cat in UNANSWERABLE_DROP:
                drop("unans_category_dropped")
                continue
            if corp not in extract(text)["companies"]:
                drop("unans_missing_company")
                continue
            cands.append({"qid": f"draft-u{len([x for x in cands if x['type'] == 5]) + 1:03d}", "type": 5,
                          "question": text, "corp": [corp], "fiscal_year": [], "category": cat,
                          "answer_text": "보고서에서 확인할 수 없음", "gold_chunk_ids": [],
                          "generated_by": f"llm:{MODEL.split('/')[1]}"})
    for c in cands:
        c["review"] = "draft (not reviewed)"
        c["split"] = None
    with open(EVAL / "evalset_draft_candidates.jsonl", "w", encoding="utf-8") as f:
        for c in cands:
            f.write(json.dumps(c, ensure_ascii=False) + "\n")
    chosen = []
    for t, n in TARGET.items():
        pool = [c for c in cands if c["type"] == t]
        rng.shuffle(pool)
        pool.sort(key=lambda c: (not c.get("paraphrased", True)))  # prefer natural paraphrases, stable otherwise
        # spread over companies (company pairs for type 3)
        keys = {"|".join(c["corp"]) for c in pool}
        cap = max(1, -(-n // max(1, len(keys))))
        seen, picked = {}, []
        for c in pool:
            key = "|".join(c["corp"])
            if seen.get(key, 0) < cap:
                picked.append(c)
                seen[key] = seen.get(key, 0) + 1
            if len(picked) == n:
                break
        chosen += picked
    with open(EVAL / "evalset_draft.jsonl", "w", encoding="utf-8") as f:
        for c in chosen:
            f.write(json.dumps(c, ensure_ascii=False) + "\n")
    stats = {"candidates_by_type": {t: sum(c["type"] == t for c in cands) for t in TARGET},
             "draft_by_type": {t: sum(c["type"] == t for c in chosen) for t in TARGET}, "drops": drops,
             "draft_paraphrased_numeric": sum(c.get("paraphrased", False) for c in chosen if c["type"] <= 3),
             "draft_companies": sorted({x for c in chosen for x in c["corp"]})}
    (OUT / "evalset_draft_stats.json").write_text(json.dumps(stats, ensure_ascii=False, indent=1))
    print(json.dumps(stats, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", choices=["generate", "build"], required=True)
    a = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    generate() if a.stage == "generate" else build()

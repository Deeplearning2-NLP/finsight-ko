"""Representative experiment 2: numeric verification — LLM alone (S0) / RAG (S1) / RAG + OpenDART lookup (S2).
Design fixed in docs/decisions.md 2026-11-19 before running.

Questions: evalset v1 numeric items (types 1-3) + unanswerable items (type 5), split dev / test.
Answer LLM: EXAONE-3.5-7.8B-Instruct (vLLM, greedy). Retrieval: v2 final system top-5 (scripts/eval_v2.py).
S2: the LLM turns the question into a lookup plan (companies, fiscal years, one of 6 accounts, value/change/compare);
Python reads OpenDART values from data/processed/financials_a.csv (fnlttSinglAcntAll, CFS; cached, no API call),
computes the key number, and the LLM only writes the explanation. If the plan says "not answerable" -> S1's answer.

Stages:
  contexts (project venv, GPU): top-5 chunks per question      -> data/eval/qa_contexts_v1.jsonl
  generate (vLLM venv, GPU):    S0, S1, S2 plan + S2 explain    -> outputs/midterm/qa_raw_<split>.jsonl
  grade    (project venv):      metrics                         -> outputs/midterm/qa_results_<split>.json
Usage:
  CUDA_VISIBLE_DEVICES=0 .venv/bin/python scripts/numeric_qa.py contexts
  CUDA_VISIBLE_DEVICES=0 /mnt/urp_jwl_shared/bk/vllm-venv/bin/python scripts/numeric_qa.py generate --split dev
  .venv/bin/python scripts/numeric_qa.py grade --split dev
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
M = ROOT / "outputs" / "midterm"
CTX = ROOT / "data/eval/qa_contexts_v1.jsonl"
MODEL = "LGAI-EXAONE/EXAONE-3.5-7.8B-Instruct"
ACCOUNTS = ["매출액", "영업이익", "당기순이익", "자산총계", "부채총계", "자본총계"]
COMPANIES = ["삼성전자", "SK하이닉스", "LG전자", "현대자동차", "기아", "SK텔레콤", "한국전력공사", "국민은행", "HMM", "두나무"]
NOT_FOUND = "보고서에서 확인할 수 없음"


def items(split: str) -> list[dict]:
    out = []
    for x in map(json.loads, open(ROOT / "data/eval/evalset_v1.jsonl", encoding="utf-8")):
        if x.get("split") == split and (x["type"] == 5 or (x["type"] in (1, 2, 3) and x.get("answer_value") is not None)):
            out.append(x)
    return out


# ---------------------------------------------------------------- stage 1: contexts
def contexts() -> None:
    sys.path.insert(0, str(ROOT / "scripts"))
    from eval_v2 import Env
    sel = json.loads((M / "selection_v2.json").read_text())["chosen"]
    with open(CTX, "w", encoding="utf-8") as f:
        for split in ("dev", "test"):
            env = Env("v1", split)
            its = items(split)
            env.q = [x["question"] for x in its]
            env.gold = [set(x["gold_chunk_ids"]) for x in its]
            ranked = env.run(sel)
            for x, r in zip(its, ranked):
                top = r[:5]
                f.write(json.dumps({"qid": x["qid"], "split": split, "chunk_ids": top,
                                    "chunks": [env.texts[env.idx_of[c]] for c in top]}, ensure_ascii=False) + "\n")
    print("wrote", CTX)


# ---------------------------------------------------------------- formatting helpers
def krw_words(v: float) -> str:
    """302231360000000 -> '302조 2,313억 6,000만원' (rounded to 만원)."""
    neg, v = v < 0, abs(round(v))
    jo, rest = divmod(v, 10**12)
    eok, rest = divmod(rest, 10**8)
    man = rest // 10**4
    parts = []
    if jo:
        parts.append(f"{jo:,}조")
    if eok:
        parts.append(f"{eok:,}억")
    if man:
        parts.append(f"{man:,}만")
    s = " ".join(parts) + "원" if parts else f"{v:,}원"
    return ("-" if neg else "") + s


JSON_RE = re.compile(r"\{.*\}", re.S)


def parse_json(text: str) -> dict | None:
    m = JSON_RE.search(text or "")
    if not m:
        return None
    try:
        return json.loads(m.group(0))
    except json.JSONDecodeError:
        try:
            return json.loads(m.group(0).replace("'", '"'))
        except json.JSONDecodeError:
            return None


KEY_RULE = ("key_value에는 질문이 묻는 핵심 금액 하나만 적습니다: 단일 수치 질문은 그 값, 두 사업연도 비교는 "
            "증감액(나중 연도 − 앞 연도, 감소면 음수), 두 회사 비교는 두 값의 차이(절댓값). "
            "'1조 2,345억원'처럼 원 단위가 드러나게 적고, 모르면 null.")


def prompt_s0(q: str) -> str:
    return (f"다음은 한국 기업 사업보고서에 관한 질문입니다. 알고 있는 지식으로 답하세요. 확실하지 않으면 key_value를 null로 두세요.\n"
            f"질문: {q}\n\n{KEY_RULE}\n"
            '출력은 JSON 한 줄만: {"answer": "2~3문장 답변", "key_value": "금액 또는 null"}')


def prompt_s1(q: str, chunks: list[str]) -> str:
    ctx = "\n\n".join(f"[문서 {i + 1}]\n{c}" for i, c in enumerate(chunks))
    return (f"아래 사업보고서 발췌만 근거로 질문에 답하세요. 표의 '(단위: 백만원)' 같은 단위를 반영해 원 단위로 환산하세요. "
            f"발췌에 답이 없으면 answer는 \"{NOT_FOUND}\", key_value는 null로 두세요.\n\n{ctx}\n\n질문: {q}\n\n{KEY_RULE}\n"
            '출력은 JSON 한 줄만: {"answer": "2~3문장 답변", "key_value": "금액 또는 null"}')


def prompt_plan(q: str) -> str:
    return ("질문을 OpenDART 재무제표(연결 기준) 조회 계획으로 바꾸세요.\n"
            f"회사는 다음 중에서만: {', '.join(COMPANIES)} (별칭은 정식 이름으로: 하이닉스→SK하이닉스, 현대차→현대자동차, "
            "한전→한국전력공사, KB국민은행→국민은행, SKT→SK텔레콤, 업비트 운영사→두나무, 기아차→기아).\n"
            f"계정은 다음 중 하나만: {', '.join(ACCOUNTS)} (매출·영업수익→매출액, 순이익→당기순이익, 영업적자·영업손실→영업이익).\n"
            "사업연도는 2022, 2023, 2024 중에서. '제N기' 같은 기수가 나오면 질문에 적힌 대로 두지 말고 사업연도로 바꾸지 못하면 answerable을 false로.\n"
            "operation: value(한 회사·한 연도 값), change(같은 회사 두 연도 증감), compare(두 회사 같은 연도 차이).\n"
            "이 6개 계정의 수치를 묻는 질문이 아니거나 사업보고서에 없는 정보(주가 전망, 개인 정보, 미래 실적 등)면 answerable은 false.\n"
            f"질문: {q}\n"
            '출력은 JSON 한 줄만: {"answerable": true, "companies": ["..."], "fiscal_years": [2023, 2024], '
            '"account": "...", "operation": "value|change|compare"}')


def prompt_explain(q: str, facts: str, chunks: list[str]) -> str:
    ctx = "\n\n".join(f"[문서 {i + 1}]\n{c}" for i, c in enumerate(chunks))
    return (f"질문에 2~3문장으로 답하세요. 금액은 아래 '확인된 수치'에 있는 것만 그대로 쓰고, 새 숫자를 만들지 마세요. "
            f"사업보고서 발췌는 설명(원인·배경)에만 참고하세요.\n\n확인된 수치 (OpenDART 연결재무제표, Python 계산):\n{facts}\n\n"
            f"사업보고서 발췌:\n{ctx}\n\n질문: {q}\n"
            '출력은 JSON 한 줄만: {"answer": "답변"}')


# ---------------------------------------------------------------- tool
def load_fin() -> dict:
    import csv
    fin = {}
    with open(ROOT / "data/processed/financials_a.csv", encoding="utf-8-sig") as f:
        for r in csv.DictReader(f):
            if r["fs_div"] == "CFS" and r["value_krw"]:  # empty = OpenDART 013 (국민은행 FY2022)
                fin[(r["corp"], int(r["fiscal_year"]), r["account_std"])] = float(r["value_krw"])
    return fin


def execute(plan: dict | None, fin: dict) -> tuple[float | None, str, dict]:
    """Returns (key value, facts text for the LLM, trace)."""
    if not plan or not plan.get("answerable"):
        return None, "", {"status": "not_answerable"}
    comps = [c for c in plan.get("companies", []) if c in COMPANIES]
    years = sorted({int(y) for y in plan.get("fiscal_years", []) if str(y).isdigit() and int(y) in (2022, 2023, 2024)})
    acc, op = plan.get("account"), plan.get("operation")
    trace = {"companies": comps, "years": years, "account": acc, "operation": op}
    if acc not in ACCOUNTS or not comps or not years:
        return None, "", trace | {"status": "bad_plan"}
    # dev fix (2026-11-19): the operation follows from the counts; the LLM's label is only a hint
    op = {(1, 1): "value", (1, 2): "change", (2, 1): "compare"}.get((len(comps), len(years)), op)
    trace["operation_used"] = op
    get = lambda c, y: fin.get((c, y, acc))  # noqa: E731
    try:
        if op == "value" and len(comps) == 1 and len(years) == 1:
            v = get(comps[0], years[0])
            facts = f"- {comps[0]} {years[0]} 사업연도 {acc}: {krw_words(v)}"
            key = v
        elif op == "change" and len(comps) == 1 and len(years) == 2:
            a, b = get(comps[0], years[0]), get(comps[0], years[1])
            key = b - a
            rate = f"{key / abs(a) * 100:+.2f}%" if a else "계산 불가(기준 연도 값 0)"
            if a is not None and a < 0:
                rate = "계산하지 않음(기준 연도가 손실)"
            facts = (f"- {comps[0]} {years[0]} 사업연도 {acc}: {krw_words(a)}\n- {comps[0]} {years[1]} 사업연도 {acc}: {krw_words(b)}\n"
                     f"- 증감액: {krw_words(key)}, 증감률: {rate}")
        elif op == "compare" and len(comps) == 2 and len(years) == 1:
            a, b = get(comps[0], years[0]), get(comps[1], years[0])
            key = abs(a - b)
            big = comps[0] if a >= b else comps[1]
            facts = (f"- {comps[0]} {years[0]} 사업연도 {acc}: {krw_words(a)}\n- {comps[1]} {years[0]} 사업연도 {acc}: {krw_words(b)}\n"
                     f"- 더 큰 쪽: {big}, 차이: {krw_words(key)}")
        else:
            return None, "", trace | {"status": "bad_plan"}
    except TypeError:  # value missing in the cache
        return None, "", trace | {"status": "no_data"}
    return key, facts, trace | {"status": "ok", "key": key}


# ---------------------------------------------------------------- stage 2: generate
def generate(split: str) -> None:
    from vllm import LLM, SamplingParams
    its = items(split)
    ctx = {c["qid"]: c for c in map(json.loads, open(CTX, encoding="utf-8"))}
    llm = LLM(MODEL, dtype="bfloat16", max_model_len=8192, gpu_memory_utilization=0.9, seed=42, trust_remote_code=True)
    tok = llm.get_tokenizer()
    sp = SamplingParams(temperature=0.0, max_tokens=512)

    def trim(chunks, budget=5600):
        out, used = [], 0
        for c in chunks:
            n = len(tok.encode(c))
            if used + n > budget:
                c = tok.decode(tok.encode(c)[: max(0, budget - used)])
            out.append(c)
            used += min(n, budget - used)
            if used >= budget:
                break
        return out

    def run(prompts):
        msgs = [[{"role": "system", "content": "You are EXAONE model from LG AI Research, a helpful assistant."},
                 {"role": "user", "content": p}] for p in prompts]
        outs = llm.chat(msgs, sp, use_tqdm=False)
        return [o.outputs[0].text for o in outs]

    chunks = {x["qid"]: trim(ctx[x["qid"]]["chunks"]) for x in its}
    s0 = run([prompt_s0(x["question"]) for x in its])
    s1 = run([prompt_s1(x["question"], chunks[x["qid"]]) for x in its])
    plans = run([prompt_plan(x["question"]) for x in its])
    fin = load_fin()
    exe = [execute(parse_json(p), fin) for p in plans]
    need = [i for i, (k, _, _) in enumerate(exe) if k is not None]
    expl = run([prompt_explain(its[i]["question"], exe[i][1], chunks[its[i]["qid"]]) for i in need])
    expl_of = dict(zip(need, expl))
    with open(M / f"qa_raw_{split}.jsonl", "w", encoding="utf-8") as f:
        for i, x in enumerate(its):
            f.write(json.dumps({"qid": x["qid"], "s0": s0[i], "s1": s1[i], "plan": plans[i],
                                "tool": {"key": exe[i][0], "facts": exe[i][1], "trace": exe[i][2]},
                                "s2_explain": expl_of.get(i)}, ensure_ascii=False) + "\n")
    print("wrote", M / f"qa_raw_{split}.jsonl")


# ---------------------------------------------------------------- stage 3: grade
def evidence_numbers(chunks: list[str]) -> list[float]:
    from finsight.numbers import find_amounts
    nums = []
    for c in chunks:
        nums += find_amounts(c)
        for m in re.finditer(r"(?<![\d.])\d{1,3}(?:,\d{3})+(?![\d,])|(?<![\d.,])\d{4,}(?![\d,])", c):
            v = float(m.group(0).replace(",", ""))
            nums += [v * u for u in (1, 1e3, 1e6, 1e8)]
    return nums


def grade(split: str) -> None:
    import numpy as np
    sys.path.insert(0, str(ROOT / "scripts"))
    from eval_corpus_a_diagnostics import cluster_bootstrap, mcnemar_exact
    from finsight.grading import grade_value
    from finsight.numbers import find_amounts, parse_krw, rel_close
    its = {x["qid"]: x for x in items(split)}
    raw = [json.loads(l) for l in open(M / f"qa_raw_{split}.jsonl", encoding="utf-8")]
    ctx = {c["qid"]: c for c in map(json.loads, open(CTX, encoding="utf-8"))}

    def key_of(out: str):
        j = parse_json(out)
        if j is None:
            return None, out, False
        kv = j.get("key_value")
        v = None
        if isinstance(kv, (int, float)):
            v = float(kv)
        elif isinstance(kv, str) and kv.strip().lower() not in ("null", "none", ""):
            m = re.search(r"[△▲\-−]?\s*\d[\d,\.]*\s*(?:조|억|만|원)[\d,\.\s조억만원]*", kv)
            if m:
                v = parse_krw(m.group(0).replace(" ", "").replace("−", "-"))
            else:  # plain number = 원
                m = re.search(r"[△▲\-−]?\s*\d[\d,]*(?:\.\d+)?", kv)
                v = float(re.sub(r"[^\d.]", "", m.group(0))) if m else None
                if v is not None and re.match(r"\s*[△▲\-−]", m.group(0)):
                    v = -v
        return v, str(j.get("answer", "")), True

    rows = []
    for r in raw:
        if r["qid"] not in its:  # narrative comparison items (no numeric answer)
            continue
        x = its[r["qid"]]
        ev_chunks = evidence_numbers(ctx[r["qid"]]["chunks"])
        row = {"qid": r["qid"], "type": x["type"], "origin": x["origin"],
               "restated": bool(x.get("answer_alt")), "tool_status": r["tool"]["trace"].get("status")}
        v0, a0, ok0 = key_of(r["s0"])
        v1, a1, ok1 = key_of(r["s1"])
        if r["tool"]["key"] is not None:
            v2 = r["tool"]["key"]
            j = parse_json(r["s2_explain"] or "")
            a2 = str(j.get("answer", "")) if j else (r["s2_explain"] or "")
            ev2 = ev_chunks + [abs(v) for v in find_amounts(r["tool"]["facts"])] + [abs(v2)]
            used_tool = True
        else:
            v2, a2, ev2, used_tool = v1, a1, ev_chunks, False
        row["s2_used_tool"] = used_tool
        for s, v, a, ev in (("s0", v0, a0, None), ("s1", v1, a1, ev_chunks), ("s2", v2, a2, ev2)):
            claims = [abs(c) for c in find_amounts(a) if abs(c) >= 1e8]  # money claims (>= 1억원)
            row[f"{s}_value"] = v
            row[f"{s}_refused"] = v is None and (NOT_FOUND in a or "확인할 수 없" in a or not a.strip())
            row[f"{s}_claims"] = len(claims)
            row[f"{s}_unsupported"] = (sum(not any(rel_close(c, e, 0.005) for e in ev) for c in claims)
                                       if ev is not None else None)
            if x["type"] != 5:
                row[f"{s}_correct"] = grade_value(v, x) or (v is not None and grade_value(-v, x) and x["type"] == 3)
            row[f"{s}_answer"] = a
        rows.append(row)

    num = [r for r in rows if r["type"] != 5]
    una = [r for r in rows if r["type"] == 5]
    res = {}
    for s in ("s0", "s1", "s2"):
        c = np.array([r[f"{s}_correct"] for r in num], bool)
        cov = np.array([r[f"{s}_value"] is not None for r in num], bool)
        claims = sum(r[f"{s}_claims"] for r in rows)
        uns = None if s == "s0" else sum(r[f"{s}_unsupported"] for r in rows)
        res[s] = {"numeric_accuracy": float(c.mean()), "coverage": float(cov.mean()),
                  "accuracy_when_answered": float(c[cov].mean()) if cov.any() else None,
                  "wrong_when_answered": int((cov & ~c).sum()),
                  "money_claims": claims, "unsupported_claim_rate": (uns / claims if claims and uns is not None else None),
                  "unanswerable_refusal": float(np.mean([r[f"{s}_value"] is None for r in una])) if una else None,
                  "by_type": {t: float(np.mean([r[f"{s}_correct"] for r in num if r["type"] == t])) for t in (1, 2, 3)},
                  "by_origin": {o: float(np.mean([r[f"{s}_correct"] for r in num if r["origin"] == o]))
                                for o in sorted({r["origin"] for r in num})},
                  "restated_only": float(np.mean([r[f"{s}_correct"] for r in num if r["restated"]] or [np.nan])),
                  "n_restated": sum(r["restated"] for r in num)}
    paired = {}
    for a, b in (("s0", "s1"), ("s1", "s2"), ("s0", "s2")):
        ha = np.array([r[f"{a}_correct"] for r in num], bool)
        hb = np.array([r[f"{b}_correct"] for r in num], bool)
        paired[f"{b} - {a}"] = {"diff": float(hb.mean() - ha.mean()), "mcnemar_p": mcnemar_exact(ha, hb)["p"],
                                "ci95": cluster_bootstrap(hb.astype(float) - ha.astype(float), np.arange(len(num)))["ci95"]}
    out = {"split": split, "n_numeric": len(num), "n_unanswerable": len(una), "model": MODEL,
           "s2_tool_used": sum(r["s2_used_tool"] for r in num), "s2_tool_status": {
               k: sum(r["tool_status"] == k for r in rows) for k in {r["tool_status"] for r in rows}},
           "results": res, "paired": paired}
    (M / f"qa_results_{split}.json").write_text(json.dumps(out, ensure_ascii=False, indent=1))
    with open(M / f"qa_rows_{split}.jsonl", "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(json.dumps({k: v for k, v in out.items() if k != "results"}, ensure_ascii=False, indent=1))
    for s, v in res.items():
        print(s, {k: (round(x, 3) if isinstance(x, float) else x) for k, x in v.items() if not isinstance(x, dict)})


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=["contexts", "generate", "grade"])
    ap.add_argument("--split", default="dev", choices=["dev", "test"])
    a = ap.parse_args()
    {"contexts": lambda: contexts(), "generate": lambda: generate(a.split), "grade": lambda: grade(a.split)}[a.stage]()


if __name__ == "__main__":
    main()

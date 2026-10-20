"""Q6 step 2: generate questions for the pilot chunks with an open model on the server (vLLM).

Runs in the separate vLLM venv (/mnt/urp_jwl_shared/bk/vllm-venv); standard library + vllm only.
Generation LLM (Qwen) != future system/answer LLM (EXAONE family), per decisions.md 2026-10-20.

Usage: CUDA_VISIBLE_DEVICES=1 /mnt/urp_jwl_shared/bk/vllm-venv/bin/python scripts/pairs_pilot_generate.py
Outputs: data/train/pilot_generations.jsonl, data/train/pilot_generation_config.json
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TRAIN = ROOT / "data" / "train"

SYSTEM = "당신은 한국 기업 공시(사업보고서)로 검색 평가용 질문을 만드는 보조자입니다. 반드시 JSON만 출력합니다."
USER_TMPL = """아래는 {corp}의 {year} 사업연도 사업보고서({section_name}) 중 한 부분입니다.

[본문]
{text}

이 본문만 읽고 답할 수 있는 질문을 2개 만드세요. 규칙:
1. 질문에 회사명 "{corp}"과 "{year} 사업연도"를 반드시 넣습니다.
2. 본문 문장을 그대로 베끼지 말고, 투자자나 학생이 실제로 물을 법한 말로 바꿔 씁니다.
3. 본문에 답이 분명히 있는 질문만 만듭니다. 본문에 없는 내용(미래 주가, 다른 회사 등)은 묻지 않습니다.
4. 2개 중 하나는 가능하면 숫자(금액, 비율, 개수 등)를 묻는 질문으로 합니다. 본문에 숫자가 없으면 둘 다 서술형으로 합니다.
5. "해당", "위", "이 표"처럼 본문을 봐야만 알 수 있는 지시어를 쓰지 않습니다.

출력 형식(JSON만):
{{"questions": [{{"question": "...", "answer": "본문 근거로 한 짧은 답", "is_numeric": true}}, {{"question": "...", "answer": "...", "is_numeric": false}}]}}"""
SECTION_NAME = {"II": "II. 사업의 내용", "IV": "IV. 이사의 경영진단 및 분석의견"}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen2.5-7B-Instruct")
    ap.add_argument("--temperature", type=float, default=0.7)
    ap.add_argument("--top-p", type=float, default=0.9)
    ap.add_argument("--max-tokens", type=int, default=512)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    import vllm
    from vllm import LLM, SamplingParams

    chunks = [json.loads(l) for l in open(TRAIN / "pilot_source_chunks.jsonl", encoding="utf-8")]
    llm = LLM(model=args.model, dtype="bfloat16", max_model_len=4096, gpu_memory_utilization=0.85,
              seed=args.seed)
    sp = SamplingParams(temperature=args.temperature, top_p=args.top_p, max_tokens=args.max_tokens,
                        seed=args.seed)
    msgs = [[{"role": "system", "content": SYSTEM},
             {"role": "user", "content": USER_TMPL.format(corp=c["corp_name"], year=c["fiscal_year"],
                                                          section_name=SECTION_NAME[c["section"]],
                                                          text=c["text"])}] for c in chunks]
    t0 = time.time()
    outs = llm.chat(msgs, sp)
    secs = time.time() - t0
    with open(TRAIN / "pilot_generations.jsonl", "w", encoding="utf-8") as f:
        for c, o in zip(chunks, outs):
            f.write(json.dumps({"chunk_id": c["chunk_id"], "raw": o.outputs[0].text}, ensure_ascii=False) + "\n")
    from huggingface_hub.constants import HF_HUB_CACHE
    ref = Path(HF_HUB_CACHE) / f"models--{args.model.replace('/', '--')}" / "refs" / "main"
    snapshot = ref.read_text().strip() if ref.exists() else None  # commit of the weights used
    cfg = {"model": args.model, "model_snapshot": snapshot,
           "vllm_version": vllm.__version__, "dtype": "bfloat16", "temperature": args.temperature,
           "top_p": args.top_p, "max_tokens": args.max_tokens, "seed": args.seed,
           "system_prompt": SYSTEM, "user_prompt_template": USER_TMPL, "n_chunks": len(chunks),
           "generation_seconds": round(secs, 1)}
    (TRAIN / "pilot_generation_config.json").write_text(json.dumps(cfg, ensure_ascii=False, indent=1))
    print(json.dumps({k: v for k, v in cfg.items() if "prompt" not in k}, ensure_ascii=False))


if __name__ == "__main__":
    main()

"""KLUE-BERT (klue/bert-base) comparison — analysis, nothing is selected here.

Bi-encoders: public KURE-v1 / KURE-v1 fine-tuned (M3 hard) / KLUE-BERT zero-shot (mean pooling, no training) /
KLUE-BERT fine-tuned on the same 4,646 M1 pairs with the same recipe (scripts/train_embedding_m.py --base klue/bert-base).
Rerankers inside the v2 pipeline (M3 hard retrieval): none / bge-ft / KLUE-RoBERTa-ft / KLUE-BERT-ft.
Data: evalset v1 dev+test, FinDartBench dev+test (test here = analysis use, see report).
Usage: CUDA_VISIBLE_DEVICES=0 python scripts/eval_kluebert.py -> outputs/midterm/eval_kluebert.json
"""
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import eval_v2  # noqa: E402
from eval_corpus_a_diagnostics import metrics  # noqa: E402

M = eval_v2.M
eval_v2.EMB["kluebert_zero"] = ("klue/bert-base", M / "emb_kluebert_zero.npy")
eval_v2.EMB["kluebert_ft"] = (str(M / "kure-m-kluebert_hard"), M / "emb_m_kluebert_hard.npy")
eval_v2.RERANKERS["kluebert-ft"] = M / "reranker-kluebert"
final = json.loads((M / "selection_v2.json").read_text())["chosen"]
out = {}
for data in ("v1", "fdb"):
    for split in ("dev", "test"):
        env = eval_v2.Env(data, split)
        res = {}
        for emb in ("public", "m3", "kluebert_zero", "kluebert_ft"):
            env.dense_prepare(emb, env.q)
            dense = [env.topk(env._dense[emb][q].astype(np.float32), None, 10) for q in env.q]
            res[f"{emb} | dense"] = metrics(dense, env.gold)
            res[f"{emb} | v2 no rerank"] = env.score(env.run(dict(final, emb=emb, rr="none")))
            res[f"{emb} | v2 + bge-ft"] = env.score(env.run(dict(final, emb=emb)))
        for rr in ("klue-ft", "kluebert-ft"):
            res[f"m3 | v2 + {rr}"] = env.score(env.run(dict(final, rr=rr)))
        out[f"{data}_{split}"] = {"n": len(env.q), "results": res}
        print(data, split, {k: round(v["MRR@10"], 3) for k, v in res.items()}, flush=True)
(M / "eval_kluebert.json").write_text(json.dumps(out, ensure_ascii=False, indent=1))

"""G1: dev-only retrieval comparison of corpus versions (set FINSIGHT_CORPUS=unitfix_AB|unitfix_ABC, or unset = original).
Configs: v2 final (selection_v2.json) and the same without reranker. Data: evalset v1 dev (43) and FinDartBench dev (672).
Also: share of top-5 chunks that are table chunks with a "(단위" line. No test split is touched.
Usage: CUDA_VISIBLE_DEVICES=0 [FINSIGHT_CORPUS=unitfix_AB] python scripts/g1_dev_eval.py -> outputs/midterm/g1_dev_<tag|orig>.json"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import eval_v2  # noqa: E402

M = eval_v2.M
final = json.loads((M / "selection_v2.json").read_text())["chosen"]
out = {"corpus": eval_v2.CORPUS_TAG or "orig", "config": final}
for data in ("v1", "fdb"):
    env = eval_v2.Env(data, "dev")
    for name, cfg in (("v2 final", final), ("v2 no rerank", dict(final, rr="none"))):
        r = env.run(cfg)
        sc = env.score(r)
        top = [c for rr in r for c in rr[:5]]
        tab = [c for c in top if env.corpus.is_table.iat[env.idx_of[c]]]
        sc["top5_table_with_unit_line"] = sum("(단위" in env.texts[env.idx_of[c]] for c in tab) / max(1, len(tab))
        out[f"{data}_dev | {name}"] = sc
        print(data, name, {k: round(v, 3) for k, v in sc.items() if isinstance(v, float)}, flush=True)
(M / f"g1_dev_{eval_v2.CORPUS_TAG or 'orig'}.json").write_text(json.dumps(out, ensure_ascii=False, indent=1))

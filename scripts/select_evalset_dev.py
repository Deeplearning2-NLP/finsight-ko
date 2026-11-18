"""Pick the reranker on the in-house evalset v1 dev split (rule fixed in docs/decisions.md 2026-11-18 before the dev run).

Rule: embedding fixed to the FinDartBench-dev choice (M3 hard) with RRF + metadata filter; candidates = no reranker,
bge-zero, klue-ft, bge-ft. Highest dev MRR@10 wins; if within 0.01 of the best, prefer the simpler one in the order
none < bge-zero < klue-ft < bge-ft (fewer trained parts).
Usage: python scripts/select_evalset_dev.py   -> outputs/midterm/selection_evalset_v1.json
"""
import json
from pathlib import Path

M = Path(__file__).resolve().parents[1] / "outputs" / "midterm"
r = json.load(open(M / "eval_evalset_v1_dev.json"))["results"]
base = next(k for k in r if "(dev 선택)" in k and k.endswith("| RRF+filter"))
order = [("none", base)] + [(n, f"{base}+{n}") for n in ("bge-zero", "klue-ft", "bge-ft") if f"{base}+{n}" in r]
scores = {n: r[k]["MRR@10"] for n, k in order}
best = max(scores.values())
pick = next(n for n, _ in order if scores[n] >= best - 0.01)
out = {"rule": __doc__.split("\n")[2:5], "dev_MRR@10": scores, "best_reranker": pick, "system": dict(order)[pick]}
(M / "selection_evalset_v1.json").write_text(json.dumps(out, ensure_ascii=False, indent=1))
print(json.dumps(out, ensure_ascii=False, indent=1))

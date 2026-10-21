"""R1 evaluation 1: extraction accuracy of finsight.query_meta on 100 hand-labelled dev questions.

Micro precision/recall over (question, company) and (question, year) pairs, exact-set accuracy,
the share of questions that state a year, and the same for years inside the corpus range (2022-2024,
the ones a filter can use). Lists every disagreement for reading.

Usage: python scripts/eval_query_meta.py [--tag v0]
Inputs: data/eval/r1_meta_labels.json (manual labels, git-ignored)
Outputs: outputs/proposal/r1_extract_<tag>_<set>.json
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from eval_findartbench_retrieval import load  # noqa: E402

from finsight.query_meta import extract

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs" / "proposal"
RANGE = {2022, 2023, 2024}


def pr(pred: list[set], gold: list[set]) -> dict:
    tp = sum(len(p & g) for p, g in zip(pred, gold))
    npred, ngold = sum(map(len, pred)), sum(map(len, gold))
    return {"precision": round(tp / npred, 3) if npred else None, "recall": round(tp / ngold, 3) if ngold else None,
            "tp": tp, "pred": npred, "gold": ngold,
            "exact_set_acc": round(sum(p == g for p, g in zip(pred, gold)) / len(gold), 3)}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="v0")
    ap.add_argument("--set", default="labels", choices=["labels", "heldout_labels"])
    args = ap.parse_args()
    labels = json.loads((ROOT / "data/eval/r1_meta_labels.json").read_text())[args.set]
    fdb = load(ROOT / "data/raw/findartbench/data.jsonl").set_index("id")
    qids = list(labels)
    preds = {q: extract(fdb.loc[q, "question"]) for q in qids}
    pc = [set(preds[q]["companies"]) for q in qids]
    gc = [set(labels[q][0]) for q in qids]
    py = [set(preds[q]["fiscal_years"]) for q in qids]
    gy = [set(labels[q][1]) for q in qids]
    res = {"n": len(qids),
           "year_stated_share": round(sum(bool(g) for g in gy) / len(gy), 3),
           "in_range_year_stated_share": round(sum(bool(g & RANGE) for g in gy) / len(gy), 3),
           "company_named_share": round(sum(bool(g) for g in gc) / len(gc), 3),
           "company": pr(pc, gc), "year": pr(py, gy),
           "year_in_range": pr([p & RANGE for p in py], [g & RANGE for g in gy]),
           "ambiguous_labels": sum(1 for q in qids if labels[q][2].startswith("ambiguous")),
           "errors": []}
    for q, p1, g1, p2, g2 in zip(qids, pc, gc, py, gy):
        if p1 != g1 or p2 != g2:
            res["errors"].append({"qid": q, "question": fdb.loc[q, "question"][:120],
                                  "pred_companies": sorted(p1), "gold_companies": sorted(g1),
                                  "pred_years": sorted(p2), "gold_years": sorted(g2), "note": labels[q][2]})
    (OUT / f"r1_extract_{args.tag}_{args.set}.json").write_text(json.dumps(res, ensure_ascii=False, indent=1))
    print(json.dumps({k: v for k, v in res.items() if k != "errors"}, ensure_ascii=False, indent=1))
    for e in res["errors"]:
        print(e["qid"], "| C", e["pred_companies"], "vs", e["gold_companies"], "| Y", e["pred_years"], "vs",
              e["gold_years"], "|", e["note"], "|", e["question"][:70])


if __name__ == "__main__":
    main()

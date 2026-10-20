"""Retrieval metrics. `ranked` is a list of doc ids (best first), `gold` a set of relevant ids."""
from __future__ import annotations

import math
from typing import Iterable, Sequence


def recall_at_k(ranked: Sequence[str], gold: set[str], k: int) -> float:
    if not gold:
        return 0.0
    return len(set(ranked[:k]) & gold) / len(gold)


def mrr_at_k(ranked: Sequence[str], gold: set[str], k: int = 10) -> float:
    for i, d in enumerate(ranked[:k]):
        if d in gold:
            return 1.0 / (i + 1)
    return 0.0


def ndcg_at_k(ranked: Sequence[str], gold: set[str], k: int = 10) -> float:
    dcg = sum(1 / math.log2(i + 2) for i, d in enumerate(ranked[:k]) if d in gold)
    idcg = sum(1 / math.log2(i + 2) for i in range(min(len(gold), k)))
    return dcg / idcg if idcg else 0.0


def evaluate(runs: Iterable[tuple[Sequence[str], set[str]]], ks=(1, 3, 5, 10)) -> dict[str, float]:
    runs = list(runs)
    n = len(runs)
    out: dict[str, float] = {}
    for k in ks:
        out[f"R@{k}"] = sum(recall_at_k(r, g, k) for r, g in runs) / n
    out["MRR@10"] = sum(mrr_at_k(r, g, 10) for r, g in runs) / n
    out["nDCG@10"] = sum(ndcg_at_k(r, g, 10) for r, g in runs) / n
    return out

"""Numeric answer grading with restatement-aware alternatives (decisions.md 2026-10-20).

A question carries ``answer_value`` (the default answer) and optionally ``answer_alt``
(a list of other accepted values, e.g. a restated prior-year figure, or the change computed
from originally reported values). A prediction is correct if it matches any of them within
the relative tolerance used for Numeric Claim Accuracy (0.5%).
"""
from __future__ import annotations

from typing import Iterable

from .numbers import rel_close

TOL = 0.005


def accepted_values(q: dict) -> list[float]:
    vals = [q["answer_value"]]
    alt = q.get("answer_alt")
    if alt is None:
        return vals
    return vals + [v for v in (alt if isinstance(alt, (list, tuple)) else [alt]) if v is not None]


def grade_value(pred: float | None, q: dict, tol: float = TOL) -> bool:
    """True if ``pred`` matches answer_value or any answer_alt within ``tol``."""
    if pred is None:
        return False
    return any(rel_close(pred, v, tol) for v in accepted_values(q))


def accuracy(preds: Iterable[float | None], qs: Iterable[dict], tol: float = TOL) -> float:
    pairs = list(zip(preds, qs))
    return sum(grade_value(p, q, tol) for p, q in pairs) / len(pairs) if pairs else 0.0

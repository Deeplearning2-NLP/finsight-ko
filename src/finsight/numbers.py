"""Korean financial number normalization.

Converts expressions found in DART filings and LLM answers into a single
float in KRW (원), so that answers can be compared with OpenDART values.

Examples
--------
>>> parse_krw("1조 2,345억 원")
1234500000000.0
>>> parse_krw("12,345", unit="백만원")
12345000000.0
>>> parse_krw("△1,234", unit="천원")
-1234000.0
"""
from __future__ import annotations

import re

# multiplier for a table/column unit label
UNIT_MULTIPLIER = {
    "원": 1,
    "천원": 1_000,
    "만원": 10_000,
    "백만원": 1_000_000,
    "천만원": 10_000_000,
    "억원": 100_000_000,
    "십억원": 1_000_000_000,
    "조원": 1_000_000_000_000,
}

# multipliers for Korean large-number words inside a value ("1조 2,345억")
_WORD = {"조": 10**12, "억": 10**8, "만": 10**4, "천": 10**3}

_NEG_MARKS = ("△", "▲", "-", "−", "–")
_NUM = r"\d[\d,]*(?:\.\d+)?"


def _to_float(s: str) -> float:
    return float(s.replace(",", ""))


def normalize_unit(unit: str | None) -> int:
    """Return the KRW multiplier for a unit label like '(단위: 백만원)'."""
    if not unit:
        return 1
    u = re.sub(r"[\s()\[\]:단위]", "", unit)
    if u not in UNIT_MULTIPLIER:
        raise ValueError(f"unknown unit: {unit!r}")
    return UNIT_MULTIPLIER[u]


def parse_krw(text: str, unit: str | None = None) -> float | None:
    """Parse one monetary expression into KRW.

    - Handles comma thousands separators, decimals.
    - Negative forms: leading △/▲/-, or wrapped in parentheses "(1,234)".
    - Korean number words: "1조 2,345억", "3.5조", "120억 원".
    - ``unit`` applies to plain numbers taken from a table column
      (e.g. '백만원'); it is ignored when the text carries its own 조/억/만.
    Returns None for empty / dash-only cells.
    """
    if text is None:
        return None
    s = text.strip()
    if s in {"", "-", "−", "–", "N/A"}:
        return None

    neg = False
    if s.startswith("(") and s.endswith(")"):
        neg, s = True, s[1:-1].strip()
    if s.startswith(_NEG_MARKS):
        neg, s = True, s[1:].strip()

    s = s.replace("원", "").replace(" ", "")

    if any(w in s for w in _WORD):
        total, matched = 0.0, False
        for num, word in re.findall(rf"({_NUM})([조억만천]?)", s):
            mult = _WORD.get(word, 1)
            total += _to_float(num) * mult
            matched = True
        if not matched:
            return None
        value = total
    else:
        m = re.fullmatch(_NUM, s)
        if not m:
            return None
        value = _to_float(s) * normalize_unit(unit)

    return -value if neg else value


def find_amounts(text: str) -> list[float]:
    """Extract all KRW amounts written with Korean number words in free text.

    Used to pull numeric claims out of generated answers, e.g.
    "매출은 300조 8,709억 원으로" -> [3.008709e14]. Plain numbers without
    a 조/억/만 word or 원 are skipped (they may be counts, years, %).
    """
    pattern = rf"[△▲-]?(?:{_NUM}\s*[조억만]\s*)+(?:{_NUM})?\s*원?|[△▲-]?{_NUM}\s*원"
    out = []
    for m in re.finditer(pattern, text):
        v = parse_krw(m.group(0))
        if v is not None:
            out.append(v)
    return out


def rel_close(a: float, b: float, tol: float = 0.005) -> bool:
    """Relative-tolerance match used for numeric claim accuracy (default ±0.5%)."""
    if b == 0:
        return a == 0
    return abs(a - b) / abs(b) <= tol

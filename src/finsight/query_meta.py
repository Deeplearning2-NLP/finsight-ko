"""Rule-based query metadata: which company and which fiscal year(s) a question is about (R1).

Used as a retrieval filter (docs/decisions.md 2026-10-21). Only the 10 evaluation companies.

- Companies: alias dictionary; each whitespace word is analysed with Kiwi and trailing particles
  (J* tags, e.g. "현대차의", "기아는") are stripped before matching. Brand aliases (KB…, 업비트)
  may start a longer word ("KB스타기업뱅킹") and get lower confidence.
- Fiscal years: "2024 사업연도", "2024년", "2024 년", "2024.12.31", "FY2024"/"FY24", "'24년",
  ranges ("2022~2024년", "2022년부터 2024년까지" -> all years), and "제56기" mapped through
  PERIOD_OFFSET when exactly one company is found.
- Relative expressions (당기/전기/전전기/전년/작년/올해/최근 …) have no fixed reference point here,
  so they are recorded in `spans` (kind="relative") but never turned into years.

extract(q) -> {"companies": [...], "fiscal_years": [...], "confidence": float, "spans": [...]}
"""
from __future__ import annotations

import re
from functools import lru_cache

# canonical name -> aliases (exact word stems). Brand aliases may prefix a longer word.
ALIASES: dict[str, list[str]] = {
    "삼성전자": ["삼성전자", "삼성", "Samsung Electronics"],
    "SK하이닉스": ["SK하이닉스", "에스케이하이닉스", "하이닉스", "SK hynix"],
    "LG전자": ["LG전자", "엘지전자", "LG Electronics"],
    "현대자동차": ["현대자동차", "현대차", "Hyundai Motor"],
    "기아": ["기아", "기아차", "기아자동차", "Kia"],
    "SK텔레콤": ["SK텔레콤", "SK 텔레콤", "에스케이텔레콤", "SKT"],
    "한국전력공사": ["한국전력공사", "한국전력", "한전", "KEPCO"],
    "국민은행": ["국민은행", "KB국민은행", "KB 국민은행", "케이비국민은행"],
    "HMM": ["HMM", "에이치엠엠", "현대상선"],
    "두나무": ["두나무"],
}
BRANDS: dict[str, str] = {"KB": "국민은행", "업비트": "두나무"}  # prefix of a word, lower confidence

# fiscal_year = 기수 + offset, from the report covers (scripts/build_period_map.py)
PERIOD_OFFSET: dict[str, int] = {
    "삼성전자": 1968, "SK하이닉스": 1947, "LG전자": 2001, "현대자동차": 1967, "기아": 1943,
    "SK텔레콤": 1983, "한국전력공사": 1960, "국민은행": 2000, "HMM": 1975, "두나무": 2011,
}

# v1 (tuned on the 100 labelled dev questions): allow "2022년(제55기)부터", "2022년말부터"
_RANGE = re.compile(r"(20\d\d)\s*년?\s*(?:\([^)]{0,20}\))?\s*(?:말|초)?\s*(?:[~∼\-–]|부터)\s*(20\d\d)\s*년?(?:\s*까지)?")
_PERIOD_RANGE = re.compile(r"제\s*(\d{1,3})\s*기\s*(?:[~∼\-–]|부터|에서)\s*제\s*(\d{1,3})\s*기")
_WORD = re.compile(r"[^\s()\[\]{}/,·]+")  # v1: words also end at brackets/slashes ("현대자동차(별도기준)에서")
_YEAR = re.compile(r"(?<!\d)(20\d\d)\s*(?:년|사업\s*연도|회계\s*연도|\.\s?\d{1,2}(?!\d))")
_FY = re.compile(r"FY\s?'?(\d{4}|\d{2})(?!\d)", re.I)
_YY = re.compile(r"['’‘](\d{2})\s?년")
_PERIOD = re.compile(r"제\s*(\d{1,3})\s*기")
_RELATIVE = re.compile(r"전전기|당기|전기(?!차|요금|통신|사용|판매|설비|공급|사업|자동차)|전년|작년|올해|금년|최근|직전")


_PUNCT_TAGS = {"SF", "SP", "SS", "SSO", "SSC", "SE", "SO", "SW"}


@lru_cache(maxsize=1)
def _kiwi():
    from kiwipiepy import Kiwi

    return Kiwi()


def _stem(word: str) -> str:
    """Strip trailing particles (Kiwi J* tags) and punctuation from one whitespace word."""
    word = word.strip("()[]{}\"'“”‘’,.?!·:;")
    if not word:
        return word
    toks = _kiwi().tokenize(word)
    cut = len(word)
    for t in reversed(toks):
        if t.tag.startswith("J") or t.tag in _PUNCT_TAGS:
            cut = t.start
        else:
            break
    return word[:cut]


def _companies(q: str, aliases: dict[str, list[str]] | None = None,
               brands: dict[str, str] | None = None) -> tuple[list[str], list[dict], dict[str, float]]:
    found: dict[str, float] = {}
    spans: list[dict] = []
    alias_to = {a: c for c, al in (aliases or ALIASES).items() for a in al}
    brands = BRANDS if brands is None else brands
    multi = sorted((a for a in alias_to if " " in a), key=len, reverse=True)
    rest = q
    for a in multi:  # multi-word aliases first ("SK 텔레콤", "KB 국민은행")
        for m in re.finditer(re.escape(a), rest):
            found.setdefault(alias_to[a], 1.0)
            spans.append({"kind": "company", "text": a, "value": alias_to[a], "start": m.start()})
        rest = rest.replace(a, " " * len(a))
    for wm in _WORD.finditer(rest):
        word, start = wm.group(0), wm.start()
        stem = _stem(word)
        if stem in alias_to:
            found[alias_to[stem]] = max(found.get(alias_to[stem], 0), 1.0)
            spans.append({"kind": "company", "text": stem, "value": alias_to[stem], "start": start})
            continue
        for b, c in brands.items():
            if stem.startswith(b):
                found.setdefault(c, 0.7)
                spans.append({"kind": "company_brand", "text": stem, "value": c, "start": start})
                break
    return list(found), spans, found


def extract(q: str, aliases: dict[str, list[str]] | None = None,
            period_offset: dict[str, int] | None = None) -> dict:
    """aliases/period_offset default to the 10 evaluation companies; pass others (e.g. training
    companies) to reuse the same rules. With custom aliases no brand prefixes are used."""
    companies, spans, comp_conf = _companies(q, aliases, None if aliases is None else {})
    offsets = PERIOD_OFFSET if period_offset is None else period_offset
    years: set[int] = set()
    conf = [min(comp_conf.values())] if comp_conf else []
    covered: list[tuple[int, int]] = []

    def add_span(kind, m, values, c=1.0):
        spans.append({"kind": kind, "text": m.group(0), "value": values, "start": m.start()})
        covered.append((m.start(), m.end()))
        years.update(values)
        conf.append(c)

    for m in _RANGE.finditer(q):
        a, b = sorted((int(m.group(1)), int(m.group(2))))
        add_span("year_range", m, list(range(a, b + 1)))
    for m in _YEAR.finditer(q):
        if not any(s <= m.start() < e for s, e in covered):
            add_span("year", m, [int(m.group(1))])
    for m in _FY.finditer(q):
        y = int(m.group(1))
        add_span("year_fy", m, [y if y > 100 else 2000 + y])
    for m in _YY.finditer(q):
        add_span("year_short", m, [2000 + int(m.group(1))], 0.9)
    one = companies[0] if len(companies) == 1 and companies[0] in offsets else None
    for m in _PERIOD_RANGE.finditer(q):
        if one:
            a, b = sorted((int(m.group(1)), int(m.group(2))))
            add_span("period_range", m, [n + offsets[one] for n in range(a, b + 1)], 0.9)
    for m in _PERIOD.finditer(q):
        if any(s <= m.start() < e for s, e in covered):
            continue
        if one:
            add_span("period", m, [int(m.group(1)) + offsets[one]], 0.9)
        else:
            spans.append({"kind": "period_unresolved", "text": m.group(0), "value": None, "start": m.start()})
    rel = [m for m in _RELATIVE.finditer(q)]
    for m in rel:
        spans.append({"kind": "relative", "text": m.group(0), "value": None, "start": m.start()})
    if rel and not years:
        conf.append(0.5)
    spans.sort(key=lambda s: s["start"])
    return {"companies": companies, "fiscal_years": sorted(years),
            "confidence": round(min(conf), 2) if conf else 0.0, "spans": spans}

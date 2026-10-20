"""Filing metadata normalization for Dartdoc / OpenDART records.

Key rule: the first 8 digits of ``rcept_no`` are the *filing date*,
not the fiscal year. The fiscal year comes from the report name,
e.g. "[기재정정]사업보고서 (2024.12)" -> fiscal_year 2024.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import date

_PERIOD = re.compile(r"\((\d{4})\.(\d{2})\)")
_AMEND = re.compile(r"^\s*\[(기재정정|첨부정정|첨부추가|변경등록|연장결정|발행조건확정|정정명령부과|정정제출요구)\]")


@dataclass(frozen=True)
class FilingMeta:
    rcept_no: str
    filing_date: date
    report_kind: str          # e.g. "사업보고서", "반기보고서"
    fiscal_year: int | None   # from "(YYYY.MM)"
    period_month: int | None  # fiscal period end month (12 for Dec year-end)
    is_amended: bool


def filing_date_from_rcept_no(rcept_no: str) -> date:
    s = str(rcept_no)
    if not re.fullmatch(r"\d{14}", s):
        raise ValueError(f"rcept_no must be 14 digits: {rcept_no!r}")
    return date(int(s[0:4]), int(s[4:6]), int(s[6:8]))


def parse_report_name(report_name: str) -> tuple[str, int | None, int | None, bool]:
    """Return (report_kind, fiscal_year, period_month, is_amended)."""
    name = report_name.strip()
    amended = bool(_AMEND.match(name))
    core = _AMEND.sub("", name).strip()
    m = _PERIOD.search(core)
    year = month = None
    if m:
        year, month = int(m.group(1)), int(m.group(2))
        core = _PERIOD.sub("", core).strip()
    return core, year, month, amended


def parse_filing(rcept_no: str, report_name: str) -> FilingMeta:
    kind, year, month, amended = parse_report_name(report_name)
    return FilingMeta(
        rcept_no=str(rcept_no),
        filing_date=filing_date_from_rcept_no(rcept_no),
        report_kind=kind,
        fiscal_year=year,
        period_month=month,
        is_amended=amended,
    )

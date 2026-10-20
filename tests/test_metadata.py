from datetime import date

import pytest

from finsight.metadata import filing_date_from_rcept_no, parse_filing, parse_report_name


def test_amended_filing_fiscal_year_differs_from_filing_date():
    # filed 2025-12-31 but covers fiscal year 2024
    m = parse_filing("20251231000123", "[기재정정]사업보고서 (2024.12)")
    assert m.filing_date == date(2025, 12, 31)
    assert m.fiscal_year == 2024
    assert m.period_month == 12
    assert m.is_amended
    assert m.report_kind == "사업보고서"


def test_plain_report():
    assert parse_report_name("반기보고서 (2024.06)") == ("반기보고서", 2024, 6, False)


def test_no_period():
    assert parse_report_name("증권신고서(지분증권)") == ("증권신고서(지분증권)", None, None, False)


def test_bad_rcept_no():
    with pytest.raises(ValueError):
        filing_date_from_rcept_no("2024")

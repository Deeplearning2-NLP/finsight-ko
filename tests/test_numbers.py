import pytest

from finsight.numbers import find_amounts, normalize_unit, parse_krw, rel_close


@pytest.mark.parametrize(
    "text,unit,expected",
    [
        ("1조 2,345억 원", None, 1_234_500_000_000),
        ("1조2,345억원", None, 1_234_500_000_000),
        ("3.5조", None, 3_500_000_000_000),
        ("120억 원", None, 12_000_000_000),
        ("12,345", "백만원", 12_345_000_000),
        ("12,345", "(단위: 백만원)", 12_345_000_000),
        ("△1,234", "천원", -1_234_000),
        ("(1,234)", "백만원", -1_234_000_000),
        ("-5,000", None, -5_000),
        ("1,000원", None, 1_000),
    ],
)
def test_parse_krw(text, unit, expected):
    assert parse_krw(text, unit) == pytest.approx(expected)


@pytest.mark.parametrize("text", ["", "-", "N/A", "abc"])
def test_parse_krw_empty(text):
    assert parse_krw(text) is None


def test_unknown_unit():
    with pytest.raises(ValueError):
        normalize_unit("달러")


def test_find_amounts():
    s = "2024년 매출은 300조 8,709억 원, 영업이익은 32조 7,260억 원으로 전년 대비 증가했다."
    assert find_amounts(s) == pytest.approx([3.008709e14, 3.2726e13])


def test_find_amounts_skips_years_and_percent():
    assert find_amounts("2024년 매출이 20% 증가") == []


def test_rel_close():
    assert rel_close(100.4, 100.0)
    assert not rel_close(101.0, 100.0)

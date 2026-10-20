from finsight.grading import accuracy, grade_value

# LG전자 FY2023 매출액: first reported 84,227,765백만원, restated in the FY2024 report to 82,262,664백만원
LG = {"answer_value": 84_227_765e6, "answer_alt": [82_262_664e6]}


def test_default_and_alt_both_accepted():
    assert grade_value(84_227_765e6, LG)
    assert grade_value(82_262_664e6, LG)
    assert grade_value(84_227_765e6 * 1.004, LG)  # within 0.5%


def test_values_between_or_outside_rejected():
    assert not grade_value(83_200_000e6, LG)  # ~1.2% from both
    assert not grade_value(None, LG)


def test_no_alt_and_scalar_alt():
    assert grade_value(100.0, {"answer_value": 100.0})
    assert not grade_value(102.0, {"answer_value": 100.0, "answer_alt": None})
    assert grade_value(90.0, {"answer_value": 100.0, "answer_alt": 90.0})


def test_negative_and_zero_answers():
    assert grade_value(-887_309e6, {"answer_value": -887_309e6})
    assert not grade_value(887_309e6, {"answer_value": -887_309e6})  # wrong sign
    assert grade_value(0.0, {"answer_value": 0.0})


def test_accuracy():
    assert accuracy([84_227_765e6, None], [LG, LG]) == 0.5

from finsight.query_meta import PERIOD_OFFSET, extract


def test_company_with_particles_and_aliases():
    assert extract("현대차의 2024년 영업이익은?")["companies"] == ["현대자동차"]
    assert extract("기아는 어떤 차종을 생산하나?")["companies"] == ["기아"]
    assert extract("한전 발전 부문 가동률")["companies"] == ["한국전력공사"]
    assert extract("SK 텔레콤이 체결한 이자율스왑")["companies"] == ["SK텔레콤"]
    assert set(extract("삼성전자와 SK하이닉스 중 어디가 큰가?")["companies"]) == {"삼성전자", "SK하이닉스"}


def test_no_false_company_from_substrings():
    assert extract("자동차 산업의 전후방 연관 효과")["companies"] == []
    assert extract("전기 판매 단가는?")["companies"] == []


def test_brand_alias_lower_confidence():
    r = extract("KB스타기업뱅킹 서비스의 개요는?")
    assert r["companies"] == ["국민은행"] and r["confidence"] < 1.0


def test_years_explicit_forms():
    assert extract("2024 사업연도 매출")["fiscal_years"] == [2024]
    assert extract("2024 년 누적 매출")["fiscal_years"] == [2024]
    assert extract("기준일 2024.12.31 현재 잔액")["fiscal_years"] == [2024]
    assert extract("FY23 실적")["fiscal_years"] == [2023]
    assert extract("'24년 하반기 계획")["fiscal_years"] == [2024]


def test_ranges_return_all_years():
    assert extract("2022년부터 2024년까지 추이")["fiscal_years"] == [2022, 2023, 2024]
    assert extract("2022~2024년 매출액")["fiscal_years"] == [2022, 2023, 2024]


def test_period_numbers_need_one_company():
    assert extract("기아의 제81기 가격")["fiscal_years"] == [2024]
    assert extract("HMM 벌크부문 제47기, 제48기 가동률")["fiscal_years"] == [2022, 2023]
    r = extract("제22기부터 제24기까지 조달 합계")  # no company -> unresolved, recorded only
    assert r["fiscal_years"] == [] and any(s["kind"] == "period_unresolved" for s in r["spans"])
    assert PERIOD_OFFSET["삼성전자"] + 56 == 2024


def test_relative_and_non_year_numbers_not_extracted():
    r = extract("당기 설비투자 실적과 전년 대비 변화")
    assert r["fiscal_years"] == [] and any(s["kind"] == "relative" for s in r["spans"])
    assert extract("계약기간이 10년 이상인 계약")["fiscal_years"] == []


def test_v1_brackets_and_range_variants():
    assert extract("현대자동차(별도기준)에서 승용 매출")["companies"] == ["현대자동차"]
    assert extract("베트남(국민은행 베트남지점) 관련")["companies"] == ["국민은행"]
    assert extract("2022년(제55기)부터 2024년(제57기)까지 매출")["fiscal_years"] == [2022, 2023, 2024]
    assert extract("2022년말부터 2024년 3분기말까지")["fiscal_years"] == [2022, 2023, 2024]
    assert extract("기아 승용차 매출은 제79기에서 제81기까지")["fiscal_years"] == [2022, 2023, 2024]


def test_custom_aliases_for_training_companies():
    al = {"롯데카드": ["롯데카드"], "DB손해보험": ["DB손해보험", "DB손보"]}
    r = extract("DB손보의 2024 사업연도 손해율은?", aliases=al)
    assert r["companies"] == ["DB손해보험"] and r["fiscal_years"] == [2024]
    assert extract("KB스타 서비스", aliases=al)["companies"] == []  # no brand prefixes with custom aliases

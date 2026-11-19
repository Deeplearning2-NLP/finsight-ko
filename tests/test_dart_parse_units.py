from finsight.dart_parse import parse_document


def _xml():
    head = "".join(f"<TR><TD>{t}</TD></TR>" for t in
                   ["연결 손익계산서", "제 76 기 2023.01.01 부터 2023.12.31 까지", "제 75 기 2022.01.01 부터 2022.12.31 까지",
                    "(단위 : 백만원)"])
    body = "<TR><TH>과목</TH><TH>제 76 기</TH></TR>" + "".join(
        f"<TR><TD>항목{i}</TD><TD>{i},000</TD></TR>" for i in range(8)) + \
        "<TR><TD>기본주당이익 (단위 : 원)</TD><TD>(12,517)</TD></TR>"
    return (f"<DOCUMENT><BODY><SECTION-1><TITLE>III. 재무에 관한 사항</TITLE><SECTION-2><TITLE>2. 연결재무제표</TITLE>"
            f"<TABLE>{head}</TABLE><TABLE>{body}</TABLE></SECTION-2></SECTION-1></BODY></DOCUMENT>").encode()


def test_unit_fix_off_reproduces_old_behaviour():
    tables = [b for b in parse_document(_xml()) if b["type"] == "table"]
    assert [t["unit"] for t in tables] == ["백만원", "원"]  # old: per-share row note taken as the table unit


def test_unit_fix_r1_r2():
    tables = [b for b in parse_document(_xml(), unit_fix=True) if b["type"] == "table"]
    assert [t["unit"] for t in tables] == ["백만원", "백만원"]

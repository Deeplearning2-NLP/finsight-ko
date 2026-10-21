from finsight.chunking import chunk_text


def ntok(s: str) -> int:  # 1 token per character keeps the test tokenizer-free
    return len(s)


def test_chunks_respect_budget_and_overlap():
    text = "\n".join(f"문장{i:02d}입니다." for i in range(60))  # 8 chars per line
    chunks = chunk_text(text, ntok, max_tok=52, overlap=10)
    assert len(chunks) > 1
    assert all(ntok(c.replace("\n", "")) <= 50 for c in chunks)
    # last line of a chunk is carried over as the first line of the next
    assert chunks[0].split("\n")[-1] == chunks[1].split("\n")[0]


def test_long_sentence_is_hard_split():
    chunks = chunk_text("가" * 300, ntok, max_tok=102, overlap=10)
    assert all(len(c) <= 100 for c in chunks)
    assert "".join(chunks) == "가" * 300


def test_bare_angle_brackets_do_not_swallow_sections():
    from finsight.dart_parse import parse_document
    xml = ("<DOCUMENT><BODY><SECTION-1><TITLE>II. 사업의 내용</TITLE><TABLE><TBODY><TR><TD><금융자산></TD>"
           "<TD>1</TD></TR></TBODY></TABLE></SECTION-1>"
           "<SECTION-1><TITLE>III. 재무에 관한 사항</TITLE><P>본문</P></SECTION-1></BODY></DOCUMENT>").encode()
    secs = {b["sec1"] for b in parse_document(xml)}
    assert secs == {"II. 사업의 내용", "III. 재무에 관한 사항"}


def test_ascii_prefixed_bracket_text_is_not_a_tag():
    from finsight.dart_parse import parse_document
    xml = ("<DOCUMENT><BODY><SECTION-1><TITLE>I. 회사의 개요</TITLE><P><JB금융지주></P></SECTION-1>"
           "<SECTION-1><TITLE>II. 사업의 내용</TITLE><P>본문</P></SECTION-1></BODY></DOCUMENT>").encode()
    assert {b["sec1"] for b in parse_document(xml)} == {"I. 회사의 개요", "II. 사업의 내용"}

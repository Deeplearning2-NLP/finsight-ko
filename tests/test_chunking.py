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

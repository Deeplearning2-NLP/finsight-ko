"""Korean tokenizers for lexical retrieval (BM25)."""
from __future__ import annotations

from functools import lru_cache

# content-word POS tags (Kiwi / Sejong tagset); verbs & adjectives matched by prefix
CONTENT_TAGS = {"NNG", "NNP", "NNB", "NR", "SL", "SN", "SH", "XR", "MAG"}


@lru_cache(maxsize=1)
def _kiwi():
    from kiwipiepy import Kiwi

    return Kiwi()


def whitespace(text: str) -> list[str]:
    return text.split()


def morphs(text: str) -> list[str]:
    """All morphemes (Kiwi)."""
    return [t.form for t in _kiwi().tokenize(text)]


def content_morphs(text: str) -> list[str]:
    """Content morphemes only: nouns, numbers, foreign words, verb/adj stems."""
    return [
        t.form
        for t in _kiwi().tokenize(text)
        if t.tag in CONTENT_TAGS or t.tag.startswith(("VV", "VA"))
    ]


TOKENIZERS = {
    "whitespace": whitespace,
    "kiwi_all": morphs,
    "kiwi_content": content_morphs,
}

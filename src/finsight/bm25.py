"""BM25 retriever over a list of (doc_id, text) with a pluggable Korean tokenizer."""
from __future__ import annotations

from typing import Callable, Sequence

import numpy as np
from rank_bm25 import BM25Okapi

from .tokenize import TOKENIZERS


class BM25Retriever:
    def __init__(self, doc_ids: Sequence[str], texts: Sequence[str], tokenizer: str | Callable = "kiwi_content"):
        self.tok = TOKENIZERS[tokenizer] if isinstance(tokenizer, str) else tokenizer
        self.doc_ids = list(doc_ids)
        self.bm25 = BM25Okapi([self.tok(t) for t in texts])

    def search(self, query: str, k: int = 10) -> list[str]:
        scores = self.bm25.get_scores(self.tok(query))
        top = np.argsort(-scores)[:k]
        return [self.doc_ids[i] for i in top]

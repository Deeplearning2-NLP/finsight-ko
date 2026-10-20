"""Token-budget chunking on line / sentence boundaries with overlap."""
from __future__ import annotations

import re
from typing import Callable

SENT_SPLIT = re.compile(r"(?<=[다요음함됨임\.\?!])\s+(?=\S)")


def split_units(text: str, ntok: Callable[[str], int], max_unit: int = 200) -> list[str]:
    """Lines first; long lines are split into sentences."""
    units: list[str] = []
    for line in text.split("\n"):
        line = line.strip()
        if not line:
            continue
        units += [line] if ntok(line) <= max_unit else [s for s in SENT_SPLIT.split(line) if s]
    return units


def chunk_text(text: str, ntok: Callable[[str], int], max_tok: int = 512, overlap: int = 50) -> list[str]:
    """Greedy pack units to <= max_tok (incl. 2 special tokens); carry last <= overlap tokens."""
    budget = max_tok - 2
    units = [(u, ntok(u)) for u in split_units(text, ntok)]
    chunks: list[str] = []
    cur: list[tuple[str, int]] = []
    cur_tok = 0
    for u, n in units:
        if n > budget:  # pathological long sentence: hard split by characters
            if cur:
                chunks.append("\n".join(x for x, _ in cur))
                cur, cur_tok = [], 0
            step = max(50, int(len(u) * budget / n * 0.9))
            chunks += [u[i:i + step] for i in range(0, len(u), step)]
            continue
        if cur and cur_tok + n > budget:
            chunks.append("\n".join(x for x, _ in cur))
            carry, t = [], 0
            for x, m in reversed(cur):
                if t + m > overlap:
                    break
                carry.insert(0, (x, m))
                t += m
            cur, cur_tok = (carry, t) if t + n <= budget else ([], 0)
        cur.append((u, n))
        cur_tok += n
    if cur:
        chunks.append("\n".join(x for x, _ in cur))
    return chunks

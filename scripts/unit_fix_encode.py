"""G1: M3-hard corpus embeddings for a unit-fixed corpus: copy emb_m_hard.npy and re-encode only the changed chunks
(ids listed in outputs/midterm/unit_audit.json). Usage: CUDA_VISIBLE_DEVICES=0 python scripts/unit_fix_encode.py AB|ABC"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
M = ROOT / "outputs/midterm"
tag = sys.argv[1]
audit = json.loads((M / "unit_audit.json").read_text())
changed = set(audit["versions"][tag]["changed_ids"])
c = pd.read_json(ROOT / f"data/processed/corpus_a_unitfix_{tag}.jsonl", lines=True, dtype={"chunk_id": str})
orig = pd.read_json(ROOT / "data/processed/corpus_a.jsonl", lines=True, dtype={"chunk_id": str})
assert list(c.chunk_id) == list(orig.chunk_id), "chunk order must match the original corpus"
E = np.load(M / "emb_m_hard.npy").astype(np.float16)
idx = [i for i, cid in enumerate(c.chunk_id) if cid in changed]
from sentence_transformers import SentenceTransformer
m = SentenceTransformer(str(M / "kure-m-hard"), device="cuda")
m.max_seq_length = 512
E[idx] = m.encode(c.text.iloc[idx].tolist(), batch_size=64, normalize_embeddings=True).astype(np.float16)
np.save(M / f"emb_m_hard_unitfix_{tag}.npy", E)
print("re-encoded", len(idx), "of", len(c))

"""Mine retriever hard negatives for reranker training (training corpus only, 20 non-evaluation companies).

For each M1 pair, rank the same company's chunks in corpus_train with the fine-tuned embedding (M3 hard) and keep
the 3 best-ranked chunks that are (a) not the positive, (b) in a different section_path than the positive (same
section in another year can be a correct answer when the question omits the year, decisions 2026-11-17), and
(c) ranked 3rd or lower among those (the top 2 are most likely unlabelled positives). Nothing from corpus_a.

Usage: CUDA_VISIBLE_DEVICES=1 python scripts/mine_negatives_m.py
Output: data/train/pairs_m1_mined.jsonl (pair_id, mined_ids, mined texts), outputs/midterm/mined_stats.json
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs" / "midterm"


def main() -> None:
    from sentence_transformers import SentenceTransformer
    pairs = pd.read_json(ROOT / "data/train/pairs_m1.jsonl", lines=True, dtype={"positive_id": str})
    c = pd.read_json(ROOT / "data/processed/corpus_train.jsonl", lines=True, dtype={"chunk_id": str})
    cache = OUT / "emb_train_m_hard.npy"
    m = SentenceTransformer(str(OUT / "kure-m-hard"), device="cuda")
    m.max_seq_length = 512
    if cache.exists():
        E = np.load(cache).astype(np.float32)
    else:
        E = m.encode(c.text.tolist(), batch_size=64, normalize_embeddings=True, show_progress_bar=False)
        np.save(cache, E.astype(np.float16))
    Q = m.encode(pairs.anchor.tolist(), batch_size=64, normalize_embeddings=True)
    ids, sec, corp = c.chunk_id.to_numpy(), c.section_path.to_numpy(), c.corp_name.to_numpy()
    by_corp = {k: np.where(corp == k)[0] for k in set(corp)}
    sec_of = dict(zip(ids, sec))
    out, n_rank = [], []
    for i, r in enumerate(pairs.itertuples()):
        pool = by_corp[r.corp_name]
        s = E[pool] @ Q[i]
        order = pool[np.argsort(-s)][:40]
        psec = sec_of.get(r.positive_id, r.section_path)
        cand = [j for j in order if ids[j] != r.positive_id and sec[j] != psec]
        pos_rank = next((k for k, j in enumerate(order) if ids[j] == r.positive_id), 40)
        n_rank.append(pos_rank)
        pick = cand[2:5]
        out.append({"pair_id": r.pair_id, "mined_ids": [ids[j] for j in pick], "mined": [c.text.iat[j] for j in pick]})
    with open(ROOT / "data/train/pairs_m1_mined.jsonl", "w", encoding="utf-8") as f:
        for o in out:
            f.write(json.dumps(o, ensure_ascii=False) + "\n")
    st = {"pairs": len(out), "mined_per_pair": float(np.mean([len(o["mined_ids"]) for o in out])),
          "positive_rank_top1": float(np.mean(np.array(n_rank) == 0)), "positive_rank_top10": float(np.mean(np.array(n_rank) < 10))}
    (OUT / "mined_stats.json").write_text(json.dumps(st, indent=1))
    print(st)


if __name__ == "__main__":
    main()

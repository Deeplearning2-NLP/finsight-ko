#!/usr/bin/env bash
# Download AutoRAG Korean retrieval benchmark (114 queries / 720 docs) into data/raw/autorag/
set -euo pipefail
OUT="${1:-data/raw/autorag}"
mkdir -p "$OUT"
BASE=https://raw.githubusercontent.com/Marker-Inc-Korea/AutoRAG-example-korean-embedding-benchmark/main/data
curl -sSL -o "$OUT/qa.parquet" "$BASE/qa_v4.parquet"
curl -sSL -o "$OUT/corpus.parquet" "$BASE/ocr_corpus_v3.parquet"
echo "saved to $OUT ($(date -I))"

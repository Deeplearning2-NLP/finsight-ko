# FinSight-ko

한국어 DART 사업보고서를 기업·사업연도별로 검색·비교하고, 재무 수치는 OpenDART 공식 값으로 검증하는 질의응답 시스템.
성균관대 DSC3033 딥러닝2: 자연어처리 기말 프로젝트 (6팀).

> 본 프로젝트의 출력은 투자 권유가 아닙니다.

## 핵심 실험
1. **공시 도메인 검색 임베딩 미세조정** — 공백 BM25 → 형태소 BM25 → 공개 임베딩 → 미세조정 임베딩 → 하이브리드
2. **수치 검증** — LLM 단독 / RAG / RAG + OpenDART 조회·계산

## 빠른 시작
```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
pip install -e .
cp .env.example .env   # OPENDART_API_KEY 입력
pytest -q

# 파일럿: AutoRAGRetrieval BM25 비교
bash scripts/download_autorag.sh
python scripts/eval_autorag_bm25.py
```

## 파일럿 결과 (AutoRAGRetrieval, 질문 114 / 문서 720)
| 모델 | R@1 | R@5 | MRR@10 |
|---|---|---|---|
| BM25 공백 | 0.667 | 0.825 | 0.733 |
| BM25 Kiwi 형태소 | 0.746 | 0.939 | 0.822 |
| BM25 Kiwi 내용어 | **0.851** | **1.000** | **0.914** |
| (리더보드) bge-m3 | 0.675 | 0.930 | – |

질문이 LLM으로 생성되어 BM25에 유리한 벤치마크이므로 보조 지표로만 사용한다.

## 구조
```
src/finsight/   tokenize · bm25 · metrics · numbers(숫자 정규화) · metadata(사업연도 파싱)
scripts/        데이터 수집 · 평가 · 학습
tests/          pytest
docs/           decisions.md(결정 로그) · data.md(데이터 출처)
data/           git-ignored
```

## 데이터
Dartdoc (CC-BY-4.0), OpenDART API, FinDartBench (CC BY-NC 4.0), AutoRAGRetrieval. 원본 데이터와 API 키는 커밋하지 않는다. 자세한 내용은 `docs/data.md`.

## 팀
백윤우 · 임병건 · 박채영

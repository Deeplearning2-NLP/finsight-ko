# 결정 로그

새 결정은 위에 추가한다. 형식: 날짜 — 결정 — 이유.

## 2026-10-19 교수님 피드백 반영
- **직접 학습 모델 필수** → 검색 임베딩 대조 학습 미세조정을 대표 실험으로. 공개 임베딩 vs 공시 미세조정 임베딩 Recall@k.
- LangGraph 에이전트·Streamlit 데모·Langfuse는 확장 목표(13주차 이후).
- 평가셋 200문항: 유형 5개 × 40, dev 50 / test 150, 약 30% 사람 작성, 답 없는 질문 약 10%.
- 표 처리를 보여주기 위해 OpenDART 원문(document.xml) 파싱 추가 (Dartdoc은 표 제거됨).
- 범위는 업종별 10–20개사 × 3–5년 권고 → Dartdoc 집계 후 확정 (후보: FinDartBench 10개사 × 사업연도 2022–2024).

## 2026-10-16 v2 (GPT 리뷰 검토)
- 접수일(`rcept_no` 앞 8자리)과 사업연도(보고서명) 분리, 정정공시는 최신본만.
- FinDartBench를 QA 평가에 채택 (context 청크로 검색 평가도 가능).
- 수치: LLM이 숫자를 만들지 않음. OpenDART 조회 + Python 계산, LLM은 설명만.
- 지표: Numeric Claim Accuracy / Coverage / Unsupported Claim Rate.
- OpenDART `013` = 조회 결과 없음 (연결재무제표 부재로 단정 금지).

## 2026-10-16 주제 확정
- FinSight: Dartdoc + OpenDART, 형태소 하이브리드 검색 + 재무 API 수치 검증.
- AutoRAGRetrieval은 보조 벤치마크 (LLM 생성 질문이라 BM25 편향 확인됨).

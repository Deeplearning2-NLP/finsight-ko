# 프로포절 전 추가 작업 (P1–P5, 10/21–10/22)

범위는 **안 A로 확정**(`docs/decisions.md` 2026-10-20). 목표: 발표에서 "계획"이 아니라 "이미 만들어 둔 것"으로 말할 수 있게 평가 코퍼스와 수치 정답을 실제로 만든다.
결과는 `docs/proposal-results.md`에 **"9. 안 A 코퍼스·수치 정답 (P1–P4)"** 절로 추가하고, `docs/feedback-response.md` 상태를 갱신한다. 데이터·키는 커밋 금지. OpenDART 호출은 전부 캐시.
LLM은 이번 단계에서 쓰지 않는다 (학습 쌍·평가셋 LLM 생성은 9주차).

## P1. 원문 30건 수집·파싱·청킹
- 10개사(삼성전자, SK하이닉스, LG전자, 현대자동차, 기아, SK텔레콤, 한국전력공사, 국민은행, HMM, 두나무) × 사업연도 2022–2024 사업보고서.
- `list.json`에서 보고서명 `(YYYY.12)`로 사업연도를 판정하고(`finsight.metadata`), 같은 사업연도에 정정본이 있으면 **최신 접수본**. 결산월이 12월이 아닌 회사가 있으면 기록.
- T3의 `opendart_parse_one.py` 로직을 함수로 정리해 `scripts/build_corpus.py`로 30건 일괄 처리. 섹션은 II(사업의 내용)·IV(경영진단) 우선, 가능하면 전체도.
- 표는 Markdown 방식 + 청크가 나뉘면 머리글 반복 (T3 비교 결과). 행 이름은 첫 열만 쓰는 규칙 반영.
- 청킹: KURE 토크나이저 기준 512 이하, overlap 50, **청크를 합친 뒤 최종 토큰 수를 다시 재서 512 초과를 0으로** (T3에서 18개 초과 발견).
- 청크 메타데이터: `chunk_id, corp_code, corp_name, fiscal_year, rcept_no, filing_date, is_amended, section_path, is_table, n_tokens`.
- 출력: `data/processed/corpus_a.jsonl`. 통계: 회사 × 연도 청크 수, 표 청크 비율, 토큰 분포, 수집 실패 목록.

## P2. 수치 정답 테이블
- 30건 각각 `fnlttSinglAcntAll` (CFS 우선, `013`이면 OFS로 대체하고 `fs_div` 기록).
- 핵심 계정: 매출액(수익), 영업이익, 당기순이익, 자산총계, 부채총계, 자본총계. 계정명이 회사마다 다르므로(`영업이익(손실)`, `수익(매출액)`, 은행·금융사는 매출액 계정이 없음) `account_id` 우선 매핑 표를 만들고, 매핑 안 되는 계정은 비워둔다(억지로 채우지 않음).
- 출력: `data/processed/financials_a.csv` (corp, fiscal_year, fs_div, account_std, account_nm_raw, account_id, value_krw, rcept_no). 회사 × 계정 커버리지 표를 결과 문서에.

## P3. 수치형 평가 문항 자동 생성 (템플릿, LLM 없음)
- P2 테이블로 유형 1(단일 수치)·2(연도 비교, 수치)·3(기업 비교, 같은 업종 공통 계정) 문항을 템플릿으로 생성. 정답은 원 단위 값 + Python으로 계산한 증감액·증감률.
- 같은 업종 쌍: 삼성전자–SK하이닉스, 현대자동차–기아, (나머지는 업종이 달라 기업 비교 제외).
- 문항마다 `qid, type, template, question, answer_value, answer_unit, source(rcept_no, account_id, fs_div), split=None` 저장 → `data/eval/numeric_auto.jsonl`. 생성 방식 `template`으로 표시.
- 결과 문서에 유형별 문항 수와 예시 3개. dev/test 분할은 9주차 평가셋 확정 때 같이 한다.

## P4. 원문 코퍼스에서 검색 예비 결과 다시 재기
- FinDartBench 사업보고서 문항 중 10개사 × 2022–2024에 해당하는 것을 골라, 각 문항의 `context`가 P1 코퍼스의 어느 청크에 해당하는지 **문자 겹침(예: 문장 단위 일치율 또는 character n-gram Jaccard)으로 정답 청크를 찾는다.** 겹침 기준과 매핑 실패율을 기록하고, 매핑된 문항만 사용.
- T4와 같은 분할(`outputs/proposal/splits.json`)을 유지해 test 문항만 평가. 코퍼스는 P1 전체 청크(수천 개) → T4(784청크)보다 어려운 조건.
- 비교: BM25 공백 / Kiwi 내용어 / KURE-v1 공개 / KURE-v1 스모크 미세조정 / RRF. 지표 R@1/5/10, MRR@10.
- 하이브리드 가중치 등은 dev로만. T4 결과와 나란히 표로.
- 시간이 부족하면 P4는 건너뛰고 P1–P3만 확실히.

## P5. 정리
- `docs/proposal-results.md` 9절 + 맨 위 "슬라이드별 핵심 숫자" 표에 P1–P4 숫자 반영 (예: "원문 30건, 청크 N개, 표 청크 X%", "수치형 자동 문항 N개").
- `docs/feedback-response.md` 갱신, Notion 회의록에 10/21–22 결과 항목 추가(가능하면), 커밋·push.

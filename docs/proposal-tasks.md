# 10/23(금) 프로포절 발표 준비 작업 지시서

목표: 교수님 "발표 전 꼭 할 일" 4개를 **실제 데이터와 숫자로** 채운다. 발표는 5분, 슬라이드 6–8장.
모든 결과는 `outputs/proposal/`에 저장하고(git-ignored), 슬라이드에 쓸 표·숫자·예시는 `docs/proposal-results.md`에 정리해 커밋한다. 데이터·키는 커밋 금지.

진행 순서대로 하고, 각 단계가 끝나면 `docs/feedback-response.md`의 상태를 갱신한다. 막히면 원인과 대안을 `docs/proposal-results.md`에 적고 다음 단계로 넘어간다.

---

## T0. 환경 점검 (먼저)
- `nvidia-smi`, `df -h ~`, `du -sh ~`, `free -h`, `python3 --version`
- 외부 접속: `curl -sI` 로 github.com, huggingface.co, pypi.org, opendart.fss.or.kr
- `python -m venv .venv && source .venv/bin/activate && pip install -r requirements.txt && pip install -e .`
- GPU 단계용: `torch`(서버 CUDA에 맞게), `transformers`, `sentence-transformers`, `datasets`, `faiss-cpu`
- `export HF_HOME=~/hf_cache` (`.bashrc`에도)
- `pytest -q`
- 결과를 `docs/handoff.md`의 "환경 점검 결과" 표에 채우기

## T1. 데이터 받기
- Dartdoc: `datasets.load_dataset("chaannwooff/Dartdoc")` 또는 `huggingface-cli download` → `data/raw/dartdoc/`
- FinDartBench: `davidkim205/FinDartBench` → `data/raw/findartbench/`
- AutoRAG: `bash scripts/download_autorag.sh`
- `docs/data.md`에 날짜·방법·건수 기록. 이 단계 코드는 `scripts/download_data.py`로.

## T2. Dartdoc 통계와 범위 확정 (꼭 할 일 1번)
`scripts/dartdoc_stats.py` 작성:
1. `report_type`으로 사업보고서만 필터. `finsight.metadata.parse_report_name`으로 `fiscal_year`, `is_amended` 추출. `rcept_no`에서 `filing_date`.
2. 같은 회사·같은 사업연도에 여러 보고서가 있으면 최신 접수본만 남기기 (정정공시 처리). 제거 전후 개수 기록.
3. 회사 × 사업연도(2020–2024) 청크 수 피벗. 섹션(II/IV)별로도.
4. 후보: FinDartBench 10개사(삼성전자, SK하이닉스, LG전자, 현대자동차, 기아, SK텔레콤, 한국전력, 국민은행/KB, HMM, 두나무)의 2022–2024 커버리지 확인. 이름 표기 차이 주의.
5. **범위 제안:** 업종별로 묶어 10–20개사 × 사업연도 2022–2024 중 세 해 모두 있는 기업. 업종 분류는 회사명 기준 수작업으로 충분.
6. 길이 분포: 문자 수, 그리고 토크나이저별 토큰 수(`nlpai-lab/KURE-v1`, `klue/bert-base`)의 평균·p50·p90·max, **512토큰 초과 비율**.
7. 슬라이드용 실제 샘플 3–5개 (회사·사업연도·섹션·앞 200자).
출력: `outputs/proposal/dartdoc_coverage.csv`, `docs/proposal-results.md`의 "데이터" 절.

## T3. 보고서 1개 파싱·청킹 (표 포함) (꼭 할 일 4번 전반)
- `.env`에 `OPENDART_API_KEY`가 있어야 한다. 없으면 이 단계는 건너뛰고 "키 발급 대기"로 적는다 (발급: opendart.fss.or.kr 인증키 신청, 개인 즉시 발급).
- `scripts/opendart_parse_one.py`:
  1. `corpCode.xml`로 삼성전자 corp_code 찾기 → `list.json`으로 2024 사업연도 사업보고서 `rcept_no` 찾기
  2. `document.xml`(ZIP 안 XML) 다운로드 → 목차 기준 섹션 분리, 본문과 **표**를 구분
  3. 표→텍스트 변환 방식 2가지 비교: (a) Markdown 표 (b) "행이름 | 열이름 | 값" 직렬화. 각각 토큰 수 확인
  4. 512토큰 이하로 청킹(문단·문장 경계, overlap 50), 청크 수·길이 분포
  5. 같은 회사·연도의 `fnlttSinglAcntAll`(CFS, 11011)로 매출액·영업이익 공식 값 조회 → `finsight.numbers`로 파싱한 표 값과 일치하는지 1–2개 확인
- 출력: 파싱 예시(섹션 하나, 표 하나의 원본 vs 변환 결과)를 `docs/proposal-results.md`에. API 응답 원본은 `data/raw/opendart/`.
- 호출은 캐시하고 최소한으로.

## T4. 공시 데이터 검색 예비 결과 (꼭 할 일 4번 후반)
FinDartBench로 공시 도메인 검색 평가를 만든다 (질문마다 근거 `context`가 있음).
1. `doc_type == 사업보고서`만, 가능하면 T2의 대상 기업만.
2. 코퍼스 = 선택된 문항들의 고유 `context`. 질문 → 자기 context가 정답.
3. **문서 단위(`doc_id`) 분할:** dev / test, 그리고 T6용 train 문서를 서로 겹치지 않게 나누고 분할 파일을 `outputs/proposal/splits.json`에 저장 (seed 고정).
4. 비교 (test 문서 기준):
   - BM25 공백 / Kiwi 전체 / Kiwi 내용어 (`finsight.bm25`)
   - 공개 임베딩 dense: `intfloat/multilingual-e5-base`(또는 large), `BAAI/bge-m3`, `nlpai-lab/KURE-v1` (GPU)
   - (가능하면) BM25 + 최고 dense RRF 하이브리드
5. 지표: Recall@1/5/10, MRR@10 (`finsight.metrics`). 청크가 512토큰을 넘으면 잘림 비율도 기록.
6. 틀린 예 10개 정도를 읽고 유형 메모 (연도 혼동, 숫자 질문, 표 질문, 동의어 등) — 오류 분석 예고용.
7. 주의: FinDartBench 질문도 LLM이 생성 → BM25 유리 편향 가능. 결과 해석에 명시.
`scripts/eval_findartbench_retrieval.py`, 결과표는 `docs/proposal-results.md`의 "검색 예비 결과".

## T5. 질문 유형표 (꼭 할 일 3번)
`docs/question_types.md`: 5유형 × 정의 · 정답 형태 · 평가 방법 · **실제 데이터에 근거한 예시 2개**(회사·사업연도·근거 rcept_no 또는 OpenDART 계정).
- 단일 수치: 정답은 OpenDART 값 (예: 2024 사업연도 연결 영업이익)
- 연도 비교: 같은 회사 두 사업연도 (수치 또는 서술)
- 기업 비교: 같은 업종 두 회사, 공통 계정만
- 서술 요약: 사업의 내용/위험 요인 요약, 근거 청크 표시
- 답 없는 질문: 보고서에 없는 정보 (예: 미래 주가)
예시는 실제 청크/API 값을 확인한 뒤 쓴다. 비교형 예시는 사람이 다듬을 수 있게 표시.

## T6. (시간 되면) 미세조정 스모크 테스트 — 발표 임팩트 큼
"직접 학습 모델"이 실제로 돌아간다는 증거.
- train 문서의 FinDartBench (질문, context) 쌍으로 `nlpai-lab/KURE-v1`(또는 `klue/roberta-base` 기반 문장 임베딩)을 `MultipleNegativesRankingLoss`로 1 epoch, max_seq_length 512, 작은 배치.
- T4 test에서 미세조정 전후 Recall@5/MRR 비교. 한 번만, seed 고정, test로 튜닝하지 말 것(하이퍼파라미터는 dev로).
- 학습 시간·GPU 메모리 기록. 체크포인트는 1개만 저장하고 디스크 확인.
- 결과가 안 좋아도 그대로 기록 (원인 메모). 본 실험은 Dartdoc에서 LLM으로 만든 학습 쌍으로 다시 한다는 계획을 적는다.

## T7. 정리
- `docs/proposal-results.md` 구성: 데이터(개수·범위·샘플·라이선스) / 범위 제안 / 입력 길이(토큰 분포, 512 초과 비율, 청킹) / 파싱 예시 / 검색 예비 결과 표 / 질문 유형 요약 / (T6) 미세조정 스모크 결과 / 한계와 다음 단계
- 슬라이드 7장 구성(문제 → 데이터 → 평가 → 모델 단계(직접 학습 표시) → 차별점·어려움 → 계획·역할 → 입력 길이·모델 계열)에 맞춰 각 장에 들어갈 숫자 1–2개를 맨 위에 요약
- `docs/feedback-response.md` 상태 갱신
- 코드·문서만 커밋해서 push (Co-Authored-By 표기). push가 안 되면 `gh auth login` 또는 PAT 설정을 사용자에게 요청

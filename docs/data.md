# 데이터 출처·수집 기록

수집할 때마다 한 줄씩 추가한다 (날짜, 누가, 무엇을, 어떻게, 건수).

| 날짜 | 담당 | 데이터 | 방법 | 건수·비고 |
|---|---|---|---|---|
| 2026-10-16 | 임병건 | AutoRAGRetrieval | `scripts/download_autorag.sh` (GitHub raw) | 질문 114, 문서 720 |
| 2026-10-20 | Claude Code (서버) | AutoRAGRetrieval | `python scripts/download_data.py --only autorag` | 질문 114, 문서 720 (재확인) |
| 2026-10-20 | Claude Code (서버) | Dartdoc | `python scripts/download_data.py --only dartdoc` (HF snapshot, jsonl 25개, 1.1GB) | 256,548청크 / 사업보고서 189,812청크·10,418건·3,426개사. 사업연도별 최신본만 남기면 8,702건·156,612청크 |
| 2026-10-20 | Claude Code (서버) | FinDartBench | `python scripts/download_data.py --only findartbench` (149MB) | 14,444 QA / 사업보고서 5,638 QA, 고유 context 195개, doc_id 486개 |
| 2026-10-20 | Claude Code (서버) | OpenDART 재무제표 | `fnlttSinglAcntAll` (CFS, 11011, 2024) via `finsight.opendart` | 삼성전자·SK하이닉스·현대자동차·기아·HMM·한국전력공사 6건, `data/raw/opendart/` 캐시 |
| 2026-10-20 | Claude Code (서버) | OpenDART 원문 | `corpCode.xml`, `list.json`, `document.xml` | corpCode 1건, list 6건, 원문 1건(삼성전자 20250311001085, ZIP 내 XML 3개, 본문 6.3MB) |
| 2026-10-20 | Claude Code (서버) | OpenDART 원문 30건 (안 A) | `scripts/build_corpus.py`: list.json 10회 + document.xml 30건 (캐시) | 10개사 × FY2022–2024 30건 → `data/processed/corpus_a.jsonl` 34,427청크. 국민은행은 corp_code 00386937 |
| 2026-10-20 | Claude Code (서버) | OpenDART 재무제표 30건 | `scripts/build_financials.py`: fnlttSinglAcntAll 30건 (+국민은행 FY2022 OFS 1건) | `data/processed/financials_a.csv` 180행 중 169칸 값 있음 |
| 2026-10-20 | Claude Code (서버) | 수치형 자동 문항 | `scripts/make_numeric_questions.py` (템플릿, LLM 없음) | `data/eval/numeric_auto.jsonl` 369문항 |
| 2026-10-20 | Claude Code (서버) | 학습 쌍 파일럿 원천 | `scripts/pairs_pilot_chunks.py` (Dartdoc, 평가 10개사 제외 20개사 × FY2022–2024, II·IV) | 512토큰 재분할 풀 8,883개, 원천 240청크 → `data/train/pilot_*.jsonl` |
| 2026-10-21 | Claude Code (서버) | OpenDART 원문 60건 (학습용 20개사) | `scripts/build_corpus.py --set train`: list.json 22회 + document.xml 60건 (캐시, 오늘 약 82회) | `data/processed/corpus_train.jsonl` 48,712청크, `filings_train.csv`. 현대엔지니어링 corp_code 00349927 |
| 2026-10-21 | Claude Code (서버) | 질문 메타데이터 라벨 | FinDartBench dev 130문항을 직접 읽고 회사·사업연도 라벨 (qid와 라벨만 저장) | `data/eval/r1_meta_labels.json` (100 + held-out 30) |
| 2026-10-21 | Claude Code (서버) | 학습 쌍 재파일럿 (원문) | `pairs_r2_chunks.py` → `pairs_pilot_generate.py --prefix r2` → `pairs_r2_build.py` | 360청크 → 질문 716 → 필터 후 517 → `data/train/pairs_r2.jsonl` 300쌍 |

## 이용 조건 확인
- Dartdoc: CC-BY-4.0 — 출처 표기.
  - 확인된 한계 (2026-10-20): FinDartBench 10개사 중 7개사는 사업연도 2022–2023 사업보고서가 없음 (삼성전자는 FY2024 섹션 II 11청크뿐, IV 없음). 표의 숫자 셀이 대부분 빠지고 머리글만 남음 (예: 한국전력 II/6 "나. 생산실적 (단위 : GWh) 사업구분 구분 2024년 …"에 값 없음). 같은 문단이 II와 IV에 중복되는 경우 있음 (HMM FY2024 II/8 = IV/1).
- FinDartBench: CC BY-NC 4.0 — 수업용 비상업 사용, 재배포 금지.
- OpenDART: 무료 인증키(개인 1개), 일 호출 한도 있음(약 2만 건으로 알려짐, 이용현황에서 확인), 키 비공개.
- AutoRAG 벤치마크: 저장소 라이선스 확인 후 기재.

## LLM 생성 데이터
| 데이터 | 생성 모델 | 생성 수 | 사람 검수 수 | 사람 직접 작성 수 |
|---|---|---|---|---|
| 평가셋 | | | | |
| 미세조정 학습 쌍 (재파일럿 R2, 원문) | Qwen/Qwen2.5-7B-Instruct (snapshot a09a3545, vLLM 0.31.0, T=0.7, top_p=0.9, seed 42) | 716 질문 → 필터 517 → 300쌍 | 30쌍 직접 판정 (답 가능 20 / 베끼기 4 / 모호 3 / 오류 3) | 0 |
| 미세조정 학습 쌍 (파일럿) | Qwen/Qwen2.5-7B-Instruct (snapshot a09a3545, vLLM 0.31.0, T=0.7, top_p=0.9, seed 42) | 474 질문 → 300쌍 | 30쌍 직접 판정 (답 가능 9 / 베끼기 4 / 모호 5 / 오류 12) | 0 |

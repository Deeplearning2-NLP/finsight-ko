# 데이터 출처·수집 기록

수집할 때마다 한 줄씩 추가한다 (날짜, 누가, 무엇을, 어떻게, 건수).

| 날짜 | 담당 | 데이터 | 방법 | 건수·비고 |
|---|---|---|---|---|
| 2026-10-16 | 임병건 | AutoRAGRetrieval | `scripts/download_autorag.sh` (GitHub raw) | 질문 114, 문서 720 |
| | | Dartdoc | HF `chaannwooff/Dartdoc` | |
| | | FinDartBench | HF `davidkim205/FinDartBench` | |
| | | OpenDART 재무제표 | `fnlttSinglAcntAll` | |
| | | OpenDART 원문 | `document.xml` | |

## 이용 조건 확인
- Dartdoc: CC-BY-4.0 — 출처 표기.
- FinDartBench: CC BY-NC 4.0 — 수업용 비상업 사용, 재배포 금지.
- OpenDART: 무료 인증키(개인 1개), 일 호출 한도 있음(약 2만 건으로 알려짐, 이용현황에서 확인), 키 비공개.
- AutoRAG 벤치마크: 저장소 라이선스 확인 후 기재.

## LLM 생성 데이터
| 데이터 | 생성 모델 | 생성 수 | 사람 검수 수 | 사람 직접 작성 수 |
|---|---|---|---|---|
| 평가셋 | | | | |
| 미세조정 학습 쌍 | | | | |

# 인수인계 문서 (2026-10-20)

새 Claude Code 세션이나 새 팀원이 이 프로젝트를 이어받을 때 읽는 문서. 프로젝트 규칙은 `CLAUDE.md`, 남은 일은 `docs/feedback-response.md`, 이번 주 작업은 `docs/proposal-tasks.md`.

## 링크
- GitHub: https://github.com/Deeplearning2-NLP/finsight-ko (조직 이름이 Deaplearning2-NLP에서 바뀜. 예전 주소는 자동 연결됨)
- Notion: "FinSight — 딥러닝2 기말 프로젝트" (회의록 DB에 날짜별 기록: 10/16 주제 선정·제안서 v2, 10/19 교수님 피드백 요약)
- 수업: 성균관대 DSC3033 딥러닝2: 자연어처리 (정미나 교수님, minajung@skku.edu, 메일 제목에 [DSC3033])
- 팀: 6팀 — 백윤우, 임병건, 박채영

## 지금까지의 흐름
1. 10/16 주제 확정: 한국어 공시(DART 사업보고서) 기업·연도 비교 질의응답 + OpenDART 수치 검증.
2. 10/16 v2: 접수일 vs 사업연도 분리, FinDartBench 채택, 수치는 API 조회 + Python 계산.
3. 10/18 교수님 피드백: 주제 적절, 과목 적합성 "보통" → **직접 학습(미세조정) 모델 필수**. 에이전트·데모는 뒤로.
4. 10/19 방향 수정: 대표 실험 = 공시 도메인 검색 임베딩 대조 학습 미세조정.
5. 10/20 레포 구성, 서버 연결.

## 이미 있는 것
- `src/finsight/`: Kiwi 토크나이저, BM25, 검색 지표, 숫자 정규화(`numbers.py`), 공시 메타데이터 파서(`metadata.py`) — 테스트 24개 통과
- `scripts/eval_autorag_bm25.py`: AutoRAGRetrieval BM25 비교 (결과 재현됨: R@1 공백 0.667 / Kiwi 내용어 0.851)
- `docs/feedback-response.md`: 교수님 피드백 항목별 대응·상태표

## 서버
- Ubuntu 22.04, 계정 `urp_jwl` (접속 정보는 레포에 적지 않는다)
- 2026-10-20 접속 시 상태: `/home` 92.2% 사용 (약 60GB 남음, 공유), swap 94%, 동시 접속 3명
- 규칙:
  - 디스크: `/home`이 거의 차 있으므로(22GB) 데이터·캐시는 `/mnt/urp_jwl_shared/bk/` 아래 (`finsight-ko/data/`, `HF_HOME=/mnt/urp_jwl_shared/bk/hf_cache`). 체크포인트는 최고 1–2개만 저장. 큰 작업 전 `df -h ~` 확인.
  - GPU: 쓰기 전 `nvidia-smi`로 빈 GPU 확인 후 `CUDA_VISIBLE_DEVICES` 지정. 다른 사람 프로세스 건드리지 않기.
  - 메모리: 큰 parquet는 필요한 컬럼만, 청크 단위로 읽기.
  - 오래 걸리는 작업은 `tmux` 안에서, 로그는 파일로.
  - 비밀값(`OPENDART_API_KEY`)은 `~/finsight-ko/.env`에만.

## 환경 점검 결과 (서버 세션이 채움)
| 항목 | 결과 |
|---|---|
| GPU | RTX 4090 24GB × 4 (2026-10-20 06:36 모두 유휴), 드라이버 580.173, CUDA 13.0 |
| Python / CUDA / torch | Python 3.10.12, nvcc 12.8, torch 2.11.0+cu128, transformers 5.20.2, sentence-transformers 6.1.0 |
| 디스크 여유 (`df -h ~`) | `/home` 22GB 남음 (98% 사용, 문서의 60GB보다 적음). `du -sh ~` 6.8GB. RAM 125GB, swap 8GB 중 7.5GB 사용 |
| 작업 위치 | 레포·데이터·HF 캐시를 `/mnt/urp_jwl_shared/bk/` (NFS, 4.4TB 여유)에 둠: 레포 `/mnt/urp_jwl_shared/bk/finsight-ko`, `HF_HOME=/mnt/urp_jwl_shared/bk/hf_cache` (`.bashrc`에 추가). `/home`은 쓰지 않음 |
| github.com / huggingface.co / pypi.org / opendart.fss.or.kr 접속 | 200 / 200 / 200 / 302 (리다이렉트, API 정상 응답 확인) |
| `pytest -q` | 24 passed |
| git push 가능 여부 | 불가 (HTTPS 인증 정보 없음, `gh` 미로그인) → `gh auth login` 또는 PAT 필요 |

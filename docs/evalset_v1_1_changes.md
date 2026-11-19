# 평가셋 v1.1 변경 내역 (G2)

`scripts/evalset_v1_1.py` → `data/eval/evalset_v1_1.jsonl` (git-ignored). 분할은 v1 그대로(`data/eval/splits.json`). 검수한 30문항에 3인 합의 판정과 메모를 반영했다(`review.by = team-consensus`). 검색·test 실행 없음.

| qid | 판정 | 반영 |
|---|---|---|
| draft-n0004 | 수정 | gold + ['20240328001188_0058', '20240328001188_0067', '20240328001188_0500'] (restated FY2022 value 9,949,357백만원 in the FY2023 report) |
| draft-n0034 | 수정 | representative evidence 20250317001029_0102 moved first; unit header conflict/missing in evidence chunks: fixed in the corpus by G1 (corpus_a_unitfix_AB, same ids) |
| draft-n0037 | 수정 | representative evidence 20230320000235_0086 moved first; unit header conflict/missing in evidence chunks: fixed in the corpus by G1 (corpus_a_unitfix_AB, same ids) |
| draft-n0066 | 수정 | unit header conflict/missing in evidence chunks: fixed in the corpus by G1 (corpus_a_unitfix_AB, same ids) |
| draft-n0070 | 통과 | 그대로 (통과) |
| draft-n0077 | 통과 | 그대로 (통과) |
| draft-n0203 | 수정 | unit header conflict/missing in evidence chunks: fixed in the corpus by G1 (corpus_a_unitfix_AB, same ids) |
| draft-n0206 | 수정 | representative evidence 20240318000570_0089 moved first; unit header conflict/missing in evidence chunks: fixed in the corpus by G1 (corpus_a_unitfix_AB, same ids) |
| draft-n0335 | 수정 | gold order interleaved by company (first 3 were ['삼성전자']) |
| draft-n0336 | 수정 | gold order interleaved by company (first 3 were ['삼성전자']) |
| draft-n0339 | 수정 | gold order interleaved by company (first 3 were ['삼성전자']) |
| draft-n0347 | 수정 | gold order interleaved by company (first 3 were ['삼성전자']) |
| draft-n0348 | 수정 | gold order interleaved by company (first 3 were ['삼성전자']) |
| draft-n0365 | 수정 | gold order interleaved by company (first 3 were ['현대자동차']); 기아 value 107,448,752백만원 found in ['20250313001390_0084', '20250313001390_0091', '20250313001390_0122', '20250313001390_0123', '20250313001390_0225', '20250313001390_0510'] |
| draft-s004 | 수정 | answer_text rewritten from the gold chunk (was truncated) |
| draft-s042 | 수정 | question: '국민은행의 2023 사업연도 유동성 및 자금조달과 지출 상황에 대한 요약은 무엇입니까?' -> '국민은행의 2023 사업연도 사업보고서에서 2023년 4분기 유동성 지표(LCR·외화 LCR)와 유동성 위기관리 체계는 어떻게 설명되나요?'; answer_text rewritten from the gold chunk (was truncated) |
| draft-u001 | 수정 | question: '다음 분기에 HMM이 확정 배당금을 얼마로 발표할 것으로 예상하십니까?' -> 'HMM의 2024 사업연도 사업보고서(2025년 3월 공시)에 2025년 2분기 확정 배당금(주당)이 나와 있나요?'; absence scan in HMM FY2024 /분기\s*배당|중간\s*배당/: 2 chunk(s) to read |
| draft-u009 | 수정 | question: '기아 주가의 향후 12개월 목표가는 얼마인가요?' -> '기아의 2024 사업연도 사업보고서(2025년 3월 공시)에 회사가 제시한 향후 12개월(2025년 3월 기준) 목표 주가가 있나요?'; absence scan in 기아 FY2024 /목표\s*주가/: 0 chunk(s) to read |
| draft-u010 | 수정 | question: '두나무와 경쟁하는 다른 가상자산 거래소들이 내부적으로 세운 영업 전략은 무엇인가요?' -> '두나무의 2024 사업연도 사업보고서(2025년 3월 공시)에 경쟁 거래소(빗썸·코인원)의 비공개 내부 영업 전략이 나와 있나요?'; absence scan in 두나무 FY2024 /빗썸|코인원/: 0 chunk(s) to read |
| hum-0011 | 수정 | question: '하이닉스 연결 매출이 2024년에 2023년보다 두 배 넘게 늘었다는데, 증가액은 얼마인가?' -> '하이닉스의 2024 사업연도 연결 매출은 2023 사업연도의 2배를 넘었는데, 증가액은 얼마인가?' |
| hum-0014 | 수정 | unit header conflict/missing in evidence chunks: fixed in the corpus by G1 (corpus_a_unitfix_AB, same ids) |
| hum-0030 | 통과 | 그대로 (통과) |
| hum-0032 | 통과 | 그대로 (통과) |
| hum-0033 | 통과 | 그대로 (통과) |
| hum-0036 | 통과 | 그대로 (통과) |
| hum-0042 | 수정 | question: 'KB국민은행 유동성커버리지비율(LCR)은 2023년 4분기와 2024년 4분기에 각각 몇 %였고, 규제비율은 어떻게 바뀌었나?' -> 'KB국민은행의 유동성커버리지비율(LCR, 은행업감독업무시행세칙 기준, 외화 LCR 제외)은 2023년 4분기와 2024년 4분기에 각각 몇 %였고, 금융감독원 규제비율은 어떻게 바뀌었나?' |
| hum-0046 | 통과 | 그대로 (통과) |
| hum-0053 | 수정 | question: '삼성전자 경영진이 사업보고서에서 밝힌 2026년 말 목표 주가는?' -> '삼성전자의 2024 사업연도 사업보고서(2025년 3월 공시)에 경영진이 밝힌 2026년 말 목표 주가가 있나요?'; absence scan in 삼성전자 FY2024 /목표\s*주가/: 0 chunk(s) to read |
| hum-0054 | 수정 | question: '한전은 2027년 1월 전기요금을 kWh당 얼마 올리기로 확정했나?' -> '한국전력공사의 2024 사업연도 사업보고서(2025년 3월 공시)에 2027년 1월 주택용 전기요금의 확정 인상액(원/kWh)이 나와 있나요?'; absence scan in 한국전력공사 FY2024 /2027.{0,40}(요금|인상)|(요금|인상).{0,40}2027/: 0 chunk(s) to read |
| hum-0055 | 수정 | question: 'SK하이닉스 경쟁사 마이크론의 HBM 내부 수율은 몇 %인가?' -> 'SK하이닉스의 2024 사업연도 사업보고서(2025년 3월 공시)에 경쟁사 마이크론의 HBM3E 생산 수율(%)이 나와 있나요?'; absence scan in SK하이닉스 FY2024 /마이크론|Micron/: 0 chunk(s) to read |

## 나머지 167문항 자동 점검 (목록만, 고칠지는 팀이 정함)

- **truncated answer:** 18문항 — draft-s073, draft-s051, draft-s047, draft-s025, draft-s032, draft-s054, draft-s086, draft-s022, draft-s002, draft-s031, draft-s074, draft-s064, draft-s056, draft-s030, draft-s068, draft-s053, draft-s059, draft-s039
- **answerless question without report/year:** 13문항 — draft-u008, draft-u012, draft-u015, draft-u003, draft-u011, draft-u007, draft-u006, draft-u005, hum-0056, hum-0057, hum-0058, hum-0059, hum-0060
- **company compare: first 3 gold chunks from one company:** 36문항 — draft-n0345, draft-n0346, draft-n0360, draft-n0357, draft-n0368, draft-n0362, draft-n0367, draft-n0341, draft-n0356, draft-n0338, draft-n0364, draft-n0363, draft-n0340, draft-n0351, draft-n0366, draft-n0337, draft-n0352, draft-n0353, draft-n0361, draft-n0334, draft-n0358, draft-n0344, draft-n0343, draft-n0369, draft-n0355, draft-n0342, hum-0017, hum-0018, hum-0019, hum-0020, hum-0021, hum-0022, hum-0023, hum-0024, hum-0025, hum-0026
- **numeric gold chunk with wrong/missing unit (G1 A/B):** 105문항 — draft-n0019, draft-n0117, draft-n0101, draft-n0055, draft-n0076, draft-n0138, draft-n0151, draft-n0143, draft-n0061, draft-n0031, draft-n0128, draft-n0093, draft-n0014, draft-n0160, draft-n0053, draft-n0131, draft-n0039, draft-n0045, draft-n0001, draft-n0018, draft-n0149, draft-n0116, draft-n0103, draft-n0169, draft-n0112, draft-n0321, draft-n0281, draft-n0234, draft-n0318, draft-n0280, draft-n0274, draft-n0181, draft-n0214, draft-n0320, draft-n0208, draft-n0207, draft-n0317, draft-n0253, draft-n0284, draft-n0241, draft-n0189, draft-n0192, draft-n0293, draft-n0242, draft-n0279, draft-n0258, draft-n0264, draft-n0171, draft-n0271, draft-n0312, draft-n0277, draft-n0303, draft-n0302, draft-n0188, draft-n0300, draft-n0345, draft-n0346, draft-n0360, draft-n0357, draft-n0368, draft-n0362, draft-n0367, draft-n0341, draft-n0356, draft-n0338, draft-n0364, draft-n0363, draft-n0340, draft-n0351, draft-n0366, draft-n0337, draft-n0352, draft-n0353, draft-n0361, draft-n0334, draft-n0358, draft-n0344, draft-n0343, draft-n0369, draft-n0355, draft-n0342, hum-0001, hum-0002, hum-0003, hum-0004, hum-0005, hum-0006, hum-0007, hum-0008, hum-0009, hum-0010, hum-0012, hum-0013, hum-0015, hum-0016, hum-0017, hum-0018, hum-0019, hum-0020, hum-0021, hum-0022, hum-0023, hum-0024, hum-0025, hum-0026

점검 대상 167문항. 단위 항목은 G1 코퍼스 수정본에서 이미 고쳐진다(같은 청크 ID).

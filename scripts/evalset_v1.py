"""In-house evaluation set v1 = reviewed LLM draft (140) + ~60 "human" items, and the dev/test split.

PROXY NOTICE (2026-11-18): the team asked Claude to stand in for the human steps for now (they are busy with a paper).
  * review of the 140 draft items: done by Claude following docs/eval_guideline.md section 5 (pass / revise / discard),
    review.by = "CL-proxy". Not a human review; no kappa (needs two independent raters).
  * the ~60 "human-written" items: written by Claude following guideline section 4 (question first, then evidence),
    generated_by = "human-proxy:Claude". The team will replace them with their own items later; rerun this script then.
Everything downstream (scripts/eval_evalset.py --split ...) reports these items separately from the LLM/template ones.

Numeric answers come from data/eval/numeric_auto.jsonl (OpenDART values, restatement alternatives); company-compare
items not in the template file are built from two single-value entries. Gold chunks for numeric items: same rule as
scripts/evalset_draft.py (chunks of that company-year whose text contains the reported figure). Narrative gold chunks
were looked up by hand in corpus_a (ids below).

Split (guideline 7): dev 50 / test rest, about 1:3 per type and per origin, items sharing a gold chunk kept on the
same side, seed 42, written once to data/eval/splits.json.

Usage: python scripts/evalset_v1.py
Outputs: data/eval/evalset_v1.jsonl, data/eval/splits.json, outputs/midterm/evalset_v1_stats.json
"""
from __future__ import annotations

import json
import random
from collections import Counter, defaultdict
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
EVAL = ROOT / "data" / "eval"
OUT = ROOT / "outputs" / "midterm"
SEED = 42
N_DEV = 50
BY = "CL-proxy"

# ---------------------------------------------------------------- review of the 140-item draft (guideline 5)
REVISE_Q = {  # qid -> (new question, reason)
    "draft-n0281": ("삼성전자의 연결 기준 당기순이익은 2023 사업연도 대비 2024 사업연도에 얼마나 변했습니까? (증감액과 증감률)",
                    "연도 순서가 뒤집힘(2024 대비 2023)"),
    "draft-n0280": ("삼성전자의 연결 기준 당기순이익은 2022 사업연도 대비 2023 사업연도에 얼마나 변했습니까? (증감액과 증감률)",
                    "연도 순서가 뒤집혔고 실제로는 감소인데 '증가했습니까'로 물음"),
    "draft-n0171": ("HMM의 연결 기준 당기순이익은 2023 사업연도 대비 2024 사업연도에 얼마나 변했습니까? (증감액과 증감률)",
                    "연도 순서가 뒤집힘"),
    "draft-n0302": ("한국전력공사의 연결 기준 매출액은 2023 사업연도 대비 2024 사업연도에 얼마나 변했습니까? (증감액과 증감률)",
                    "연도 순서가 뒤집힘"),
    "draft-n0234": ("2024 사업연도 SK하이닉스의 연결 기준 영업이익은 2023 사업연도와 비교해 얼마나 달라졌나요? (증감액)",
                    "2023년이 영업손실이라 증감률이 정의되지 않음 → 증감액만 묻기"),
    "draft-n0300": ("2024 사업연도 한국전력공사의 연결 기준 당기순이익(손실)은 2022 사업연도와 비교해 얼마나 달라졌나요? (증감액)",
                    "2022년이 손실이라 증감률이 정의되지 않음 → 증감액만 묻기"),
    "draft-s040": ("국민은행은 2023 사업연도에 유동성리스크 위기상황분석을 어떤 분석기간으로 하며, 부족분이 생기면 어떻게 대응한다고 했나요?",
                   "운영리스크 부분은 정답 청크에 없음 → 유동성리스크로 좁힘"),
    "draft-s051": ("기아의 2023 사업연도 사업보고서에서 승용·RV·상용별 주요 차종은 무엇인가요?",
                   "정답 청크에 '수익성 영향'이 없음 → 청크가 답하는 범위로 좁힘"),
    "draft-s025": ("2024 사업연도 SK텔레콤 사업보고서에서 SK브로드밴드의 매출이 늘어난 이유는 무엇인가요?",
                   "질문(무선 이익 전략)과 청크(한 문장 + SK브로드밴드)가 어긋남"),
    "draft-s032": ("SK하이닉스는 2023 사업연도 사업보고서에서 메모리 수요의 계절성이 어떻게 변하고 있다고 설명했나요?",
                   "'사업·위험·전략·시장 요약'은 청크 하나로 답할 수 없음"),
    "draft-s054": ("기아의 2024 사업연도 사업보고서에서 사업 부문 구성과 주요 차종은 어떻게 설명되나요?",
                   "포괄 요약 질문 → 청크 범위로 좁힘"),
    "draft-s022": ("SK텔레콤의 2023 사업연도 사업보고서에서 에이닷(A.)과 이프랜드 서비스의 성과는 어떻게 소개되었나요?",
                   "포괄 질문 → 청크 범위로 좁힘"),
    "draft-s056": ("두나무가 2022 사업연도 사업보고서에서 소개한 주요 서비스는 무엇인가요?",
                   "포괄 요약 질문 → 청크 범위로 좁힘"),
    "draft-s068": ("삼성전자 2023 사업연도 사업보고서에서 디스플레이(OLED) 시장 상황과 회사의 대응은 어떻게 설명되나요?",
                   "하만 부분은 정답 청크에 없음"),
    "draft-s039": ("국민은행 2022 사업연도 사업보고서에 나오는 알뜰폰(Liiv M) 서비스에는 어떤 요금제와 부가서비스가 있나요?",
                   "정답 청크가 Liiv M 요금제 목록이라 '주요 사업·위험' 질문과 무관 → 청크에 맞게 다시 씀"),
    # unanswerable: rewrite the broken / awkward LLM phrasings
    "draft-u010": ("두나무와 경쟁하는 다른 가상자산 거래소들이 내부적으로 세운 영업 전략은 무엇인가요?", "LLM 머리말이 질문에 섞임"),
    "draft-u004": ("LG전자의 2026 사업연도 연간 영업이익은 최종적으로 얼마였나요?", "어색한 문장"),
    "draft-u009": ("기아 주가의 향후 12개월 목표가는 얼마인가요?", "어색한 문장"),
    "draft-u006": ("SK하이닉스 대표이사의 자택 주소와 개인 연락처는 무엇인가요?",
                   "임원 소유주식은 공시되므로 '개인 자산'은 일부 답이 있을 수 있음 → 명백히 없는 정보로"),
    "draft-u007": ("SK하이닉스 주가는 앞으로 1년 안에 얼마까지 오를까요?", "어색한 문장"),
}
REVISE_NOTE = {
    "draft-s030": "answer_text의 '$816억(-12.1%)'은 낸드가 아니라 앞 문단(D램) 수치 → 답 문장 수정 필요 (검색 평가에는 영향 없음)",
}
DISCARD = {
    "draft-s075": "정답 청크가 회계기준 개정 문단 — 질문(사업·위험·전략)에 답하지 않음",
    "draft-s021": "정답 청크가 비금융자산 손상 회계정책 — 질문에 답하지 않음",
    "draft-s080": "정답 청크가 회계기준 개정 문단 — 질문에 답하지 않음",
}

# ---------------------------------------------------------------- proxy "human" items (guideline 4)
# numeric: (type, question, numeric_auto qid(s), notes)
HUM_NUM = [
    (1, "HMM 2024 사업연도 연결 영업이익은 얼마 났어?", ["num-0016"], ""),
    (1, "하이닉스는 2023년에 연결 기준으로 영업적자를 얼마나 냈나?", ["num-0064"], "별칭(하이닉스), '영업적자' 표현"),
    (1, "한전 2022년 연결 영업손실은 얼마였지?", ["num-0137"], "별칭(한전)"),
    (1, "KB국민은행의 2024 사업연도 연결 영업이익은?", ["num-0080"], "별칭(KB국민은행)"),
    (1, "업비트 운영사 두나무의 2023 사업연도 연결 영업이익은 얼마인가?", ["num-0108"], ""),
    (1, "기아차 2023년 연결 매출액 알려줘", ["num-0090"], "별칭(기아차)"),
    (1, "SKT의 2024 사업연도 연결 매출(영업수익)은 얼마였어?", ["num-0050"], "별칭(SKT), 매출=영업수익"),
    (1, "LG전자 제22기 연결 매출액은 얼마인가?", ["num-0026"], "기수 표기(제22기=2023 사업연도)"),
    (2, "현대차는 2023년에 2022년보다 돈을 얼마나 더 벌었어? 연결 당기순이익 기준으로.", ["num-0316"], "구어체"),
    (2, "삼성전자 연결 영업이익은 2023년 바닥에서 2024년에 얼마나 회복됐나?", ["num-0290"], "구어체"),
    (2, "하이닉스 연결 매출이 2024년에 2023년보다 두 배 넘게 늘었다는데, 증가액은 얼마인가?", ["num-0228"], "별칭"),
    (2, "HMM의 2024 사업연도 연결 매출은 2023 사업연도보다 몇 % 늘었나?", ["num-0174"], ""),
    (2, "업비트 운영사의 연결 부채총계는 2023 사업연도에서 2024 사업연도 사이에 얼마나 불었나?", ["num-0269"],
     "회사명 대신 브랜드(업비트) — 회사 추출 시험"),
    (2, "KB국민은행 연결 영업이익은 2024 사업연도에 2023 사업연도 대비 얼마나 늘었나?", ["num-0244"], ""),
    (2, "현대차 연결 영업이익, 2024년에는 2023년보다 줄었나? 얼마나?", ["num-0326"], "구어체"),
    (2, "한전의 연결 자본총계는 2023 사업연도 말에서 2024 사업연도 말까지 얼마나 회복됐나?", ["num-0311"], "별칭(한전)"),
    (3, "2024 사업연도에 영업이익을 더 많이 낸 곳은 삼성전자와 SK하이닉스 중 어디이고, 얼마나 차이 나나? (연결)",
     ["num-0349"], "가이드라인 4절 예시 1"),
    (3, "삼성이랑 하이닉스, 2024년 말 연결 자본총계 차이는 얼마야?", ["num-0350"], "별칭"),
    (3, "2022 사업연도 말 연결 부채총계는 현대차와 기아 중 어디가 얼마나 더 많았나?", ["num-0354"], ""),
    (3, "현대차와 기아 중 2023 사업연도 연결 매출이 더 큰 곳과 그 차이는?", ["num-0359"], ""),
    (3, "2024 사업연도 연결 매출은 삼성전자와 LG전자 중 어디가 얼마나 더 컸나?", ["num-0129", "num-0032"], ""),
    (3, "2023년 삼성전자와 LG전자의 연결 영업이익 차이는 얼마였나?", ["num-0125", "num-0028"], ""),
    (3, "2022 사업연도 말 연결 부채총계, 삼성전자와 LG전자는 얼마나 차이 났나?", ["num-0118", "num-0021"], ""),
    (3, "2024 사업연도 연결 영업이익 기준으로 삼성전자가 LG전자보다 얼마나 더 벌었나?", ["num-0131", "num-0034"], ""),
    (3, "삼성전자와 LG전자의 2023 사업연도 말 연결 자산총계 차이는?", ["num-0127", "num-0030"], ""),
    (3, "2022 사업연도 연결 당기순이익은 삼성전자와 LG전자 중 어디가 얼마나 많았나?", ["num-0116", "num-0019"], ""),
]
# narrative: (type, question, corp list, fiscal years, gold chunk ids, short answer, notes)
HUM_TXT = [
    (4, "하이닉스는 2024년 사업보고서에서 HBM 시장 리더십을 어떻게 이어가겠다고 했나?", ["SK하이닉스"], [2024],
     ["20250319000665_0071", "20250319000665_0624"], "대역폭·적층 단수 상향 등 1등 특성/품질과 Time to Market, 수요 확대 시기에 맞춘 적기 공급", ""),
    (4, "한전이 2022년에 매출이 늘었는데도 큰 적자를 낸 이유를 사업보고서에서 뭐라고 설명했나?", ["한국전력공사"], [2022],
     ["20230525000309_1849", "20230525000309_1854", "20230525000309_1856", "20230525000309_1860"],
     "국제 연료가격 폭등(러-우 전쟁, 환율)으로 연료비·구입전력비 급증", ""),
    (4, "HMM 2024 사업보고서에서 컨테이너 운임을 끌어올린 요인으로 꼽은 것은?", ["HMM"], [2024],
     ["20250318000828_0045", "20250318000828_0498", "20250318000828_0499"],
     "홍해 이슈, 미 동안·캐나다 서안 항만노조 파업 우려, 관세 앞둔 중국발 선적 앞당기기", ""),
    (4, "KB국민은행은 2024년 홍콩 H지수 ELS 사태가 실적에 어떤 영향을 줬다고 설명했나?", ["국민은행"], [2024],
     ["20250314001552_1154", "20250314001552_1155", "20250314001552_0430", "20250314001552_0872"],
     "ELS 고객손실 보상에도 순이익 3조 2,518억원, ELS 판매 중단으로 신탁손익·수수료이익 감소, 자율조정 보상금 충당부채", ""),
    (4, "SKT가 말하는 'AI 피라미드' 전략은 어떤 내용인가? (2024 사업연도 보고서)", ["SK텔레콤"], [2024],
     ["20250317000684_0085", "20250317000684_0086", "20250317000684_0760"],
     "AI인프라·AIX·AI서비스 3대 영역, 자강과 협력으로 글로벌 AI 컴퍼니 도약", ""),
    (4, "기아의 2024 사업보고서에서 PBV 사업은 어떻게 준비되고 있다고 했나?", ["기아"], [2024],
     ["20250313001390_0045", "20250313001390_0516"], "2025년 첫 PBV(PV5) 출시, 수요처 사전 확보, 유즈 케이스별 맞춤형 차량", ""),
    (4, "삼성전자는 2024 사업연도 보고서에서 차세대 HBM 제품 계획을 어떻게 밝혔나?", ["삼성전자"], [2024],
     ["20250311001085_0626", "20250311001085_0075"], "HBM3E 개선제품 판매 확대, 하반기 HBM4 개발·양산", ""),
    (4, "한전은 2024년에 전기요금을 조정해서 매출이 어떻게 변했다고 했나?", ["한국전력공사"], [2024],
     ["20250318000747_2276", "20250318000747_2271"], "판매단가 6.6% 상승, 매출 5.9% 증가한 93조 3,989억원", ""),
    (4, "두나무는 2023 사업보고서에서 어떤 법률들의 적용을 받는다고 밝혔나?", ["두나무"], [2023],
     ["20240328001689_0042"], "특정금융정보법상 가상자산사업자, 전자상거래소비자보호법 등", ""),
    (4, "LG전자 2024 사업보고서에서 TV 수요는 2023년 부진 이후 어떻게 됐다고 했나?", ["LG전자"], [2024],
     ["20250317001029_0086", "20250317001029_0087", "20250317001029_0088"], "2024년 점진적 수요 회복으로 성장세 전환(스포츠 이벤트 영향)", ""),
    (4, "2022년 삼성전자 메모리 평균 판매가격은 전년보다 얼마나 떨어졌나?", ["삼성전자"], [2022],
     ["20230307000542_0038"], "약 17% 하락", "본문 서술 속 수치 (OpenDART 계정 아님)"),
    (4, "하이닉스는 2023년에 왜 7조원 넘는 영업적자를 냈다고 설명했나?", ["SK하이닉스"], [2023],
     ["20240319000684_0064", "20240319000684_0065", "20240319000684_0663", "20240319000684_0665"],
     "메모리 가격 급락과 재고평가손실 (상반기 수요 약세)", ""),
    (2, "HMM은 2023년 말과 2024년의 컨테이너 운임 상승 원인을 각각 무엇으로 설명했나?", ["HMM"], [2023, 2024],
     ["20240328001188_0047", "20240328001188_0048", "20240328001188_0495", "20250318000828_0045",
      "20250318000828_0498", "20250318000828_0499"],
     "2023: 파나마 운하 가뭄, 홍해 사태 / 2024: 홍해, 항만노조 파업 우려, 중국발 선적 앞당기기", "가이드라인 4절 예시 2"),
    (2, "한전 영업손익이 2022 사업연도보다 2023 사업연도에 크게 나아진 이유는?", ["한국전력공사"], [2022, 2023],
     ["20240329002856_2361", "20240329002856_2362"], "요금조정으로 판매단가 상승, 연료가격 안정으로 연료비 감소", ""),
    (2, "하이닉스는 2022 사업보고서에서 2023년 메모리 시장을 어떻게 전망했고, 2023 사업보고서에서는 실제 업황을 어떻게 설명했나?",
     ["SK하이닉스"], [2022, 2023], ["20230321001209_0057", "20240319000684_0064"],
     "전망: Downturn 지속 / 실제: 상반기 약세, 하반기 AI 메모리 수요와 감산으로 개선", ""),
    (2, "KB국민은행 유동성커버리지비율(LCR)은 2023년 4분기와 2024년 4분기에 각각 몇 %였고, 규제비율은 어떻게 바뀌었나?",
     ["국민은행"], [2023, 2024], ["20240814004195_1169", "20250314001552_1163", "20250314001552_0120"],
     "102.90% → 102.50%, 규제비율 95.0% → 97.5%", ""),
    (2, "기아 RV 주요 차종 목록은 2023 사업보고서와 2024 사업보고서 사이에 어떻게 바뀌었나?", ["기아"], [2023, 2024],
     ["20240312000909_0030", "20250313001390_0030"], "모하비 빠지고 EV3 추가", ""),
    (2, "삼성전자 종속기업 수는 2022 사업연도와 2024 사업연도에 각각 몇 개였나?", ["삼성전자"], [2022, 2024],
     ["20230307000542_0035", "20250311001085_0033"], "232개 → 228개", ""),
    (2, "SK브로드밴드 매출은 2023년과 2024년에 각각 얼마였나? (SK텔레콤 사업보고서)", ["SK텔레콤"], [2023, 2024],
     ["20240318000570_0039", "20250317000684_0041"], "2024년 4조 4,089억원 (전년 4조 2,747억원, +3.1%)", ""),
    (2, "두나무 주요 서비스 목록에서 2022 사업연도에는 있었는데 2024 사업연도에는 빠진 서비스는?", ["두나무"], [2022, 2024],
     ["20230331003960_0015", "20250328000589_0022"], "세컨블록", ""),
    (3, "2024 사업보고서 기준으로 삼성전자와 SK하이닉스는 각각 HBM 관련해 어떤 계획을 밝혔나?", ["삼성전자", "SK하이닉스"],
     [2024], ["20250311001085_0626", "20250319000665_0071", "20250319000665_0624"],
     "삼성: HBM3E 개선제품 확대·HBM4 양산 / 하이닉스: HBM 시장 리더십 유지", ""),
    (3, "삼성전자와 LG전자는 2024 사업연도 TV 시장 수요를 각각 어떻게 설명했나?", ["삼성전자", "LG전자"], [2024],
     ["20250311001085_0069", "20250317001029_0087", "20250317001029_0088"], "삼성: 소비심리 악화로 수요 감소 추이 / LG: 2024 점진적 회복", ""),
    (3, "2022 사업보고서에서 삼성전자와 SK하이닉스는 메모리 업황을 각각 어떻게 봤나?", ["삼성전자", "SK하이닉스"], [2022],
     ["20230307000542_0076", "20230321001209_0057", "20230321001209_0437"], "둘 다 수요 약세·Downturn 지속 전망", ""),
    (3, "현대차와 기아는 2024년 전기차·하이브리드 판매에 대해 각각 뭐라고 했나?", ["현대자동차", "기아"], [2024],
     ["20250312001148_1047", "20250313001390_0074", "20250313001390_0044"],
     "현대: EV·HEV 판매 확대로 믹스 개선 / 기아: 친환경차 비중 39.9%, EV3 런칭", ""),
    (3, "2023 사업보고서에서 삼성전자와 SK하이닉스는 메모리 시황이 언제부터 나아졌다고 했나?", ["삼성전자", "SK하이닉스"], [2023],
     ["20240312000736_0075", "20240319000684_0064"], "하반기 (재고 정상화, AI 수요, 감산)", ""),
    (3, "현대차와 기아의 2024 사업보고서상 사업(보고) 부문 구성은 어떻게 다른가?", ["현대자동차", "기아"], [2024],
     ["20250312001148_0003", "20250312001148_0050", "20250313001390_0030"],
     "현대: 차량·금융·기타 부문 / 기아: 하나의 보고 부문", ""),
]
HUM_UNANS = [
    ("삼성전자 경영진이 사업보고서에서 밝힌 2026년 말 목표 주가는?", "삼성전자", "가이드라인 4절 예시 3"),
    ("한전은 2027년 1월 전기요금을 kWh당 얼마 올리기로 확정했나?", "한국전력공사", ""),
    ("SK하이닉스 경쟁사 마이크론의 HBM 내부 수율은 몇 %인가?", "SK하이닉스", ""),
    ("업비트에서 거래를 가장 많이 한 개인 고객 상위 10명은 누구인가?", "두나무", ""),
    ("현대자동차 회장 개인 소유 부동산 목록을 알려줘", "현대자동차", ""),
    ("HMM이 2027년에 새로 발주할 컨테이너선 척수 확정치는?", "HMM", ""),
    ("기아 노동조합 조합원 개개인의 임금 명세는?", "기아", ""),
    ("국민은행 특정 지점의 2024년 일별 현금 시재액은?", "국민은행", ""),
]


def main() -> None:
    corpus = pd.read_json(ROOT / "data/processed/corpus_a.jsonl", lines=True, dtype={"rcept_no": str, "chunk_id": str})
    by_cy = {k: g for k, g in corpus.groupby(["corp_name", "fiscal_year"])}
    known = set(corpus.chunk_id)
    num = {q["qid"]: q for q in map(json.loads, open(EVAL / "numeric_auto.jsonl", encoding="utf-8"))}

    def figure_strings(v: float) -> list[str]:  # same rule as evalset_draft.py
        out = []
        for div in (1e6, 1e3, 1, 1e8):
            x = abs(v) / div
            if x >= 1 and abs(x - round(x)) < 1e-6:
                out.append(f"{int(round(x)):,}")
        return out

    def find_gold(corp: str, fy: int, value: float) -> list[str]:
        g = by_cy.get((corp, fy))
        if g is None:
            return []
        figs = figure_strings(value)
        hit = g[g.text.apply(lambda t: any(f in t for f in figs))]
        core = hit[hit.section_path.str.contains("요약재무정보|재무제표|경영진단|손익|재무상태")]
        return (core if len(core) else hit).chunk_id.tolist()[:8]

    items = []
    # 1) reviewed draft
    for x in map(json.loads, open(EVAL / "evalset_draft.jsonl", encoding="utf-8")):
        q = x["qid"]
        hist = []
        status = "pass"
        if q in DISCARD:
            status, hist = "discarded", [DISCARD[q]]
        elif q in REVISE_Q:
            new, why = REVISE_Q[q]
            hist = [f"question: '{x['question']}' -> '{new}' ({why})"]
            x["question"], status = new, "revise"
        elif q in REVISE_NOTE:
            status, hist = "revise", [REVISE_NOTE[q]]
        x["review"] = {"status": status, "by": BY, "history": hist}
        x["origin"] = "llm"
        items.append(x)
    # 2) proxy human items
    k = 0
    for t, qtext, qids, note in HUM_NUM:
        k += 1
        if len(qids) == 1:
            s = num[qids[0]]
            assert s["type"] == t, (qids, t)
            src, val, alt, extra = s["source"], s["answer_value"], s.get("answer_alt"), s["answer_extra"]
        else:  # company compare from two single values
            a, b = num[qids[0]], num[qids[1]]
            va, vb = a["answer_value"], b["answer_value"]
            src, val, alt = a["source"] + b["source"], abs(va - vb), None
            extra = {"value_a": va, "value_b": vb, "larger": (a if va >= vb else b)["source"][0]["corp"],
                     "signed_diff_a_minus_b": va - vb}
        if t == 1:
            gold = find_gold(src[0]["corp"], src[0]["fiscal_year"], val)
        elif t == 2:
            later = max(src, key=lambda d: d["fiscal_year"])
            gold = find_gold(later["corp"], later["fiscal_year"], extra["value_to"])
        else:
            gold = find_gold(src[0]["corp"], src[0]["fiscal_year"], extra["value_a"]) + \
                   find_gold(src[1]["corp"], src[1]["fiscal_year"], extra["value_b"])
        assert gold, qtext
        items.append({"qid": f"hum-{k:04d}", "type": t, "question": qtext,
                      "corp": sorted({d["corp"] for d in src}), "fiscal_year": sorted({d["fiscal_year"] for d in src}),
                      "answer_value": val, "answer_alt": alt, "answer_unit": "원", "answer_extra": extra,
                      "gold_chunk_ids": gold, "source": src, "from_template": qids, "generated_by": "human-proxy:Claude",
                      "review": {"status": "pass", "by": BY, "history": []}, "origin": "human_proxy", "notes": note})
    for t, qtext, corps, years, gold, ans, note in HUM_TXT:
        k += 1
        bad = [g for g in gold if g not in known]
        assert not bad, bad
        items.append({"qid": f"hum-{k:04d}", "type": t, "question": qtext, "corp": corps, "fiscal_year": years,
                      "answer_text": ans, "gold_chunk_ids": gold, "generated_by": "human-proxy:Claude",
                      "review": {"status": "pass", "by": BY, "history": []}, "origin": "human_proxy", "notes": note})
    for qtext, corp, note in HUM_UNANS:
        k += 1
        items.append({"qid": f"hum-{k:04d}", "type": 5, "question": qtext, "corp": [corp], "fiscal_year": [],
                      "answer_text": "보고서에서 확인할 수 없음", "gold_chunk_ids": [], "generated_by": "human-proxy:Claude",
                      "review": {"status": "pass", "by": BY, "history": []}, "origin": "human_proxy", "notes": note})

    live = [x for x in items if x["review"]["status"] != "discarded"]
    # 3) split: union items that share a gold chunk; stratify by (type, origin)
    parent = list(range(len(live)))

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i
    owner = {}
    for i, x in enumerate(live):
        for g in x["gold_chunk_ids"]:
            if g in owner:
                parent[find(i)] = find(owner[g])
            else:
                owner[g] = i
    groups = defaultdict(list)
    for i in range(len(live)):
        groups[find(i)].append(i)
    strata = Counter((x["type"], x["origin"]) for x in live)
    target = {s: n * N_DEV / len(live) for s, n in strata.items()}
    rng = random.Random(SEED)
    glist = list(groups.values())
    best = None
    for _ in range(2000):  # seeded random greedy fills; keep the one closest to N_DEV and to the per-stratum targets
        rng.shuffle(glist)
        have, dev = Counter(), set()
        for grp in glist:
            add = Counter((live[i]["type"], live[i]["origin"]) for i in grp)
            if len(dev) + len(grp) <= N_DEV + 2 and all(have[s] + n <= target[s] + 1.5 for s, n in add.items()):
                dev |= set(grp)
                have.update(add)
        score = (abs(len(dev) - N_DEV), sum(abs(have[s] - target[s]) for s in strata))
        if best is None or score < best[0]:
            best = (score, set(dev))
    dev = best[1]
    for i, x in enumerate(live):
        x["split"] = "dev" if i in dev else "test"
    for x in items:
        if x["review"]["status"] == "discarded":
            x["split"] = None

    with open(EVAL / "evalset_v1.jsonl", "w", encoding="utf-8") as f:
        for x in items:
            f.write(json.dumps(x, ensure_ascii=False) + "\n")
    splits = {"seed": SEED, "rule": "gold-chunk groups, stratified by (type, origin), dev ~= 50",
              "dev": sorted(x["qid"] for x in live if x["split"] == "dev"),
              "test": sorted(x["qid"] for x in live if x["split"] == "test")}
    (EVAL / "splits.json").write_text(json.dumps(splits, ensure_ascii=False, indent=1))
    stats = {
        "n_total_written": len(items), "n_live": len(live),
        "review_status": dict(Counter(x["review"]["status"] for x in items if x["origin"] == "llm")),
        "by_type_origin_split": {f"{t}|{o}|{s}": n for (t, o, s), n in
                                 sorted(Counter((x["type"], x["origin"], x["split"]) for x in live).items())},
        "by_split": dict(Counter(x["split"] for x in live)),
        "by_type": dict(sorted(Counter(x["type"] for x in live).items())),
        "proxy_notice": "review and human-written items are Claude proxies (team busy); replace and rerun",
    }
    (OUT / "evalset_v1_stats.json").write_text(json.dumps(stats, ensure_ascii=False, indent=1))
    print(json.dumps(stats, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()

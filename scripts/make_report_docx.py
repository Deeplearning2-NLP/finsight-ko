"""Build the midterm report as Word (.docx). Numbers are copied from docs/midterm-report.md (sections 0-16).
Usage: /mnt/urp_jwl_shared/bk/venv_docs/bin/python scripts/make_report_docx.py OUT.docx
"""
import sys

from docx import Document
from docx.enum.table import WD_TABLE_ALIGNMENT
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Cm, Pt, RGBColor

FONT = "맑은 고딕"
doc = Document()
sec = doc.sections[0]
sec.page_width, sec.page_height = Cm(21.0), Cm(29.7)
for side in ("left_margin", "right_margin"):
    setattr(sec, side, Cm(2.2))
sec.top_margin = sec.bottom_margin = Cm(2.0)


def set_font(style_or_run, size=None, bold=None, color=None):
    f = style_or_run.font
    f.name = FONT
    rpr = style_or_run.element.get_or_add_rPr() if hasattr(style_or_run, "element") and hasattr(style_or_run.element, "get_or_add_rPr") else style_or_run._element.get_or_add_rPr()
    fonts = rpr.find(qn("w:rFonts"))
    if fonts is None:
        fonts = OxmlElement("w:rFonts")
        rpr.append(fonts)
    for k in ("w:ascii", "w:hAnsi", "w:eastAsia", "w:cs"):
        fonts.set(qn(k), FONT)
    if size:
        f.size = Pt(size)
    if bold is not None:
        f.bold = bold
    if color:
        f.color.rgb = RGBColor.from_string(color)


for name, size in (("Normal", 10), ("Heading 1", 15), ("Heading 2", 12), ("Heading 3", 10.5), ("Title", 20)):
    st = doc.styles[name]
    set_font(st, size, bold=name != "Normal", color=None if name == "Normal" else "1F3864")
    pf = st.paragraph_format
    pf.space_after = Pt(4 if name == "Normal" else 6)
    pf.space_before = Pt(0 if name == "Normal" else 12)
    if name != "Normal":
        pf.keep_with_next = True
doc.styles["Normal"].paragraph_format.line_spacing = 1.25
for name in ("List Bullet", "List Number"):
    set_font(doc.styles[name], 10)


def H(text, lvl=1):
    doc.add_heading(text, level=lvl)


def P(text, bold_prefix=None, italic=False, size=None):
    p = doc.add_paragraph()
    if bold_prefix:
        r = p.add_run(bold_prefix)
        r.bold = True
    r = p.add_run(text)
    r.italic = italic
    if size:
        r.font.size = Pt(size)
    return p


def B(text, bold_prefix=None, level=0):
    p = doc.add_paragraph(style="List Bullet")
    if level:
        p.paragraph_format.left_indent = Cm(0.63 * (level + 1))
    if bold_prefix:
        r = p.add_run(bold_prefix)
        r.bold = True
    p.add_run(text)
    return p


def shade(cell, hex_):
    tcPr = cell._element.get_or_add_tcPr()
    sh = OxmlElement("w:shd")
    sh.set(qn("w:val"), "clear")
    sh.set(qn("w:color"), "auto")
    sh.set(qn("w:fill"), hex_)
    tcPr.append(sh)


def T(header, rows, caption=None, widths=None, bold_last=False):
    if caption:
        c = doc.add_paragraph()
        r = c.add_run(caption)
        r.bold = True
        r.font.size = Pt(9.5)
        c.paragraph_format.keep_with_next = True
    t = doc.add_table(rows=1, cols=len(header))
    t.style = "Table Grid"
    t.alignment = WD_TABLE_ALIGNMENT.CENTER
    for i, h in enumerate(header):
        cell = t.rows[0].cells[i]
        cell.text = ""
        run = cell.paragraphs[0].add_run(h)
        run.bold = True
        run.font.size = Pt(9)
        shade(cell, "D9E2F3")
    for ri, row in enumerate(rows):
        cells = t.add_row().cells
        for i, v in enumerate(row):
            cells[i].text = ""
            txt = str(v)
            bold = txt.startswith("**") and txt.endswith("**")
            run = cells[i].paragraphs[0].add_run(txt.strip("*"))
            run.font.size = Pt(9)
            run.bold = bold or (bold_last and ri == len(rows) - 1)
            if i > 0:
                cells[i].paragraphs[0].alignment = WD_ALIGN_PARAGRAPH.CENTER
    if widths:
        for row in t.rows:
            for i, w in enumerate(widths):
                row.cells[i].width = Cm(w)
    doc.add_paragraph().paragraph_format.space_after = Pt(2)
    return t


# ------------------------------------------------------------------ title
t = doc.add_paragraph(style="Title")
t.add_run("FinSight-ko 중간 보고서")
P("DART 사업보고서 검색 임베딩 미세조정과 OpenDART 수치 검증 질의응답", size=12)
P("성균관대학교 DSC3033 딥러닝2: 자연어처리 · 6팀 (백윤우, 임병건, 박채영) · 2026년 11월", size=9.5)
P("모든 수치는 학교 서버에서 실제로 실행한 결과다. 본 문서는 연구 목적이며 투자 권유가 아니다.", italic=True, size=9)

# ------------------------------------------------------------------ 0
H("요약")
P("한국어 사업보고서(DART)를 회사·사업연도별로 찾아 주고, 재무 수치는 OpenDART 공식 값으로 검증해 답하는 "
  "질의응답 시스템을 만들고 있다. 중간 단계까지의 핵심 결과는 다음과 같다.")
T(["항목", "결과"], [
    ["직접 학습 1: 검색 임베딩", "KURE-v1을 학습 쌍 4,646개(평가와 겹치지 않는 20개사)로 대조 학습. 자체 평가셋 test에서 공개 KURE 대비 R@5 +18.7%p [+11.2, +26.1] (같은 파이프라인)"],
    ["직접 학습 2: 재정렬기", "cross-encoder 3종(KLUE-RoBERTa, KLUE-BERT, bge-reranker) 학습. bge-reranker 미세조정이 최종, KLUE-BERT 재정렬기도 재정렬 없음 대비 R@5 +30.6%p"],
    ["최종 검색 시스템 (v2)", "형태소 BM25 + 미세조정 임베딩 RRF, 회사·연도 필터, 두 회사 질문 분해, 재정렬. 자체 평가셋 test R@5 0.888, FinDartBench test R@1 0.592 / R@5 0.818"],
    ["수치 검증", "EXAONE-3.5-7.8B 기준 수치 정답률: LLM 단독 0.0% → RAG 4.2% → RAG + OpenDART 95.8%. 근거 없는 금액 비율 40.4% → 0%"],
    ["오류 분석", "30건 직접 분류(1차), v2 남은 오류 15건, 수치 RAG 오답 92건 유형 분류 — 단위 환산(백만원) 오류가 가장 많음"],
], widths=[4.0, 12.6])
P("평가셋의 사람 검수와 사람 작성 문항(60개)은 팀 요청으로 Claude가 임시로 대신했고, 보고서 전체에서 따로 표시한다(8절).",
  italic=True, size=9)

# ------------------------------------------------------------------ 1
H("1. 문제와 목표")
P("사업보고서는 회사·연도마다 구조가 비슷해서, 검색기가 같은 회사의 다른 사업연도 문단이나 다른 회사의 같은 섹션을 "
  "가져오는 혼동이 잦다. 또한 재무 수치는 표 안에 '(단위: 백만원)'처럼 단위와 함께 있어, LLM이 그대로 읽으면 자릿수를 틀리기 쉽다.")
B("검색: 공개 한국어 임베딩(KURE-v1)을 공시 문서로 직접 미세조정해, 같은 데이터·같은 지표에서 기준선과 비교한다.", "목표 1 ")
B("수치: LLM은 숫자를 만들지 않고, 값은 OpenDART에서 조회해 Python으로 계산한 뒤 LLM은 설명만 쓰게 한다.", "목표 2 ")
B("한국어 전처리: Kiwi 형태소 분석(BM25), 회사명·사업연도·기수 추출, 금액 단위 정규화의 효과를 숫자로 보인다.", "목표 3 ")

# ------------------------------------------------------------------ 2
H("2. 데이터")
T(["데이터", "규모", "용도·주의"], [
    ["평가 코퍼스", "10개사 × 2022–2024 사업보고서 30건 → 34,427청크", "OpenDART document.xml 원문 직접 파싱(표 포함), 정정공시는 최신본"],
    ["학습 코퍼스", "평가와 겹치지 않는 20개사 원문 60건 → 48,712청크", "학습 쌍 생성용. 평가 회사·청크·질문과 겹침 검사(assert)"],
    ["학습 쌍 (M1)", "4,646쌍", "Qwen2.5-7B로 생성. 형식 비율(연도 명시 40% / 생략 45% / 상대 15% 등)을 생성 전에 고정"],
    ["FinDartBench", "dev 672 / test 1,947문항", "공개 QA(LLM 생성)를 원문 코퍼스에 매핑. 질문 69%에 연도 없음"],
    ["자체 평가셋 v1", "197문항, dev 50 / test 147", "EEVE-10.8B 초안 140 + 대행 작성 60, 5유형, 정답 근거 청크 ID"],
    ["OpenDART 재무", "10개사 × 3년 × 6계정 (연결)", "fnlttSinglAcntAll, 모든 호출 캐시. 국민은행 FY2022는 '013'(조회 결과 없음)"],
], widths=[3.0, 5.4, 8.2])
P("생성 LLM과 답변 LLM을 나눴다: 학습 쌍은 Qwen2.5-7B, 평가셋 초안은 EEVE-10.8B, 답변은 EXAONE-3.5-7.8B. "
  "질문을 만든 모델이 답까지 하면 자기 말투에 맞춘 질문이라 점수가 부풀 수 있기 때문이다.")

# ------------------------------------------------------------------ 3
H("3. 방법")
H("3.1 기준선과 한국어 전처리", 2)
B("BM25: 공백 토큰 vs Kiwi 형태소(내용어). 공개 임베딩: KURE-v1 (bge-m3 기반 한국어 검색 모델).")
B("질문 메타데이터 추출(규칙 기반): 회사 별칭(하이닉스·한전·KB국민은행 등), '2024 사업연도', '제56기'(기수 → 사업연도), 상대 표현.")
B("금액 정규화: '1조 2,345억 원', △·괄호 음수, 표 단위(천원·백만원·억원)를 원 단위 float로 통일 (finsight.numbers).")
H("3.2 직접 학습 1: 검색 임베딩 미세조정", 2)
B("KURE-v1을 MultipleNegativesRanking(CachedMNRL) 손실로 1 epoch 학습. negative 구성 5종(in-batch / easy / hard / hard 50% / 다른 섹션)을 dev MRR로 비교해 'hard'(같은 회사의 다른 사업연도 문단)를 골랐다.")
B("연도가 없는 질문에는 다른 연도 문단도 정답일 수 있으므로, 연도를 명시한 질문에만 다른 연도 negative를 쓴다.")
H("3.3 하이브리드 검색과 재정렬", 2)
B("Kiwi BM25와 임베딩 순위를 RRF(k=60)로 합치고, 질문에서 뽑은 회사·사업연도로 필터(연도 y와 y+1 보고서 허용).")
B("직접 학습 2: cross-encoder 재정렬기를 같은 학습 쌍으로 학습 — KLUE-RoBERTa, KLUE-BERT, bge-reranker-v2-m3.")
H("3.4 v2 개선 (dev로 단계별 선택)", 2)
B("'2024 사업보고서' 같은 표기를 사업연도 2024로 해석.")
B("두 회사 비교 질문을 회사별 단일 질문('기아의 2022 사업연도 연결 기준 자산총계는 얼마인가?')으로 나눠 검색한 뒤 번갈아 합침.")
B("재정렬 후보 20 → 40, 연도가 없는 질문은 최신 사업연도(2024) 보고서에서 검색.")
H("3.5 수치 검증 파이프라인", 2)
T(["시스템", "LLM에 주는 것", "핵심 수치를 누가 정하나"], [
    ["S0 LLM 단독", "질문", "LLM"],
    ["S1 RAG", "질문 + 검색 top-5 발췌", "LLM (발췌를 읽고 단위 환산·계산)"],
    ["S2 RAG + OpenDART", "질문 → 조회 계획(회사·연도·계정·연산) → 확인된 수치 + 발췌", "Python (OpenDART 값으로 계산), LLM은 설명만"],
], widths=[3.4, 7.4, 5.8])

# ------------------------------------------------------------------ 4
H("4. 실험 설정")
B("선택 규칙: 조합·임계값·하이퍼파라미터는 dev로만 고르고 test는 마지막에 한 번. 규칙은 실행 전에 결정 로그(docs/decisions.md)에 적었다. test를 다시 쓴 경우는 8절에 모두 밝혔다.")
B("검색 지표: Recall@1·5·10, MRR@10. 수치 지표: Numeric Claim Accuracy(0.5% 허용, 재작성 값 허용), Coverage, Unsupported Claim Rate, 답 없는 질문 거절률.")
B("통계: McNemar 정확 검정, 부트스트랩 95% CI(FinDartBench는 근거 문맥 단위 클러스터, 자체 평가셋은 문항 단위). seed 42 (임베딩·재정렬기는 43·44로 반복 확인).")

# ------------------------------------------------------------------ 5
H("5. 결과")
H("5.1 기준선과 임베딩 미세조정 (FinDartBench test 1,947문항)", 2)
T(["시스템", "R@5"], [
    ["BM25 공백", "0.445"], ["BM25 Kiwi 내용어", "0.615"], ["KURE-v1 공개 (dense)", "0.597"],
    ["KURE-v1 미세조정 (dense)", "0.615"], ["공개 KURE + BM25 RRF", "0.667"], ["RRF + 회사·연도 필터", "0.705"],
], caption="표 1. 기준선 (원문 코퍼스 34,427청크)", widths=[8, 3])
B("형태소 분석만으로 BM25 R@5가 0.445 → 0.615 (+17%p)로 올랐다.")
B("미세조정 임베딩은 전체로는 공개 KURE와 유의차가 없었다: +1.8%p [−2.2, +5.9]. seed 3개 모두 같은 결론.")
B("연도가 있는 질문에서는 +8.6%p [+2.4, +15.1], 회사가 있는 질문에서는 +7.0%p [+1.3, +13.3]로 유의했다.")
H("5.2 자체 평가셋 v1 (test 134문항, 정답 청크 있는 문항)", 2)
T(["시스템", "R@1", "R@5", "MRR@10"], [
    ["BM25 Kiwi 내용어", "0.067", "0.164", "0.102"],
    ["공개 KURE + RRF + 필터", "0.097", "0.299", "0.188"],
    ["미세조정 KURE + RRF + 필터", "0.187", "0.455", "0.313"],
    ["+ KLUE-RoBERTa 재정렬 (직접 학습)", "0.284", "0.575", "0.417"],
    ["+ bge 재정렬 미세조정 (v1 최종)", "0.418", "0.657", "0.525"],
    ["**v2 최종 (분해·연도 규칙·후보 40)**", "**0.463**", "**0.888**", "**0.656**"],
], caption="표 2. 자체 평가셋 v1 test", widths=[7.4, 2.4, 2.4, 2.4])
B("자체 평가셋은 모든 질문에 회사·연도가 있어, 미세조정 이득이 분명하다: 같은 설정에서 공개 KURE 대비 R@5 +15.7%p [+9.0, +22.4].")
B("v2에서 기업 비교형 R@5가 0.20 → 0.83으로 가장 크게 올랐다(두 회사 근거를 회사별로 따로 찾음).")
H("5.3 v2 개선 (FinDartBench test, 첫 사용)", 2)
T(["시스템", "R@1", "R@5", "MRR@10"], [
    ["공개 KURE + RRF + 필터", "0.332", "0.703", "0.485"],
    ["v1 최종", "0.399", "0.757", "0.544"],
    ["**v2 최종**", "**0.592**", "**0.818**", "**0.690**"],
], caption="표 3. FinDartBench test 1,947문항", widths=[7.4, 2.4, 2.4, 2.4])
B("이득은 대부분 '연도 없는 질문 → 최신 사업연도' 규칙에서 나왔다(R@1 +19.3%p). 단, FinDartBench의 연도 없는 질문은 96%가 2024 보고서가 정답이라 데이터셋 성질에 기댄다. 실제 시스템은 '최신 보고서 기준'임을 답에 밝힌다.")
B("FinDartBench에서는 v2 규칙 아래 공개 KURE가 미세조정과 비슷하거나 약간 낫다(R@5 −1.0%p). 미세조정 이득은 수치 표 질문에서 나온다.")
H("5.4 수치 검증 (자체 평가셋 v1 test, 수치 96 + 답 없음 13)", 2)
T(["시스템", "수치 정답률", "Coverage", "근거 없는 금액 비율", "답 없는 질문 거절"], [
    ["S0 LLM 단독", "0.0%", "11.5%", "해당 없음", "13/13"],
    ["S1 RAG", "4.2%", "55.2%", "40.4%", "13/13"],
    ["**S2 RAG + OpenDART**", "**95.8%**", "**99.0%**", "**0.0%**", "13/13"],
], caption="표 4. 수치 검증 (EXAONE-3.5-7.8B-Instruct, greedy)", widths=[4.4, 2.8, 2.6, 3.6, 3.2])
B("S2 − S1: +91.7%p [+85.4, +97.9], McNemar p < 10⁻²⁴.")
B("RAG(S1)는 정답 표를 찾아도 틀렸다. 92문항 중 26개가 '백만원'을 원으로 바꾸지 않아 10⁶배 작은 값이었고, 연도 비교·기업 비교(계산 필요)는 0%였다.")
B("S2 오답 4건은 모두 조회 계획 오류(연도 해석, 기수 '제22기', '영업손실' 계정, 두 회사 계획)였다.")
H("5.5 KLUE-BERT 비교", 2)
T(["모델", "dense R@5 (v1 / FDB)", "v2 + 재정렬 R@5 (v1 / FDB)"], [
    ["KLUE-BERT 학습 없음", "0.037 / 0.185", "0.567 / 0.814"],
    ["KLUE-BERT 미세조정 (직접 학습)", "0.269 / 0.449", "0.806 / 0.823"],
    ["KURE-v1 공개", "0.201 / 0.600", "0.701 / 0.828"],
    ["KURE-v1 미세조정", "0.313 / 0.614", "0.888 / 0.818"],
], caption="표 5. 바이인코더 비교 (v1 test 134 / FinDartBench test 1,947)", widths=[6.0, 5.0, 5.6])
B("범용 BERT(KLUE-BERT)는 학습 없이는 검색에 거의 못 쓰지만, 같은 학습 쌍으로 미세조정하면 7배(0.04 → 0.27)가 된다. 검색용으로 사전학습된 KURE에서 시작하는 편이 더 좋아서 KURE를 기준 모델로 삼았다.")
B("KLUE-BERT 재정렬기는 재정렬 없음 대비 v1 R@5 0.522 → 0.828로, KLUE-RoBERTa(0.784)보다 높고 bge 재정렬 미세조정(0.888)보다 낮다.")

# ------------------------------------------------------------------ 6
H("6. 오류 분석")
T(["분석", "결과"], [
    ["1차 오류 30건 (FinDartBench, 직접 읽음)", "회사 없는 질문 11 / 연도 없는 질문 8 / 인접·유사 청크 5 / 라벨 불완전 4 / 모호 2"],
    ["연도 혼동 (1위가 같은 회사·섹션의 다른 연도)", "미세조정 전후 26.4% → 26.2%로 그대로. 연도 없는 질문에서 35% → v2의 최신 연도 규칙으로 대응"],
    ["v2 남은 오류 15건 (자체 평가셋)", "라벨 불완전·모호 6 / 다른 연도 보고서 3 / 같은 회사의 다른 표(주석·부문) 4 / 관련 서술 청크 2"],
    ["수치 RAG 오답 (92건)", "모름 43 / 백만원 미반영(10⁶배) 26 / 천원 단위 오류 3 / 다른 숫자 20 / 정답 4"],
], widths=[5.6, 11.0])
P("공통 원인은 '연도·회사 정보가 질문에 없거나 문서 구조가 비슷한 것'과 '표 단위'다. 전자는 메타데이터 필터·질문 분해·최신 연도 규칙으로, "
  "후자는 OpenDART 조회와 Python 계산으로 대응했다. 남은 오류는 섹션 사전확률(재무제표 섹션 우선)과 정답 라벨 보강으로 다룰 계획이다.")

# ------------------------------------------------------------------ 7
H("7. 재현성")
B("코드: GitHub finsight-ko (scripts/, src/finsight/, tests/ pytest 44개 통과). 데이터·체크포인트·.env는 커밋하지 않는다.")
B("파이프라인은 tmux 단계별 완료 표시(.done_*)로 이어서 실행된다: midterm_run.sh → evalset_v1_run.sh → v2_run.sh → numeric_qa.py → klue_bert_run.sh.")
B("모든 선택 규칙과 변경 이력은 docs/decisions.md, 상세 표는 docs/midterm-report.md에 있다.")

# ------------------------------------------------------------------ 8
H("8. 한계와 공개 사항")
B("자체 평가셋의 검수(초안 140문항: 통과 116 / 수정 21 / 폐기 3)와 사람 작성 60문항은 팀 요청으로 Claude가 대행했다. 사람 검수·사람 문항이 아니며, Cohen's κ 교차 검수는 아직 하지 않았다. 팀 문항으로 바꾼 뒤 같은 스크립트로 다시 잰다.", "대행 표시: ")
B("자체 평가셋 test는 v1 평가(11/18), v2 평가(11/19), KLUE-BERT 분석(11/19)에서 세 번 쓰였다. v2 개선은 처음 test에서 본 약점(기업 비교형)을 고친 것이므로 v2 수치는 낙관적일 수 있다. 방법 선택은 모두 dev로만 했다. FinDartBench test는 v2에서 처음 썼다.", "test 사용 이력: ")
B("자체 평가셋은 수치 문항이 많고 학습 쌍의 숫자 질문 형식과 가깝다. dev가 50문항(검색 평가 43)으로 작아 선택이 흔들릴 수 있다.", "평가셋 성격: ")
B("FinDartBench의 최신 연도 규칙 이득은 데이터셋 성질(연도 없는 질문의 정답이 대부분 2024)에 기댄다.", "데이터셋 편향: ")
B("수치 검증은 6개 계정(매출·영업이익·순이익·자산·부채·자본), 연결 기준만 다룬다. 서술형 답변의 품질은 사람 평가 전이다.", "범위: ")
B("학습 쌍 사람 검수는 아직 없다(Claude가 30쌍 판정).", "학습 쌍: ")

# ------------------------------------------------------------------ 9
H("9. 다음 단계 (13–15주차)")
T(["할 일", "목적"], [
    ["팀 작성 문항·사람 검수로 평가셋 교체, κ 교차 검수", "평가셋 신뢰도 확보, 새 test로 깨끗한 최종 평가"],
    ["새 test 문항(아직 안 쓴 계정: 유동자산·현금흐름 등)", "여러 번 쓴 test 대신 처음 보는 test로 일반화 확인"],
    ["섹션 사전확률(재무제표·요약재무정보 우선), 라벨 보강", "남은 검색 오류(다른 표·라벨 불완전) 감소"],
    ["조회 계획에 기수·별칭 규칙 결합, 계정 확대", "수치 검증 S2 계획 오류 감소, 범위 확장"],
    ["서술형 답변 사람 평가(30문항) 또는 LLM judge + 일치도", "답변 품질 평가"],
    ["데모(Streamlit)와 답변 출처 표시", "확장 목표"],
], widths=[8.2, 8.4])

doc.save(sys.argv[1])
print("saved", sys.argv[1])

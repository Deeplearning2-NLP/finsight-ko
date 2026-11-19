// FinSight 중간 보고서 (MidProgress 가이드 5.2 구성). 숫자 출처: docs/midterm-report.md
const fs = require("fs");
const D = require("docx");
const { Document, Packer, Paragraph, TextRun, Table, TableRow, TableCell, WidthType, ShadingType,
  AlignmentType, HeadingLevel, BorderStyle, ImageRun, LevelFormat, Footer, PageNumber, PageBreak } = D;

const CFG = JSON.parse(fs.readFileSync(__dirname + "/config.json", "utf8"));
const FONT = "맑은 고딕";
const W = 9638; // A4 text width in DXA with 2.0cm margins

// ---------- helpers ----------
function runs(text, base = {}) {
  // **bold** markup
  const out = [];
  text.split(/(\*\*[^*]+\*\*)/).forEach((t) => {
    if (!t) return;
    if (t.startsWith("**")) out.push(new TextRun({ text: t.slice(2, -2), bold: true, font: FONT, ...base }));
    else out.push(new TextRun({ text: t, font: FONT, ...base }));
  });
  return out;
}
const P = (t, o = {}) => new Paragraph({ children: runs(t, o.run || {}), spacing: { after: 80, line: 290 }, alignment: o.align, ...(o.p || {}) });
const B = (t, lvl = 0) => new Paragraph({ children: runs(t), numbering: { reference: "bul", level: lvl }, spacing: { after: 40, line: 280 } });
const H1 = (t) => new Paragraph({ heading: HeadingLevel.HEADING_1, children: [new TextRun({ text: t, font: FONT })] });
const H2 = (t) => new Paragraph({ heading: HeadingLevel.HEADING_2, children: [new TextRun({ text: t, font: FONT })] });
const CAP = (t) => new Paragraph({ children: runs(t, { size: 17, color: "000000" }), spacing: { before: 40, after: 160 } });
const TCAP = (t) => new Paragraph({ children: runs(t, { size: 18, bold: true, color: "000000" }), spacing: { before: 120, after: 60 }, keepNext: true });

function table(rows, widths, opt = {}) {
  const total = widths.reduce((a, b) => a + b, 0);
  const border = { style: BorderStyle.SINGLE, size: 4, color: "BFBFBF" };
  const borders = { top: border, bottom: border, left: border, right: border };
  return new Table({
    width: { size: total, type: WidthType.DXA }, columnWidths: widths,
    rows: rows.map((r, i) => new TableRow({
      tableHeader: i === 0, cantSplit: true,
      children: r.map((c, j) => {
        const hl = opt.highlight && opt.highlight.includes(i);
        return new TableCell({
          borders, width: { size: widths[j], type: WidthType.DXA },
          margins: { top: 50, bottom: 50, left: 80, right: 80 },
          shading: i === 0 ? { type: ShadingType.CLEAR, fill: "E7E6E6", color: "auto" } : hl ? { type: ShadingType.CLEAR, fill: "FBE9DF", color: "auto" } : undefined,
          children: String(c).split("\n").map((line) => new Paragraph({
            alignment: j > 0 && opt.numCols && opt.numCols.includes(j) ? AlignmentType.CENTER : AlignmentType.LEFT,
            children: runs(line, { size: opt.size || 17, bold: i === 0 || undefined }),
          })),
        });
      }),
    })),
  });
}
const img = (file, wPx, hPx) => new Paragraph({ alignment: AlignmentType.CENTER, children: [new ImageRun({ type: "png", data: fs.readFileSync(__dirname + "/" + file), transformation: { width: wPx, height: hPx } })] });

// ---------- human-review numbers (filled when the team returns sheets) ----------
const HR = CFG.human || {};
const pairsLine = HR.pairs_n
  ? `학습 쌍 **${HR.pairs_n}쌍을 팀원 3명이 함께 읽고 합의로 판정**했다: 답가능 ${HR.pairs_ok} / 베끼기 ${HR.pairs_copy} / 모호 ${HR.pairs_amb} / 오류 ${HR.pairs_err}. **사용 가능(답가능+베끼기) ${HR.pairs_use_rate}, 오류 ${HR.pairs_err_rate}, 모호까지 포함한 문제 쌍 ${HR.pairs_bad_rate}** (95% CI ${HR.pairs_bad_ci}). 오류는 숫자 질문에서 더 많았다(18.2% vs 서술 12.1%): 표 단위가 없거나 합계 행을 개별 항목으로 읽은 경우다. 모호는 회사 생략형의 일반 질문이 많다. 함께 판정했으므로 독립 일치도(κ)는 재지 않았다. 학습 쌍의 약 40%가 문제 쌍이라는 점은 fine-tuning 이득이 작은 원인 후보이며, 13주차에 생성 필터를 보강한다.`
  : "학습 쌍 사람 검수는 진행 중이다.";
const evalLine = HR.eval_n
  ? `평가셋 **${HR.eval_n}문항(유형별 6개)을 백윤우·박채영이 함께 합의로 검수**했다: 통과 ${HR.eval_pass} / 수정 ${HR.eval_rev} / 폐기 ${HR.eval_drop}. 수정 사유는 주로 (1) 청크 머리말 단위('원')와 표 안 단위('백만원')가 엇갈리는 근거, (2) 답 없는 질문의 대상 보고서·기준 시점이 불명확함, (3) 서술형 정답이 잘린 경우다. 폐기는 없었다. 두 사람이 따로 판정하지 않아 κ는 없으며, 13주차 사람 문항에서 30문항 독립 교차 검수로 κ를 잰다. 수정 판정 문항은 검색 평가에서 근거 청크가 맞는지 다시 확인한다.`
  : "평가셋 교차 검수는 진행 중이다.";

const ids = CFG.student_ids || {};
const member = (n) => (ids[n] ? `${n} (${ids[n]})` : n);

// ---------- content ----------
const C = [];
// Title block
C.push(new Paragraph({ alignment: AlignmentType.CENTER, spacing: { after: 60 }, children: [new TextRun({ text: "FinSight 중간 보고서", font: FONT, size: 36, bold: true, color: "000000" })] }));
C.push(new Paragraph({ alignment: AlignmentType.CENTER, spacing: { after: 60 }, children: [new TextRun({ text: "DART 사업보고서 검색 모델 fine-tuning과 OpenDART 수치 검증", font: FONT, size: 24 })] }));
C.push(new Paragraph({ alignment: AlignmentType.CENTER, spacing: { after: 40 }, children: [new TextRun({ text: "DSC3033 딥러닝2: 자연어처리 · 6팀", font: FONT, size: 20 })] }));
C.push(new Paragraph({ alignment: AlignmentType.CENTER, spacing: { after: 40 }, children: [new TextRun({ text: [member("백윤우"), member("임병건"), member("박채영")].join(" · "), font: FONT, size: 20 })] }));
C.push(new Paragraph({ alignment: AlignmentType.CENTER, spacing: { after: 200 }, children: [new TextRun({ text: "2026년 11월 19일", font: FONT, size: 17, color: "000000" })] }));

// 0. 초록
C.push(H1("초록"));
C.push(P("한국어 사업보고서는 회사·사업연도마다 문단이 거의 같아서, 검색기가 같은 회사의 다른 사업연도 문단을 1위로 가져오는 일이 많다(FinDartBench test에서 공개 임베딩 하이브리드 1위의 26%). 우리는 평가와 겹치지 않는 20개사 원문으로 만든 질문–문단 쌍 4,646개로 KURE-v1을 대조 학습하고, 같은 쌍으로 cross-encoder 재정렬기를 학습했다. 평가는 10개사 × 2022–2024 사업보고서 원문 34,427청크에서 했다. **자체 평가셋 test 134문항(첫 사용)에서 fine-tuning 임베딩은 공개 KURE-v1보다 R@5 +15.7%p [+9.0, +22.4] 높았지만, 공개 벤치마크 FinDartBench test 1,947문항에서는 +1.8%p [−2.2, +5.9]로 유의하지 않았다.** 이득은 질문에 회사·연도가 있을 때 생긴다. 남은 질문은 사람이 쓴 처음 보는 문항에서도 이 차이가 유지되는가이다."));

// 1. 서론
C.push(H1("1. 서론"));
C.push(P("**입력과 출력.** 입력은 한국어 질문 하나다(예: \"현대자동차 2024 사업연도 연결 영업이익은 전년보다 얼마나 변했나?\"). 출력은 근거 문단 top-5와, 수치 질문이면 OpenDART 공식 값으로 확인한 숫자다(15.13조 → 14.24조원, −5.9%)."));
C.push(P("**왜 중요한가.** 공시 질의응답에서 틀린 연도의 문단을 가져오면 답의 숫자가 통째로 틀린다. 또 표의 숫자는 '(단위: 백만원)' 같은 단위와 함께 있어서, LLM이 그대로 읽으면 자릿수를 틀린다(6.5절: RAG가 답한 53개 중 24개가 자릿수만 틀림)."));
C.push(P("**연구 질문.**"));
C.push(B("RQ1. 공시 문서로 직접 fine-tuning한 검색 모델은 공개 모델보다 **같은 문항에서** 맞는 회사·사업연도의 근거를 유의하게 더 잘 찾는가?"));
C.push(B("RQ2. 숫자를 LLM이 만들지 않고 OpenDART 조회 + Python 계산으로 정하면 틀린 숫자가 얼마나 줄어드는가?"));

// 2. 관련 연구
C.push(P("**이 보고서에서 보이는 것.** (1) 평가와 회사가 겹치지 않는 학습 데이터로 검색 모델 두 종류(임베딩, 재정렬기)를 직접 학습하고, 공개 모델과 같은 문항에서 짝지어 비교한 결과. (2) 형태소 분석, 회사·연도 필터 같은 전처리·규칙의 효과를 학습 모델의 효과와 분리한 결과. (3) 학습 쌍과 평가셋을 팀이 직접 읽고 검수한 결과와, 그 검수가 실험 결과를 어떻게 설명하는지. (4) LLM이 표 숫자를 틀리는 양상과 공식 API 조회로 이를 막는 효과."));
C.push(H1("2. 관련 연구"));
C.push(table([
  ["연구", "내용", "이 프로젝트와 다른 점"],
  ["DPR (Karpukhin et al., 2020)", "질문–문단 쌍과 in-batch negative로 dense 검색기를 대조 학습", "같은 방법을 쓰되, negative를 '같은 회사·다른 사업연도 문단'으로 골라 연도 구분을 직접 학습"],
  ["FinQA (Chen et al., 2021), TAT-QA (Zhu et al., 2021)", "영어 재무 보고서의 표+문장에서 수치 추론", "정답 문맥이 주어진다. 우리는 30개 보고서에서 맞는 회사·연도 문맥을 먼저 찾아야 하고, 한국어"],
  ["FinanceBench (Islam et al., 2023)", "미국 공시 문서 기반 질의응답, 검색 실패가 주요 오류", "검색 실패를 '다른 연도/다른 회사/같은 회사 다른 섹션'으로 나눠 측정하고 그 유형을 학습 목표로 삼음"],
  ["M3-Embedding / bge-m3 (Chen et al., 2024), KURE-v1", "다국어·한국어 검색 임베딩 (KURE-v1은 bge-m3를 한국어로 학습, 568M)", "공시 도메인으로 다시 fine-tuning하고, 학습 회사와 평가 회사를 완전히 분리해 효과를 측정"],
  ["RAG (Lewis et al., 2020), RAGAS (Es et al., 2023)", "검색 결과를 넣어 생성, 근거 일치 자동 평가", "숫자는 LLM이 아니라 공식 API 값으로 정하고, 그 효과를 숫자 단위 정확도로 측정"],
], [2400, 3500, 3738]));
C.push(P("동료평가에서 \"회사·연도 맞추기는 규칙으로 대체할 수 있지 않나\"라는 질문이 있었다. 실제로 회사·연도 필터와 비교 질문 분해는 규칙이다. 그래서 규칙의 효과와 학습 모델의 효과를 같은 문항에서 **따로 떼어** 보고한다(6.1절 표 1의 행 순서). 필터를 켠 뒤에도 학습한 재정렬기가 R@5를 더 올린다."));

// 3. 데이터
C.push(H1("3. 데이터"));
C.push(H2("3.1 출처, 이용 조건, 수집·정제"));
C.push(table([
  ["데이터", "규모", "용도", "이용 조건"],
  ["OpenDART 사업보고서 원문 (document.xml)", "평가 10개사 × FY2022–2024 = 30건 → 34,427청크\n학습 다른 20개사 × 3년 = 60건 → 48,712청크", "검색 코퍼스 / 학습 쌍 재료", "공공 API, 키는 환경 변수, 응답 캐시"],
  ["OpenDART 재무 API (fnlttSinglAcntAll)", "30건 × 6계정 중 169칸", "수치 정답, 수치 검증", "공공 API"],
  ["FinDartBench (HF)", "사업보고서 질문 → 원문 청크 매핑: dev 672 / test 1,947", "공개 평가셋", "CC BY-NC 4.0, 비상업·재배포 금지 (본문에 원문을 싣지 않음)"],
  ["자체 평가셋 v1", "197문항 (dev 50 / test 147)", "주 평가셋", "팀 제작"],
], [2300, 3300, 1700, 2338]));
C.push(P("**정제 과정.** 대상 회사의 사업보고서 목록에서 사업연도는 접수일이 아니라 보고서명의 (YYYY.12)로 정했다(FY2024 보고서는 2025년에 접수). 같은 사업연도의 정정공시는 최신본만 남겨 30건 중 9건이 정정본이다. 원문 XML의 이스케이프되지 않은 '<' 등을 고쳐 섹션 누락을 없앴다(현대차 FY2024 679 → 1,400청크). 표는 Markdown 행으로 바꾸고, 청크가 나뉘면 표 머리글을 반복했다."));
C.push(P("**분할 기준(누수 방지).** 학습 쌍은 평가와 겹치지 않는 20개사에서만 만든다. 학습 스크립트가 평가 회사, 평가 보고서(rcept_no), 평가 청크 ID, FinDartBench 질문이 학습 쌍에 없음을 assert하고, 모든 학습 실행에서 통과했다. 연도 분할(2022–23 학습, 2024 평가)은 같은 회사 문단이 해마다 거의 같아서 누수가 생기므로 쓰지 않았다. 평가셋의 dev/test는 근거 청크를 공유하는 문항을 같은 쪽에 두는 방식으로 나눴다(seed 42)."));

C.push(TCAP("표 B. 프로포절(10/23) 대비 바뀐 숫자와 이유"));
C.push(table([
  ["항목", "프로포절", "현재", "바뀐 이유"],
  ["학습 쌍", "5,000쌍, 모든 질문에 연도·회사 포함", "4,646쌍, 연도 생략 45% · 회사 생략 23% · 별칭 7%", "연도가 없는 질문에서 fine-tuning이 오히려 나빠짐(0.584 → 0.562). 형식 기준을 평가셋을 보기 전에 다시 정하고 생성 필터를 형식별로 바꿈"],
  ["자체 평가셋", "200문항 (45×4 + 답 없음 20)", "197문항 (40 / 48 / 48 / 41 / 20)", "초안 140문항 검수에서 3문항 폐기. 비교형(연도·기업)이 검색에서 가장 어려워 비교형을 늘림"],
  ["dev / test", "50 / 150", "50 / 147", "근거 청크를 공유하는 문항을 한쪽에 몰아 누수를 막다 보니 test가 3문항 줄어듦"],
  ["주 평가 지표", "R@1·R@5·MRR", "R@5·MRR@10 + 짝지은 CI·McNemar", "프로포절 피드백(짝지은 비교로 유의성 보고)"],
  ["직접 학습 모델", "임베딩 1종", "임베딩 + 재정렬기 (KLUE-RoBERTa, KLUE-BERT, bge)", "임베딩만으로 공개 모델 대비 유의한 이득이 없을 때를 대비한 plan B를 앞당김"],
], [1500, 2300, 2600, 3238], { size: 16 }));
C.push(H2("3.2 학습 쌍"));
C.push(P("Qwen2.5-7B-Instruct(로컬 vLLM, seed 42)가 20개사 원문 청크 3,539개를 보고 질문 6,838개를 만들었고, 형식 필터를 통과한 **4,646개(68%)**를 썼다. 프로포절 피드백(\"평가셋과 독립된 기준으로 다양화\")에 따라 질문 형식을 **평가셋을 보기 전에** 고정했다: 연도 명시 40% / 생략 45% / 상대 표현 15%, 회사 공식명 55% / 별칭 15% / 생략 30%, 수치 질문 50%. Qwen이 규칙을 자주 어겨 최종 비율은 연도 명시 46% / 생략 45% / 상대 8%, 회사 공식명 70% / 생략 23% / 별칭 7%이다. 생성과 필터의 각 단계 개수는 표 C와 같다."));
C.push(TCAP("표 C. 학습 쌍 생성·정제 (평가 외 20개사 원문)"));
C.push(table([
  ["단계", "개수", "설명"],
  ["원문 청크", "48,712", "20개사 × FY2022–2024 사업보고서 60건"],
  ["생성에 쓴 청크", "3,539", "핵심 섹션(사업의 내용, 재무, 경영진단) 위주, 회사·연도당 최대 65개"],
  ["생성 질문", "6,838", "Qwen2.5-7B-Instruct, 형식 지시 포함"],
  ["  − 회사 생략 지시인데 회사명 포함", "562", "형식 위반"],
  ["  − 상대 연도 규칙 위반 / 연도 생략 지시인데 연도 포함", "448 / 439", "형식 위반"],
  ["  − 별칭 규칙 위반", "266", "형식 위반"],
  ["  − 연도 누락 / 회사명 누락 (명시 지시인데)", "200 / 164", "형식 위반"],
  ["  − 답의 숫자가 청크에 없음 등 / 지시어·'원문에 없음'", "85 / 28", "내용 오류"],
  ["최종 학습 쌍", "4,646 (68%)", "hard negative: 다른 사업연도 2,155개, 그 외 같은 회사 다른 섹션"],
], [4300, 1500, 3838], { numCols: [1], size: 16 }));
C.push(P(pairsLine));
C.push(table([
  ["판정 (99쌍, 3인 합의)", "전체", "숫자 질문 (33)", "서술 질문 (66)"],
  ["답가능", "54", "15", "39"],
  ["베끼기", "4", "0", "4"],
  ["모호", "27", "12", "15"],
  ["오류", "14", "6", "8"],
  ["**사용 가능 비율**", "**58.6%** [48.7, 67.8]", "45.5%", "65.2%"],
], [3400, 2200, 2000, 2038], { numCols: [1, 2, 3], size: 16 }));
C.push(CAP("판정 기준은 부록 C. 대괄호는 Wilson 95% CI. 오류 예: 표의 '합계' 행 값을 개별 항목(기타채무)의 값으로 답한 쌍, 단위 표기가 없는 표에서 수치를 묻는 쌍. 모호 예: 회사를 생략한 질문이 여러 회사에 공통인 내용을 묻는 쌍, 청크가 질문의 일부만 답하는 쌍."));

C.push(H2("3.3 자체 평가셋 v1"));
C.push(table([
  ["유형", "정의", "문항 (dev / test)", "실제 예시"],
  ["1 단일 수치", "한 회사·한 연도·계정 1개", "40", "삼성전자 2024 사업연도 연결 영업이익은? → 32.7조원"],
  ["2 연도 비교", "같은 회사, 두 사업연도", "48", "현대차 영업이익 2023→2024 변화는? → 15.13 → 14.24조 (−5.9%)"],
  ["3 기업 비교", "같은 업종 두 회사, 같은 연도", "48", "2024 영업이익, 삼성전자 vs SK하이닉스? → 32.7 > 23.5조"],
  ["4 서술 요약", "사업·위험·시장 요약", "41", "HMM 2024 보고서의 2025 시황 위험은? → 공급과잉·관세·중동 분쟁"],
  ["5 답 없음", "보고서에 원래 없는 정보", "20", "삼성전자 경영진의 2026년 말 목표 주가는? → 확인할 수 없음"],
  ["합계", "", "197 (50 / 147)", "검색 평가는 근거 청크가 있는 dev 43 / test 134문항"],
], [1400, 2500, 1500, 4238], { highlight: [] }));
C.push(P("**출처.** 두 가지 방식으로 만든 LLM 생성 문항이다. (1) EEVE-Korean-Instruct-10.8B 초안 140문항(수치 96문항은 OpenDART 정답 템플릿을 자연어로 바꿔 씀) 중 검수를 통과한 137문항. (2) 실제 사용자 표현에 가깝게 별칭(하이닉스, 한전, 기아차), 기수(제22기), 구어체, 새 회사 쌍을 섞은 변형 문항 60개. 생성 모델은 학습 쌍 생성 모델(Qwen)·답변 모델(EXAONE)과 다르다. 두 방식의 점수는 따로 적는다(6.2절). 원문을 보지 않고 사람이 직접 쓴 문항은 13주차에 만들어 새 test로 쓴다."));
C.push(P("**라벨링 가이드라인 요약** (전문은 부록 C). 질문에 회사명·사업연도·연결/별도를 쓴다. 숫자는 원 단위로 바꾸고, 재작성 값(다음 해 보고서의 전기 값)은 대체 정답으로 둔다. 근거 청크는 답을 담은 청크를 모두 적는다. 검수는 ① 질문이 명확한가 ② 정답이 하나인가 ③ 근거 청크만 읽고 답할 수 있는가를 보고 통과/수정/폐기로 판정한다."));
C.push(P(evalLine));
C.push(TCAP("표 D. 평가셋 30문항 검수에서 '수정' 23건의 사유"));
C.push(table([
  ["사유", "건수", "검색 평가에 미치는 영향"],
  ["답 없는 질문에 대상 보고서·기준 시점을 더 명시하라", "6", "없음 (정답 청크가 없는 유형). 질문 문구만 다듬음"],
  ["기업 비교의 두 번째 회사 근거가 시트에 안 보임", "6", "없음. 정답 목록에는 두 회사 청크가 모두 있고, 시트가 근거를 3개까지만 보여 줌"],
  ["청크 머리말 단위('원')와 표 안 단위('백만원') 불일치, 또는 표 단위 누락", "6", "검색 정답은 맞음. 단, RAG가 숫자를 읽을 때 자릿수 오류의 원인이 될 수 있음 → 파서 점검(8절)"],
  ["질문 문구가 두 가지로 읽힘 ('두 배 넘게', LCR 적용 기준)", "2", "문구 수정"],
  ["서술 정답이 잘림 / 질문 범위가 근거보다 넓음", "2", "정답 문장 보완, 질문 범위 축소"],
  ["재작성 값의 근거 청크 누락", "1", "대체 정답의 근거 청크 추가"],
], [4100, 800, 4738], { numCols: [1], size: 16 }));
C.push(P("정리하면 검색 평가의 정답(근거 청크 ID)이 틀린 문항은 없었고, 수정은 질문 문구·서술 정답·표시 방식에 관한 것이었다. 다만 청크 머리말의 단위가 표 안의 단위와 엇갈리는 경우가 30문항 중 6건에서 보였는데, 이는 6.5절에서 RAG가 자릿수를 틀린 24건과 관련이 있을 수 있어 13주차에 파서를 점검한다."));

// 4. 전처리
C.push(H1("4. 전처리 (한국어 특성)"));
C.push(B("**형태소 토큰화:** BM25를 공백 단위에서 Kiwi 형태소(명사·동사·외국어 등 내용어만)로 바꿨다. '매출액은', '매출액이'가 같은 토큰이 되어 FinDartBench test R@5가 0.445 → 0.615(+17.0%p)로 오른다. 이 프로젝트에서 단일 요소로 가장 큰 개선이다."));
C.push(B("**숫자·단위 정규화:** '1조 2,345억 원', '12,345(백만원)', '△1,234', '(1,234)'를 원 단위 float로 바꾸는 함수(numbers.py)를 테스트와 함께 만들었다. 표의 단위는 표 바로 앞의 '(단위: …)'를 우선한다. 공식 값과 2/2 일치."));
C.push(B("**회사·연도 추출:** 회사명·별칭 사전과 규칙으로 질문의 회사·사업연도를 뽑는다(질문 30개에서 회사 P/R 0.875/1.000, 연도 1.000/1.000). '제22기' 같은 기수, '2024 사업보고서' 표기도 처리한다. 뽑힌 값으로 후보를 거르는 필터가 R@1을 +4%p 올린다."));
C.push(B("**입력 길이:** 임베딩·재정렬기 입력은 512토큰으로 고정했다(KURE는 8,192까지 받지만 GPU 메모리와 재정렬기 입력을 맞추려는 선택). FinDartBench 근거 문단의 88%가 512토큰을 넘고, 그냥 자르면 R@1이 0.642 → 0.434로 떨어졌다. 그래서 문단·문장·표 행 경계에서 512토큰 이하로 나누고 50토큰을 겹쳤다. 결과 34,427청크는 평균 466토큰, 512 초과 0개다. 청크 87%가 표를 포함한다. LLM에는 top-5 청크만 넣는다."));

// 5. 방법
C.push(TCAP("표 E. 공시 데이터 함정과 처리 (실제 사례)"));
C.push(table([
  ["함정", "실제 사례", "처리", "확인"],
  ["접수일 ≠ 사업연도", "FY2024 사업보고서는 2025년 3월에 접수", "보고서명 (2024.12)에서 사업연도 추출", "단위 테스트"],
  ["정정공시", "평가 30건 중 9건이 [기재정정]", "같은 사업연도는 최신 접수본만", "30건 목록 대조"],
  ["단위", "표마다 원 / 천원 / 백만원", "표 바로 앞 '(단위: …)' 우선, 원 단위로 변환", "OpenDART 값과 2/2 일치"],
  ["연결 / 별도", "국민은행 FY2022는 OpenDART 응답 013", "013은 '조회 결과 없음'으로 처리, 연결재무제표가 없다고 단정하지 않음", "해당 문항 제외"],
  ["원문 XML 결함", "이스케이프되지 않은 '<'로 섹션 누락", "파싱 전 정규화", "현대차 FY2024 679 → 1,400청크, 누락 섹션 0"],
], [1700, 2900, 3000, 2038], { size: 16 }));
C.push(H1("5. 방법"));
C.push(P("검색은 다음 단계를 하나씩 더하며 비교한다: 공백 BM25 → Kiwi BM25 → 공개 KURE-v1 dense → **fine-tuning KURE-v1** → Kiwi BM25와 dense의 RRF(k=60) → 회사·연도 필터 → **학습한 cross-encoder 재정렬**(top-20/40). 직접 학습한 모델은 굵게 표시했다."));
C.push(TCAP("표 A. 학습 설정 (모든 학습은 같은 4,646쌍)"));
C.push(table([
  ["항목", "임베딩 (bi-encoder)", "재정렬기 (cross-encoder)"],
  ["모델", "KURE-v1 (bge-m3 기반, 568M) / 비교: klue/bert-base (110M)", "klue/roberta-base, klue/bert-base, BAAI/bge-reranker-v2-m3"],
  ["입력", "질문 / 청크 각각 최대 512토큰", "[질문; 청크] 최대 512토큰"],
  ["손실", "CachedMNRL (in-batch negative + hard negative 1개)", "BCE: (질문, 정답)=1, 저장된 negative 최대 3개=0 (16,093행)"],
  ["hard negative", "연도 있는 질문: 같은 회사·같은 섹션의 다른 사업연도 / 그 외: 같은 회사 다른 섹션. 5가지 구성을 dev로 비교", "임베딩과 같은 negative"],
  ["하이퍼파라미터", "batch 32, lr 1e-5, 1 epoch, seed 42 (43·44 반복)", "batch 16, lr 2e-5, 1 epoch, seed 42 (43·44 반복)"],
  ["자원", "RTX 4090 1장, 135–230초, 최대 13.2GB", "KLUE 81초 4.7GB / bge 260초 15.2GB"],
], [1700, 4100, 3838]));
C.push(P("**수치 검증.** 답변 LLM은 EXAONE-3.5-7.8B-Instruct(greedy)다. S0 LLM 단독(질문만), S1 RAG(검색 top-5 발췌), S2 RAG + OpenDART(LLM이 회사·연도·계정·연산의 조회 계획을 세우고 Python이 OpenDART 값으로 계산, LLM은 설명만)를 비교한다."));
C.push(P("**지표와 선택 규칙.** 검색은 R@1·R@5·MRR@10(정답 청크 중 하나라도 top-k에 있으면 맞음), 수치는 Numeric Claim Accuracy(공식 값 ±0.5%)·Coverage·근거 없는 금액 비율. 두 시스템 차이는 같은 문항 짝지은 부트스트랩 95% CI(FinDartBench는 근거 문맥 단위 클러스터, 1,000회)와 McNemar 검정으로 본다. **모든 선택(negative 구성, 재정렬기, 규칙)은 dev로만 했고 규칙은 실행 전에 결정 로그에 적었다.**"));

// 6. 중간 결과
C.push(H2("재현성"));
C.push(P("seed는 42로 고정하고 임베딩·재정렬기는 43·44로도 학습했다. 분할 파일(FinDartBench dev/test, 자체 평가셋 dev/test), dev 선택 결과(negative 구성·재정렬기·규칙), 학습 설정, 결정 로그(규칙을 실행 전에 기록)를 저장소에 남겼다. 학습·평가는 단계별 완료 표시가 있는 스크립트 하나로 이어서 실행할 수 있고, 최종 제출 때 이 코드로 표 1의 숫자를 다시 만든다. 원문 데이터와 체크포인트는 라이선스와 용량 때문에 저장소에 올리지 않고 다운로드 스크립트로 대신한다."));

C.push(H1("6. 중간 결과"));
C.push(P("**핵심 질문에 대한 답.** 학습 질문을 고친 뒤 fine-tuning한 KURE-v1은 공개 KURE-v1보다 **질문에 회사·연도가 있는 문항에서는 유의하게 낫고**(자체 평가셋 R@5 +15.7%p, FinDartBench의 연도 있는 질문 +8.6%p), **연도·회사가 없는 질문까지 섞인 공개 벤치마크 전체에서는 유의하지 않다**(+1.8%p). 같은 학습 쌍으로 학습한 재정렬기는 두 평가셋 모두에서 유의하게 좋아졌다."));
C.push(H2("6.1 결과표"));
C.push(TCAP("표 1. 검색 결과 (원문 코퍼스 34,427청크). FinDartBench dev 672 / test 1,947문항, 자체 평가셋 v1 dev 43 / test 134문항(첫 사용)"));
C.push(table([
  ["모델", "종류", "FDB dev\nMRR", "FDB test\nR@5", "FDB test\nMRR", "자체 dev\nMRR", "자체 test\nR@5", "자체 test\nMRR"],
  ["BM25 공백", "baseline", "—", "0.445", "0.298", "—", "—", "—"],
  ["BM25 Kiwi 내용어", "baseline (전처리)", "—", "0.615", "0.427", "—", "0.164", "0.102"],
  ["공개 KURE-v1", "baseline (사전학습)", "0.491", "0.597", "0.412", "—", "0.201", "0.170"],
  ["**KURE-v1 fine-tuning**", "fine-tuning", "0.494", "0.615", "0.423", "—", "0.313", "0.193"],
  ["공개 KURE + RRF + 필터", "baseline + 규칙", "0.586", "0.705", "0.485", "—", "0.299", "0.188"],
  ["**fine-tuning + RRF + 필터**", "fine-tuning + 규칙", "0.595", "0.714", "0.497", "0.314", "0.455", "0.313"],
  ["  + bge 재정렬 (학습 없음)", "공개 모델", "0.599", "0.759", "0.553", "0.282", "0.455", "0.335"],
  ["  + **KLUE-RoBERTa 재정렬 학습**", "fine-tuning", "0.573", "—", "0.523", "0.470", "0.575", "0.417"],
  ["  + **bge 재정렬 fine-tuning**", "fine-tuning", "0.595", "0.757", "0.544", "0.510", "0.657", "0.525"],
], [2650, 1550, 900, 900, 900, 900, 900, 938], { numCols: [2, 3, 4, 5, 6, 7], size: 16, highlight: [7, 9] }));
C.push(CAP("주황 줄은 각 평가셋의 dev로 고른 최종 설정이다(FinDartBench dev는 bge 재정렬 학습 없음, 자체 dev는 bge 재정렬 fine-tuning). '—'는 그 조건에서 재지 않은 값. seed 42 기준이며, seed 43·44에서 fine-tuning dense test R@5는 0.613–0.619이다. FinDartBench는 공개 LLM 생성 질문이라 연도 없는 질문이 69%, 회사 없는 질문이 68%다. 자체 평가셋은 모든 질문에 회사·연도가 있다."));
C.push(P("dev와 test가 같은 방향인지도 확인했다. FinDartBench에서 fine-tuning은 dev·test 모두 공개 모델과 거의 같다(dev MRR 0.494 vs 0.491). 재정렬기는 FinDartBench dev에서 거의 효과가 없었지만(0.595 → 0.599) test에서는 유의하게 좋아져(0.497 → 0.553) dev와 test가 어긋났다. dev 근거 문맥이 22개뿐이라 선택이 흔들린 것으로 본다."));

C.push(H2("6.2 공개 모델 vs fine-tuning: 같은 문항 짝지은 비교 (RQ1)"));
C.push(TCAP("표 2. fine-tuning − 공개 KURE-v1 (같은 설정, 차이 %p [95% CI], McNemar p)"));
C.push(table([
  ["평가셋 (문항)", "설정", "R@5 차이", "해석"],
  ["FinDartBench test (1,947)", "dense", "+1.8 [−2.2, +5.9], p=0.07", "유의하지 않음 (seed 6개 모두 CI가 0 포함)"],
  ["", "RRF + 필터", "+1.0 [−1.3, +3.1], p=0.19", "유의하지 않음"],
  ["  연도 있는 질문 (608)", "dense", "**+8.6 [+2.4, +15.1]**", "유의"],
  ["  연도 없는 질문 (1,339)", "dense", "−1.3 [−5.8, +3.1]", "차이 없음"],
  ["  회사 있는 질문 (617)", "dense", "**+7.0 [+1.3, +13.3]**", "유의"],
  ["자체 평가셋 v1 test (134, 첫 사용)", "dense", "**+11.2 [+5.2, +17.9]**, p=0.001", "유의"],
  ["", "RRF + 필터", "**+15.7 [+9.0, +22.4]**, p<0.001", "유의"],
  ["  LLM·템플릿 문항 (95)", "RRF + 필터", "+16.8 [+9.5, +25.3]", "유의"],
  ["  변형 문항 (39)", "RRF + 필터", "+12.8 [+0.0, +25.6]", "경계 (MRR은 +12.2 [+5.6, +20.0])"],
], [2700, 1300, 2800, 2838]));
C.push(img("fig2_year_r5.png", 520, 227));
C.push(CAP("그림 1. FinDartBench test, 질문에 연도가 있는지에 따른 dense R@5. fine-tuning 이득은 연도가 있는 질문에만 있다."));
C.push(P("**해석.** 두 평가셋의 결과는 같은 이야기다. fine-tuning 이득은 **질문에 회사·연도가 있을 때** 생긴다. FinDartBench는 연도 없는 질문이 69%라 전체 평균에서 이득이 묻힌다. 학습 질문의 45%를 연도 생략형으로 바꿨는데도 연도 없는 질문은 나아지지 않았다(S1 연도 포함 학습과 test 수치가 거의 같음). 연도가 없으면 질문만으로 정답 연도를 정할 수 없는 경우가 많아, 학습으로 풀 문제가 아니라고 본다. 주의할 점: 자체 평가셋은 수치 문항이 113/134로 많고 학습 쌍의 숫자 질문 형식과 가까워 이득이 과대평가됐을 수 있다. 같은 회사·연도 요약재무 청크를 여러 문항이 공유해 문항 단위 CI는 실제보다 좁을 수 있다."));
C.push(img("fig1_type_r5.png", 520, 244));
C.push(CAP("그림 2. 자체 평가셋 v1 test(첫 사용) 유형별 R@5. 수치형에서 이득이 크고, 기업 비교형(두 회사 근거 필요)은 모든 시스템에서 낮다."));
C.push(P("**재정렬기(직접 학습).** 같은 쌍으로 학습한 KLUE-RoBERTa 재정렬기는 FinDartBench test에서 공개 KURE RRF+필터보다 R@5 **+4.2%p [+1.3, +7.1]**이며 seed 3개 모두 +4.1–4.2%p로 유의하다. 자체 평가셋에서는 bge 재정렬기를 학습하지 않으면 효과가 없고(0.455 → 0.455), 우리 쌍으로 fine-tuning하면 0.657로 오른다. 재정렬기도 공시 쌍으로 학습해야 효과가 난다."));

C.push(H2("6.3 연도·회사 혼동률"));
C.push(TCAP("표 3. FinDartBench test 1,947문항, 1위 청크의 종류 (전체 문항 대비)"));
C.push(table([
  ["시스템", "1위 정답", "다른 사업연도", "다른 회사", "같은 회사 다른 섹션"],
  ["공개 KURE RRF + 필터", "33.3%", "26.4%", "14.8%", "25.5%"],
  ["fine-tuning RRF + 필터", "34.4%", "26.2%", "15.2%", "24.2%"],
  ["fine-tuning RRF + 필터 + bge 재정렬", "40.5%", "27.1%", "10.3%", "22.1%"],
], [3600, 1400, 1500, 1400, 1738], { numCols: [1, 2, 3, 4] }));
C.push(P("프로포절의 연도 혼동 27.1%는 줄지 않았다(26.4% → 26.2%). 연도가 있는 질문은 필터 덕분에 6.6%까지 내려가고, **남은 연도 혼동은 거의 전부 연도가 없는 질문(35%)**이다. fine-tuning dense만 보면 '다른 회사' 혼동이 17.2% → 25.2%로 늘었는데, 학습 쌍의 23%가 회사 생략형이라 회사 단서에 덜 의존하게 배운 것으로 보인다. 재정렬을 붙이면 10.3%로 내려간다."));

C.push(H2("6.4 오류 분석"));
C.push(P("최종 시스템(fine-tuning RRF+필터+bge)의 FinDartBench test 실패(정답이 top-5 밖) 469건에서 seed 42로 30건을 뽑아 질문·정답 청크·1위 청크를 읽고 분류했다."));
C.push(TCAP("표 4. 오류 30건 유형 (FinDartBench test)"));
C.push(table([
  ["유형", "건수", "실제 예"],
  ["회사 없는 질문 → 다른 회사", "11", "\"이자율 100bp 변동 시 법인세비용차감전순손익 영향은?\" → 모든 회사 주석에 같은 문구가 있어 고를 근거가 없음"],
  ["연도 없는 질문 → 같은 회사 다른 사업연도", "8", "\"향후 투자계획에서 2025년 예상투자액 합계는?\" → '2025년'은 계획 연도라 필터가 못 잡고, FY2023 보고서의 같은 표가 1위"],
  ["같은 보고서의 인접·유사 청크", "5", "옵션 계약 설명이 두 청크에 나뉘어 바로 앞 청크가 1위"],
  ["라벨 불완전 (1위도 답을 담음)", "4", "\"기아의 주요 원재료와 공급 경로\" → 1위 '원재료 현황 표'가 정답으로 표시된 '사업의 개요'보다 직접적"],
  ["모호 (여러 연도를 묻는 질문)", "2", "\"2022~2024년 문서의 디지털 전환 사례\" → 정답이 FY2024 청크 하나로만 표시"],
], [2900, 700, 6038], { numCols: [1] }));
C.push(P("프로포절 때 1위였던 '다른 사업연도'(9/30)가 줄고 **'회사 없는 질문'이 가장 큰 유형**(6 → 11)이 됐다. 필터가 회사·연도가 있는 질문을 정리하니, 남은 실패는 질문 자체에 단서가 없는 경우다. 라벨 불완전·모호 6건(20%)은 시스템 잘못이라 보기 어려워, 자체 평가셋에서는 정답 청크를 여러 개 허용한다. 수치 RAG 오답(6.5절)도 따로 분류했다."));

C.push(H2("6.5 수치 검증 (RQ2)"));
C.push(TCAP("표 5. 수치 검증 (EXAONE-3.5-7.8B). 자체 평가셋 v1 dev 수치 26문항 / test 수치 96문항 + 답 없는 질문 13문항"));
C.push(table([
  ["시스템", "dev 정확도\n(±0.5%)", "test 정확도\n(±0.5%)", "test 정확도\n(단위 무시)", "test Coverage", "근거 없는 금액 비율", "답 없는 질문 거절"],
  ["S0 LLM 단독", "0.0%", "0.0%", "0.0%", "11.5%", "— (근거 없음)", "13/13"],
  ["S1 RAG (top-5)", "19.2%", "4.2%", "29.2%", "55.2%", "40.4%", "13/13"],
  ["**S2 RAG + OpenDART**", "**100.0%**", "**95.8%**", "96.9%", "99.0%", "0.0%", "13/13"],
], [2000, 1200, 1200, 1250, 1200, 1400, 1388], { numCols: [1, 2, 3, 4, 5, 6] }));
C.push(CAP("dev 26문항으로 답 파싱·연산 규칙만 고치고 test는 한 번 돌렸다. '단위 무시'는 10^k배 차이를 허용한 보조 채점으로, 저장된 답을 다시 채점한 것이다(새 생성 없음)."));
C.push(P("S2 − S1 = +91.7%p [+85.4, +97.9] (test, McNemar p<10⁻²⁴). S1이 답한 53개 중 4개만 맞았고, **24개는 자릿수만 틀렸으며(백만원·천원 단위 미반영) 25개는 숫자 자체가 틀렸다**(다른 셀을 읽거나 증감·차이 계산 오류). 단위를 무시해도 계산이 필요한 기업 비교는 0/36이다. 검색은 정답 표를 가져오는데 LLM이 단위와 계산에서 틀린다. S2의 오답 4건은 모두 조회 계획 오류(기수 '제22기', '영업손실' 계정, 두 회사 질문 계획)이며 모두 변형 문항(별칭·기수·구어체)에서 나왔다. **한계:** 수치 문항은 OpenDART가 다루는 6개 계정(매출·영업이익·순이익·자산·부채·자본) 템플릿에서 나와 S2가 유리한 조건이다. 95.8%는 LLM 실력이 아니라 조회기의 정확도이며, 6개 계정 밖의 숫자는 S2가 조회하지 못한다."));

C.push(H2("6.6 test 사용 이력과 그 뒤의 개선"));
C.push(table([
  ["평가셋", "사용", "무엇을 했나", "선택에 썼나"],
  ["자체 v1 test", "1회 (11/18)", "표 1·2의 자체 평가셋 숫자. 이 보고서의 핵심 숫자", "아니오 (dev 43문항으로 선택)"],
  ["", "2회 (11/19)", "v2 개선 측정. 개선 대상(기업 비교형, '2024 사업보고서' 표기)은 1회 test에서 본 약점", "아니오, 하지만 낙관적일 수 있음"],
  ["", "3회 (11/19)", "KLUE-BERT 비교 (분석)", "아니오"],
  ["FinDartBench test", "프로포절부터 여러 번", "모든 비교. 선택은 FinDartBench dev 672문항으로만", "아니오"],
], [1700, 1500, 4400, 2038]));
C.push(P("v2 개선은 dev로 단계별로 골랐다: 두 회사 질문을 회사별 단일 질문으로 나눠 검색, 재정렬 후보 20 → 40, '2024 사업보고서' = 2024 사업연도, 연도 없는 질문은 최신 사업연도. 자체 v1 test(2회째 사용)에서 R@5 0.657 → 0.888, 기업 비교형 0.20 → 0.83이 됐지만 **test에서 본 약점을 고친 것이라 이 숫자는 핵심 결과로 쓰지 않는다**(부록 D). FinDartBench R@1 0.399 → 0.592도 대부분 '연도 없으면 최신 연도' 규칙에서 나오며, FinDartBench dev의 연도 없는 질문 96%의 정답이 2024 보고서라는 데이터셋 성질에 기댄다. 최종 평가는 팀이 쓴, 시스템이 한 번도 보지 않은 문항으로 한다(8절)."));

// 7. 피드백 대응
C.push(H2("6.7 결과를 읽을 때 주의할 점"));
C.push(B("**평가셋의 성격:** 자체 평가셋은 수치 문항이 134문항 중 113문항이고, 모든 문항이 LLM으로 생성되었고 팀은 그중 30문항을 검수했다. 사람이 원문을 보지 않고 쓴 문항은 아직 없어서, 실제 사용자 질문에서의 성능은 13주차의 새 test로 확인해야 한다."));
C.push(B("**학습 데이터 품질:** 사람 검수에서 학습 쌍의 41%가 모호하거나 틀렸다. 학습 신호가 깨끗하지 않은 상태의 결과이므로, 현재 fine-tuning 이득은 하한에 가깝다고 본다."));
C.push(B("**dev 크기:** FinDartBench dev는 근거 문맥 22개, 자체 dev는 43문항이라 dev에서 고른 설정이 test와 어긋난 적이 있다(재정렬기). 선택을 바꾸지 않고 그대로 보고했다."));
C.push(B("**test 재사용:** 자체 평가셋 test를 세 번 썼다. 핵심 숫자는 첫 사용 결과만 쓰고, 이후 결과는 참고로만 적었다."));
C.push(B("**수치 검증 범위:** OpenDART 6개 계정에 한정된다. 주석이나 본문에만 있는 숫자는 아직 검증하지 못한다."));

C.push(H1("7. 피드백 대응"));
C.push(table([
  ["지적 (출처)", "한 일", "결과", "남은 일"],
  ["직접 학습 모델의 기여가 안 보임 (교수, 부족한 점 1)", "학습 질문 형식 다양화 후 재학습, 재정렬기 직접 학습", "자체 test +15.7%p (유의), FinDartBench +1.8%p (유의하지 않음), KLUE 재정렬 +4.2%p (유의)", "사람 문항 test에서 재확인"],
  ["학습 데이터 사람 검수 없음 (교수, 부족한 점 2)", "99쌍 3인 합의 검수", HR.pairs_n ? `3인 합의 판정: 사용 가능 ${HR.pairs_use_rate}, 오류 ${HR.pairs_err_rate}` : "진행 중", "오류 유형을 생성 필터에 반영"],
  ["평가에 맞춰 질문을 바꾸면 순환 구조 (교수·동료)", "형식 기준을 평가셋 보기 전에 고정 (연도·회사 생략·별칭·수치 비율)", "연도 없는 질문은 그래도 안 나아짐 → 학습이 아니라 규칙·되묻기로", "—"],
  ["같은 지표가 다른 설정에서 나와 헷갈림 (교수, 부족한 점 3)", "모든 표 제목에 코퍼스·문항 수·split, 결과를 표 1 하나로", "표 1", "—"],
  ["hard negative가 너무 어려울 수 있음 (교수, 보완 2)", "negative 5가지 구성(in-batch만/무작위 다른 회사/형식 맞춤 hard/절반 hard/다른 섹션)을 dev로 비교", "hard 선택. 차이 0.013 이내로 작음. hard는 R@1·MRR만 3 seed 모두 높음", "—"],
  ["짝지은 비교로 유의성 (교수, 보완 3)", "McNemar + 클러스터 부트스트랩, seed 3개", "표 2", "—"],
  ["발표 시간·밀도 (교수·동료)", "슬라이드 10장, 쪽마다 핵심 숫자 하나, 표는 부록", "—", "리허설로 10분 확인"],
  ["규칙으로 대체 가능하지 않나 (동료)", "규칙(필터·분해)과 학습 모델의 효과를 따로 측정", "필터 R@1 +4%p, 그 위에 학습한 재정렬기 R@5 +4.2%p", "—"],
  ["예외 입력 처리 (동료)", "답 없는 질문 20문항, 연도 없는 질문 규칙", "답 없는 질문 13/13 거절", "연도 없는 질문 되묻기 문구"],
  ["자체 평가셋 200, 사람 작성·일치도 (교수, 체크 1)", "197문항, dev 50 / test 147", HR.eval_n ? `30문항 합의 검수: 통과 ${HR.eval_pass} / 수정 ${HR.eval_rev}, κ는 13주차` : "κ 진행 중, 사람 작성 문항 0", "사람 문항 60 작성 → 새 test"],
], [2500, 2700, 2500, 1938], { size: 16 }));

// 8. 남은 계획
C.push(TCAP("표 F. 중간보고 가이드의 6팀 '꼭 보여 줄 것'과 이 보고서의 위치"));
C.push(table([
  ["항목", "상태", "위치"],
  ["자체 평가셋 200문항: Dev 50 / Test 150, 유형별 개수, 사람 작성 비율, 교차 검수 일치도", "197문항, dev 50 / test 147, 유형별 개수, 30문항 2인 합의 검수. 사람 작성 문항과 독립 κ는 13주차", "3.3절, 표 D"],
  ["학습 질문 다양화 기준 + 학습 쌍 약 200개 사람 검수 오류율", "기준은 생성 전에 고정. 99쌍 3인 검수: 사용 가능 58.6%, 오류 14.1%", "3.2절, 표 C"],
  ["공개 vs fine-tuned KURE-v1의 유형별 R@5, paired bootstrap 또는 McNemar", "두 평가셋, 유형별·연도 유무별, 클러스터 부트스트랩 + McNemar, seed 3개", "6.1–6.2절, 표 2, 그림 1·2"],
  ["회사·연도 혼동률(27.1%)의 변화와 남은 오류 유형", "26.4% → 26.2% (변화 없음), 남은 혼동은 연도 없는 질문. 오류 30건 분류", "6.3–6.4절"],
  ["fine-tuning 효과가 작을 때의 plan B(재정렬기) 진행", "재정렬기 3종 학습, KLUE-RoBERTa +4.2%p (seed 3개 모두 유의)", "6.2절, 부록 D"],
  ["BM25·필터와 fine-tuning 효과를 따로, 표마다 코퍼스·문항 수", "표 1의 행 순서로 분리, 모든 표 제목에 코퍼스·문항 수·split", "표 1"],
], [3600, 4000, 2038], { size: 16 }));

C.push(H1("8. 남은 계획 (13–16주차)"));
C.push(table([
  ["주차", "할 일", "담당"],
  ["13주 (11/23–)", "원문을 보지 않고 쓴 사람 문항 60개 작성(비교형 위주) → 근거 청크 표시 → 30문항 교차 검수 κ ≥ 0.6. 이 문항을 **새 test**로 고정(시스템 미사용)", "백윤우 주관, 공동"],
  ["13주", "학습 쌍 검수 결과로 생성 필터 보강(합계 행·단위 없는 표·회사 생략형 일반 질문 제거), 재생성·재학습 (dev로만 선택)", "임병건"],
  ["13주", "청크 머리말 단위와 표 안 단위가 엇갈리는 문제 점검·수정, 재정렬 결과와 RAG 자릿수 오류 재측정 (dev)", "박채영 · 임병건"],
  ["14주 (11/30–)", "수치 질문의 섹션 사전확률(재무제표·요약재무 우선), 정답 라벨 보강, 연도 없는 질문 '최신 연도 + 되묻기' 비교 (dev)", "임병건"],
  ["14주", "수치 조회 계획에 기수·별칭 규칙 결합, 계정 확대, 서술형 답변 사람 평가 30문항 (LLM 채점과 일치도)", "박채영"],
  ["15주 (12/7–)", "새 test 1회로 최종 평가, 오류 분석 재실시, Streamlit 데모(출처 표시, '투자 권유 아님'), 최종 발표", "공동"],
  ["16주", "최종 보고서, 코드·split·config 제출 (제출 코드로 표의 숫자 재현)", "공동"],
], [1700, 6200, 1738]));
C.push(P("**위험과 plan B.** (1) 사람 문항 test에서도 fine-tuning 임베딩의 이득이 유의하지 않으면, 이득이 확인된 **학습한 재정렬기를 대표 직접 학습 모델**로 하고 임베딩 결과는 '회사·연도가 있을 때만 효과'로 정리한다. (2) 사람 문항 작성이 늦어지면 30문항이라도 새 test로 쓰고 CI를 함께 적는다. (3) GPU는 학교 서버에서 tmux로 돌리고 단계별 완료 표시로 이어서 실행한다."));

// References
C.push(H1("참고문헌"));
[
  "Karpukhin et al. (2020). Dense Passage Retrieval for Open-Domain Question Answering. EMNLP.",
  "Lewis et al. (2020). Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks. NeurIPS.",
  "Chen et al. (2021). FinQA: A Dataset of Numerical Reasoning over Financial Data. EMNLP.",
  "Zhu et al. (2021). TAT-QA: A Question Answering Benchmark on a Hybrid of Tabular and Textual Content in Finance. ACL.",
  "Islam et al. (2023). FinanceBench: A New Benchmark for Financial Question Answering. arXiv:2311.11944.",
  "Es et al. (2023). RAGAS: Automated Evaluation of Retrieval Augmented Generation. arXiv:2309.15217.",
  "Chen et al. (2024). M3-Embedding (BGE-M3): Multi-Linguality, Multi-Functionality, Multi-Granularity Text Embeddings. arXiv:2402.03216.",
  "Park et al. (2021). KLUE: Korean Language Understanding Evaluation. NeurIPS Datasets and Benchmarks.",
  "nlpai-lab KURE-v1 (Hugging Face 모델 카드), FinDartBench (Hugging Face 데이터셋 카드), Kiwi 형태소 분석기 (bab2min/Kiwi).",
].forEach((t) => C.push(B(t)));

// Appendix
C.push(new Paragraph({ children: [new PageBreak()] }));
C.push(H1("부록 A. AI 도구 사용 내역"));
C.push(table([
  ["도구", "어디에", "어떻게 / 몇 건", "사람 확인"],
  ["Claude (Anthropic, Claude Code)", "코드 작성, 평가셋 초안 1차 검수 보조", "데이터 파싱·학습·평가 스크립트 작성, 초안 140문항의 형식 점검", "검수 결과는 팀이 확인"],
  ["Qwen2.5-7B-Instruct (로컬)", "학습 쌍 생성", "청크 3,539개 → 질문 6,838개 → 필터 통과 4,646개. 프롬프트: 청크와 형식 지시(연도·회사 표기 방식, 수치/서술)를 주고 질문 하나와 답을 생성", HR.pairs_n ? `${HR.pairs_n}쌍 팀원 3인 합의 검수` : "사람 검수 진행 중 (99쌍)"],
  ["EEVE-Korean-Instruct-10.8B (로컬)", "평가셋 초안", "140문항 (수치 96은 OpenDART 정답 템플릿을 자연어로 바꿔 씀, 회사·연도·계정 유지 확인)", HR.eval_n ? `${HR.eval_n}문항 2인 합의 검수` : "교차 검수 진행 중"],
  ["EXAONE-3.5-7.8B-Instruct (로컬)", "답변 생성 (수치 검증 실험)", "조회 계획과 설명 문장. 숫자는 Python이 OpenDART 값으로 계산", "—"],
], [2200, 2000, 3500, 1938], { size: 16 }));
C.push(P("생성 LLM(Qwen·EEVE)과 답변 LLM(EXAONE)을 서로 다른 모델로 두어, 답변 모델에 유리한 문장으로 평가셋이 만들어지지 않게 했다. LLM이 만든 데이터에는 generated_by 필드를 남겼다."));

C.push(H1("부록 B. 데이터 이용 조건"));
C.push(B("OpenDART: 금융감독원 공공 API. 인증키는 환경 변수로만 관리하고 코드·로그·저장소에 남기지 않는다. 응답은 캐시해 일 호출 한도를 지킨다."));
C.push(B("FinDartBench: CC BY-NC 4.0. 수업 과제용 비상업 사용, 재배포 금지. 보고서와 저장소에 원문 질문·문단을 싣지 않는다."));
C.push(B("모델: KURE-v1, bge-m3, bge-reranker-v2-m3, klue/bert-base, klue/roberta-base, Qwen2.5, EEVE, EXAONE-3.5는 각 모델 카드의 라이선스를 따른다(EXAONE은 연구 목적 비상업 라이선스)."));
C.push(B("학습 체크포인트·원문 데이터·평가셋 파일은 공개 저장소에 올리지 않는다."));
C.push(B("이 프로젝트의 결과와 시스템 출력은 연구 목적이며 투자 권유가 아니다."));

C.push(H1("부록 C. 라벨링 가이드라인 (요약 전문)"));
C.push(B("**범위:** 10개사 × FY2022–2024 사업보고서 30건. 근거는 원문 코퍼스의 청크."));
C.push(B("**질문에 쓰는 것:** 회사명, 사업연도(접수 연도 아님), 재무 수치면 연결/별도. 예외는 답 없는 질문과 일부러 연도를 빼는 문항(이유를 notes에)."));
C.push(B("**정답:** 숫자는 원 단위 float(표의 '(단위: 백만원)' 반영, △·괄호는 음수). 다음 해 보고서에서 다시 쓴 값은 대체 정답. 공식 값과 ±0.5% 안이면 정답. 답 없는 질문은 '보고서에서 확인할 수 없음', 근거 없음. 범위 밖이지만 실제 공시된 정보(예: FY2025 실적)는 답 없는 질문으로 만들지 않는다."));
C.push(B("**근거 청크:** 답을 담은 청크를 모두 적는다(표가 나뉘면 머리글 청크 우선). 수치 문항은 rcept_no·계정 ID·연결/별도를 함께 적는다."));
C.push(B("**사람 문항:** 원문을 보지 않고 먼저 질문을 쓰고, 그다음 근거를 찾는다. 원문 문장을 붙여 넣지 않는다. 비교형 위주."));
C.push(B("**검수:** ① 질문 명확 ② 정답 하나 ③ 근거 청크만으로 답 가능. 셋 다 예 = 통과, 고치면 통과 = 수정, 정답 없음·둘 이상·근거 못 찾음 = 폐기. 30문항(유형별 6)을 두 사람이 따로 판정해 Cohen's κ(3범주)와 p_o를 보고, κ < 0.6이면 기준을 고친 뒤 다시 한다."));
C.push(P("**학습 쌍 판정 기준:** 답가능(이 문단만으로 답이 되고 회사·연도가 맞음) / 베끼기(답은 되지만 문단 문장을 거의 그대로 옮김) / 모호(다른 회사·연도 문단도 답이 되거나 일부만 답함) / 오류(답하지 않거나 숫자·증감·회사·연도가 틀림)."));

C.push(H1("부록 D. 추가 표"));
C.push(TCAP("표 D1. hard negative 구성 비교 (FinDartBench dev 672문항, 선택 기준 dense MRR@10)"));
C.push(table([
  ["구성", "dense R@1", "dense R@5", "dense MRR", "RRF+필터 R@5"],
  ["공개 KURE-v1", "0.369", "0.668", "0.491", "0.772"],
  ["in-batch만", "0.344", "0.686", "0.484", "0.783"],
  ["무작위 다른 회사", "0.330", "0.696", "0.481", "0.798"],
  ["**형식 맞춤 hard (선택)**", "0.350", "0.716", "0.494", "0.807"],
  ["hard 절반", "0.342", "0.696", "0.491", "0.793"],
  ["다른 섹션만", "0.344", "0.680", "0.484", "0.790"],
], [3400, 1500, 1500, 1500, 1738], { numCols: [1, 2, 3, 4] }));
C.push(TCAP("표 D2. KLUE-BERT vs KURE (R@5; 자체 v1 test 134 / FinDartBench test 1,947, 자체 test 3번째 사용, 분석만)"));
C.push(table([
  ["모델", "dense 자체", "dense FDB", "+ 재정렬 bge-ft 자체 / FDB"],
  ["klue/bert-base 학습 없음", "0.037", "0.185", "0.567 / 0.814"],
  ["**klue/bert-base fine-tuning**", "0.269", "0.449", "0.806 / 0.823"],
  ["KURE-v1 공개", "0.201", "0.600", "0.701 / 0.828"],
  ["**KURE-v1 fine-tuning**", "0.313", "0.614", "0.888 / 0.818"],
], [3600, 1700, 1700, 2638], { numCols: [1, 2, 3] }));
C.push(P("범용 KLUE-BERT는 학습 없이는 검색에 거의 못 쓰지만(dense 0.04) 같은 쌍으로 학습하면 7배(0.27)가 된다. 검색용으로 사전학습된 KURE에서 시작하는 편이 낫다. KLUE-BERT로 학습한 재정렬기는 재정렬 없음 대비 자체 test R@5 +30.6%p다."));
C.push(TCAP("표 D3. v2 개선 (자체 v1 test 2번째 사용 / FinDartBench test, 참고용)"));
C.push(table([
  ["시스템", "자체 R@5", "자체 기업 비교 R@5", "FDB R@1", "FDB R@5"],
  ["v1 최종 (fine-tuning RRF+필터+bge-ft)", "0.657", "0.200", "0.399", "0.757"],
  ["v2 최종 (+ 질문 분해, 후보 40, 연도 규칙)", "0.888", "0.825", "0.592", "0.818"],
  ["v2 규칙 + 공개 KURE", "0.701", "—", "0.597", "0.828"],
], [3900, 1300, 1700, 1300, 1438], { numCols: [1, 2, 3, 4] }));
C.push(P("v2 규칙 아래에서 FinDartBench는 공개 KURE가 fine-tuning보다 약간 낫고(R@5 −1.0 [−1.8, −0.3]), 자체 평가셋은 fine-tuning이 +18.7%p 낫다. 미세조정 이득이 수치 표 질문에 있다는 6.2절 결론과 같다."));

// ---------- document ----------
const doc = new Document({
  styles: {
    default: { document: { run: { font: FONT, size: 20 } } },
    paragraphStyles: [
      { id: "Heading1", name: "Heading 1", basedOn: "Normal", next: "Normal", quickFormat: true, run: { font: FONT, size: 26, bold: true, color: "000000" }, paragraph: { spacing: { before: 260, after: 100 }, keepNext: true, outlineLevel: 0 } },
      { id: "Heading2", name: "Heading 2", basedOn: "Normal", next: "Normal", quickFormat: true, run: { font: FONT, size: 22, bold: true, color: "000000" }, paragraph: { spacing: { before: 180, after: 80 }, keepNext: true, outlineLevel: 1 } },
    ],
  },
  numbering: { config: [{ reference: "bul", levels: [{ level: 0, format: LevelFormat.BULLET, text: "•", alignment: AlignmentType.LEFT, style: { paragraph: { indent: { left: 360, hanging: 240 } } } }, { level: 1, format: LevelFormat.BULLET, text: "–", alignment: AlignmentType.LEFT, style: { paragraph: { indent: { left: 720, hanging: 240 } } } }] }] },
  sections: [{
    properties: { page: { size: { width: 11906, height: 16838 }, margin: { top: 1134, bottom: 1134, left: 1134, right: 1134 } } },
    footers: { default: new Footer({ children: [new Paragraph({ alignment: AlignmentType.CENTER, children: [new TextRun({ children: [PageNumber.CURRENT], font: FONT, size: 16, color: "000000" })] })] }) },
    children: C,
  }],
});
Packer.toBuffer(doc).then((b) => { fs.writeFileSync(process.argv[2] || "report.docx", b); console.log("ok"); });

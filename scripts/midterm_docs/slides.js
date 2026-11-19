// FinSight-ko 중간 발표 슬라이드 (MidProgress 가이드 4.1 구성, 10장 + 백업). 프로포절 덱 스타일.
const fs = require("fs");
const pptxgen = require("pptxgenjs");
const { applyTheme } = require(process.env.APPLY_THEME || "/mnt/skills/public/pptx/scripts/apply_theme.js");
const CFG = JSON.parse(fs.readFileSync(__dirname + "/config.json", "utf8"));
const HR = CFG.human || {};

const F = "Pretendard";
const K = { bg: "F2F0EC", card: "FBFAF8", line: "E2DDD6", ink: "1F1C19", sub: "6B655E", mute: "9A938A", org: "E8622C", dark: "2A2622", green: "2E7D4F", red: "C0392B", thead: "EDE9E3" };
const THEME = { name: "FinSight", headFontFace: F, bodyFontFace: F,
  colors: { dk1: "1F1C19", lt1: "FFFFFF", dk2: "2A2622", lt2: "F2F0EC", accent1: "E8622C", accent2: "2A2622", accent3: "B5AEA4", accent4: "2E7D4F", accent5: "6B655E", accent6: "C0392B", hlink: "E8622C", folHlink: "6B655E" } };

const pres = new pptxgen();
pres.layout = "LAYOUT_WIDE"; // 13.33 x 7.5
pres.theme = { headFontFace: F, bodyFontFace: F };
pres.title = "FinSight 중간 발표";
const TOTAL = 10;

pres.defineSlideMaster({ title: "CONTENT", background: { color: K.bg }, objects: [
  { text: { text: "FINSIGHT", options: { x: 0.55, y: 7.0, w: 3, h: 0.3, fontFace: F, fontSize: 9, color: K.mute, charSpacing: 4 } } },
  { line: { x: 0.55, y: 6.92, w: 12.23, h: 0, line: { color: K.line, width: 0.75 } } },
] });
pres.defineSlideMaster({ title: "COVER", background: { color: K.dark }, objects: [] });

function T(slide, text, o) { slide.addText(text, { fontFace: F, margin: 0, isTextBox: true, valign: "top", ...o }); }
function header(slide, n, label, title, pill) {
  slide.addShape(pres.shapes.RECTANGLE, { x: 0.55, y: 0.42, w: 0.11, h: 0.11, fill: { color: K.org }, line: { color: K.org } });
  T(slide, `${String(n).padStart(2, "0")} · ${label}`, { x: 0.75, y: 0.36, w: 6, h: 0.25, fontSize: 10, color: K.mute, charSpacing: 4 });
  T(slide, title, { x: 0.55, y: 0.62, w: 9.6, h: 0.75, fontSize: 30, bold: true, color: K.ink });
  if (pill) {
    slide.addShape(pres.shapes.ROUNDED_RECTANGLE, { x: 10.35, y: 0.55, w: 2.43, h: 0.46, rectRadius: 0.23, fill: { color: "FFFFFF" }, line: { color: K.line, width: 0.5 } });
    T(slide, pill, { x: 10.35, y: 0.55, w: 2.43, h: 0.46, fontSize: 11, color: K.org, align: "center", valign: "middle" });
  }
  T(slide, `${String(n).padStart(2, "0")} / ${String(TOTAL).padStart(2, "0")}`, { x: 11.3, y: 7.0, w: 1.48, h: 0.3, fontSize: 9, color: K.mute, align: "right", charSpacing: 4 });
}
function card(slide, x, y, w, h, fill = K.card) {
  slide.addShape(pres.shapes.ROUNDED_RECTANGLE, { x, y, w, h, rectRadius: 0.12, fill: { color: fill }, line: { color: K.line, width: 0.5 }, shadow: { type: "outer", color: "000000", opacity: 0.06, blur: 6, offset: 1, angle: 90 } });
}
function keynum(slide, x, y, w, h, num, label) {
  slide.addShape(pres.shapes.ROUNDED_RECTANGLE, { x, y, w, h, rectRadius: 0.12, fill: { color: K.dark }, line: { color: K.dark } });
  T(slide, "핵심 숫자", { x: x + 0.25, y: y + 0.2, w: w - 0.5, h: 0.25, fontSize: 10, color: "B5AEA4" });
  T(slide, num, { x: x + 0.25, y: y + 0.45, w: w - 0.5, h: 0.7, fontSize: 30, bold: true, color: K.org });
  T(slide, label, { x: x + 0.25, y: y + 1.15, w: w - 0.5, h: h - 1.3, fontSize: 11.5, color: "F2F0EC", lineSpacingMultiple: 1.15 });
}
function table(slide, rows, o) {
  const data = rows.map((r, i) => r.map((c) => {
    const cell = typeof c === "object" ? c : { text: String(c) };
    return { text: cell.text, options: { fontFace: F, fontSize: o.fs || 11, color: K.ink, bold: i === 0 || cell.bold, fill: { color: i === 0 ? K.thead : (cell.fill || "FFFFFF") }, valign: "middle", margin: [3, 6, 3, 6], align: cell.align || "left", ...(cell.color ? { color: cell.color } : {}) } };
  }));
  slide.addTable(data, { x: o.x, y: o.y, w: o.w, colW: o.colW, border: { type: "solid", color: K.line, pt: 0.5 }, rowH: o.rowH || 0.32, autoPage: false });
}
function bottomNote(slide, label, text) {
  card(slide, 0.55, 6.2, 12.23, 0.55, "FFFFFF");
  T(slide, [{ text: label + "  ", options: { color: K.org, bold: true } }, { text, options: { color: K.ink } }], { x: 0.8, y: 6.2, w: 11.8, h: 0.55, fontSize: 12, valign: "middle" });
}
const pairsTxt = HR.pairs_n ? `99쌍 3인 합의: 사용 가능 ${HR.pairs_use_rate} · 오류 ${HR.pairs_err_rate}` : "99쌍 3인 검수 진행 중";
const evalTxt = HR.eval_n ? `30문항 2인 합의: 통과 ${HR.eval_pass} · 수정 ${HR.eval_rev} · 폐기 0 (κ는 13주차)` : "30문항 2인 교차 검수 진행 중";

// ---------- 1. Cover ----------
let s = pres.addSlide({ masterName: "COVER" });
s.addShape(pres.shapes.RECTANGLE, { x: 0.65, y: 0.62, w: 0.13, h: 0.13, fill: { color: K.org }, line: { color: K.org } });
T(s, "FINSIGHT", { x: 0.88, y: 0.55, w: 4, h: 0.3, fontSize: 13, bold: true, color: "F2F0EC", charSpacing: 6 });
T(s, "DSC3033 딥러닝2 · TEAM 6", { x: 8.5, y: 0.55, w: 4.2, h: 0.3, fontSize: 11, color: "9A938A", align: "right", charSpacing: 4 });
T(s, "기말 프로젝트 중간 발표", { x: 0.65, y: 2.0, w: 7, h: 0.4, fontSize: 15, color: "B5AEA4" });
T(s, "FinSight:\nDART 공시 특화 검색 모델", { x: 0.65, y: 2.5, w: 7.2, h: 1.9, fontSize: 40, bold: true, color: "F2F0EC", lineSpacingMultiple: 1.1 });
T(s, "공시로 직접 fine-tuning한 검색 모델은\n공개 모델보다 맞는 회사·사업연도의 근거를 더 잘 찾는가?", { x: 0.65, y: 4.55, w: 7.2, h: 0.9, fontSize: 15, color: "D9D4CC", lineSpacingMultiple: 1.2 });
s.addShape(pres.shapes.ROUNDED_RECTANGLE, { x: 8.3, y: 2.2, w: 4.4, h: 3.0, rectRadius: 0.15, fill: { color: "34302B" }, line: { color: "4A453F", width: 0.75 } });
T(s, "오늘의 답", { x: 8.6, y: 2.45, w: 3.8, h: 0.3, fontSize: 12, bold: true, color: K.org });
T(s, [
  { text: "회사·연도가 있는 질문에서는 그렇다.", options: { bold: true, color: "F2F0EC", breakLine: true } },
  { text: " ", options: { breakLine: true, fontSize: 6 } },
  { text: "자체 평가셋 R@5 +15.7%p (유의)", options: { color: "F2F0EC", breakLine: true } },
  { text: "공개 벤치마크 전체 +1.8%p (유의하지 않음)", options: { color: "B5AEA4", breakLine: true } },
  { text: " ", options: { breakLine: true, fontSize: 6 } },
  { text: "숫자는 OpenDART로: 수치 정답률 4% → 96%", options: { color: "F2F0EC" } },
], { x: 8.6, y: 2.85, w: 3.85, h: 2.2, fontSize: 14, lineSpacingMultiple: 1.25 });
s.addShape(pres.shapes.LINE, { x: 0.65, y: 6.55, w: 12.03, h: 0, line: { color: "4A453F", width: 0.75 } });
T(s, "백윤우 · 임병건 · 박채영", { x: 0.65, y: 6.75, w: 5, h: 0.3, fontSize: 13, color: "D9D4CC" });
T(s, "2026.11.20", { x: 9.5, y: 6.75, w: 3.18, h: 0.3, fontSize: 11, color: "9A938A", align: "right", charSpacing: 4 });
s.addNotes("[0:15] 6팀 FinSight입니다. 오늘 질문은 하나입니다. 공시 문서로 직접 학습한 검색 모델이 공개 모델보다 맞는 회사·맞는 사업연도의 근거를 더 잘 찾는가. 답부터 말씀드리면, 질문에 회사와 연도가 있을 때는 그렇고, 공개 벤치마크 전체로는 아직 유의하지 않습니다.");

// ---------- 2. 문제 다시 보기 ----------
s = pres.addSlide({ masterName: "CONTENT" }); header(s, 2, "PROBLEM", "문제 다시 보기", "프로포절 이후 바뀐 점");
card(s, 0.55, 1.6, 5.4, 1.45); T(s, "입력  한국어 질문", { x: 0.8, y: 1.75, w: 5, h: 0.3, fontSize: 11, color: K.sub, bold: true });
T(s, "현대자동차의 2024 사업연도 연결 영업이익은 전년보다 얼마나 변했나?", { x: 0.8, y: 2.1, w: 5.0, h: 0.8, fontSize: 15, bold: true, color: K.ink });
s.addShape(pres.shapes.RIGHT_ARROW, { x: 6.1, y: 2.1, w: 0.5, h: 0.45, fill: { color: "A8A29A" }, line: { color: "A8A29A" } });
card(s, 6.75, 1.6, 6.03, 1.45); T(s, "출력  근거 문단 top-5 + 검증된 수치", { x: 7.0, y: 1.75, w: 5.6, h: 0.3, fontSize: 11, color: K.sub, bold: true });
T(s, "현대차 FY2024 사업보고서 요약재무정보 문단\n15.13조 → 14.24조원 (−5.9%), OpenDART 공식 값과 일치", { x: 7.0, y: 2.1, w: 5.6, h: 0.8, fontSize: 14, color: K.ink, lineSpacingMultiple: 1.2 });
T(s, "프로포절 이후 바뀐 점", { x: 0.55, y: 3.35, w: 6, h: 0.35, fontSize: 15, bold: true, color: K.ink });
[["학습 질문 형식", "모두 연도를 넣은 질문 → 연도 생략 45% · 회사 생략 23% · 별칭을 섞어 4,646쌍으로 재생성"],
 ["직접 학습 모델 추가", "임베딩 fine-tuning에 더해 cross-encoder 재정렬기를 같은 쌍으로 학습 (plan B를 앞당김)"],
 ["자체 평가셋 · 수치 검증", "197문항(dev 50 / test 147)을 만들고, LLM 단독 / RAG / RAG + OpenDART를 실제로 비교"]]
  .forEach((r, i) => {
    card(s, 0.55, 3.8 + i * 0.75, 8.6, 0.62, "FFFFFF");
    T(s, r[0], { x: 0.8, y: 3.8 + i * 0.75, w: 2.2, h: 0.62, fontSize: 13, bold: true, color: K.org, valign: "middle" });
    T(s, r[1], { x: 3.0, y: 3.8 + i * 0.75, w: 6.0, h: 0.62, fontSize: 12.5, color: K.ink, valign: "middle" });
  });
keynum(s, 9.4, 3.35, 3.38, 2.65, "26%", "공개 모델 하이브리드 검색의 1위 결과 중 '같은 회사·같은 섹션의 다른 사업연도' 문단\n(FinDartBench test 1,947문항)");
bottomNote(s, "왜 어려운가", "같은 회사의 사업보고서는 해마다 문단이 거의 같고, 숫자는 '(단위: 백만원)'과 함께 표 안에 있다");
s.addNotes("[0:30] 입력은 한국어 질문 하나, 출력은 근거 문단 다섯 개와 공식 값으로 확인한 숫자입니다. 어려운 이유는 같은 회사 보고서가 해마다 거의 같아서, 공개 모델은 1위 결과의 26%가 다른 사업연도 문단입니다. 프로포절 이후 세 가지가 바뀌었습니다. 학습 질문 형식을 다양화했고, 재정렬기를 하나 더 직접 학습했고, 자체 평가셋과 수치 검증을 실제로 돌렸습니다.");

// ---------- 3. 피드백 대응 ----------
s = pres.addSlide({ masterName: "CONTENT" }); header(s, 3, "FEEDBACK", "피드백 대응", "교수님 · 동료 피드백");
table(s, [
  ["지적", "한 일", "결과"],
  ["① 직접 학습한 모델의 기여가 안 보임", "학습 질문 형식을 다양화해 재학습\n+ cross-encoder 재정렬기 직접 학습", { text: "자체 평가셋 +15.7%p (유의)\nFinDartBench +1.8%p (유의하지 않음)", bold: true }],
  ["② 학습 데이터 사람 검수 없음 · 평가에 맞추면 순환 구조", "형식 기준(연도·회사 생략 비율)을 평가셋 보기 전에 고정\n학습 쌍 99쌍 3인 검수", pairsTxt],
  ["③ 같은 지표가 다른 설정에서 나와 헷갈림", "결과를 표 하나로, 표마다 코퍼스·문항 수·split 표기", "5쪽 결과표"],
  ["(동료) 규칙으로 대체 가능하지 않나", "규칙(회사·연도 필터)과 학습 모델의 효과를 따로 측정", "필터 R@1 +4%p, 그 위에 학습한 재정렬기 R@5 +4.2%p (유의)"],
], { x: 0.55, y: 1.6, w: 8.7, colW: [2.7, 3.3, 2.7], fs: 11.5, rowH: 0.85 });
keynum(s, 9.5, 1.6, 3.28, 2.6, "+15.7%p", "fine-tuning − 공개 KURE-v1, R@5\n자체 평가셋 test 134문항 (첫 사용)\n95% CI [+9.0, +22.4]");
card(s, 9.5, 4.4, 3.28, 1.6); T(s, "그대로 남은 것", { x: 9.72, y: 4.55, w: 2.9, h: 0.3, fontSize: 12, bold: true, color: K.red });
T(s, "공개 벤치마크 전체에서는 아직 유의하지 않다 → 6·9쪽에서 이유", { x: 9.72, y: 4.9, w: 2.9, h: 1.0, fontSize: 11.5, color: K.ink, lineSpacingMultiple: 1.15 });
bottomNote(s, "가장 중요한 지적 3개만", "나머지(hard negative 비율, 짝지은 검정, 발표 밀도)는 보고서 7절 대응표");
s.addNotes("[0:40] 가장 중요한 지적 세 가지입니다. 첫째, 직접 학습한 모델의 기여가 안 보인다는 지적에, 학습 질문을 다양화해 다시 학습하고 재정렬기를 추가로 학습했습니다. 자체 평가셋에서는 15.7%p로 유의하지만 공개 벤치마크 전체로는 여전히 유의하지 않습니다. 둘째, 학습 데이터를 사람이 보지 않았다는 지적에, 형식 기준을 평가셋을 보기 전에 고정했고 99쌍을 세 명이 검수했습니다. 셋째, 표가 헷갈린다는 지적에 결과를 표 하나로 모았습니다. 동료분이 물은 '규칙으로 되지 않느냐'는, 규칙과 학습 모델의 효과를 따로 재서 답했습니다.");

// ---------- 4. 데이터 확정 ----------
s = pres.addSlide({ masterName: "CONTENT" }); header(s, 4, "DATA", "데이터 확정", "실제 개수 · split · 라벨 품질");
table(s, [
  ["데이터", "실제 개수", "split", "비고"],
  ["평가 코퍼스 (OpenDART 원문)", "10개사 × FY2022–24 = 30건 → 34,427청크", "검색 대상", "정정공시는 최신본 (9건)"],
  ["학습 쌍", "다른 20개사 60건 → 질문 6,838 → 필터 통과 4,646쌍", "학습 전용", "Qwen2.5-7B 생성"],
  ["FinDartBench (공개)", "사업보고서 질문 2,619", "dev 672 / test 1,947", "CC BY-NC, 재배포 금지"],
  ["자체 평가셋 v1", "197문항 = 단일 40 · 연도 비교 48 · 기업 비교 48 · 서술 41 · 답 없음 20", "dev 50 / test 147", "LLM 생성: EEVE 초안 137 + 변형 60"],
], { x: 0.55, y: 1.6, w: 8.7, colW: [2.2, 3.4, 1.5, 1.6], fs: 11, rowH: 0.62 });
card(s, 0.55, 4.85, 4.2, 1.2); T(s, "분할 기준: 회사 단위", { x: 0.78, y: 4.97, w: 3.8, h: 0.3, fontSize: 12.5, bold: true, color: K.ink });
T(s, "학습 20개사 ∩ 평가 10개사 = ∅\n회사·보고서·청크·질문 겹침 0을 학습 때마다 assert", { x: 0.78, y: 5.32, w: 3.9, h: 0.7, fontSize: 11, color: K.sub, lineSpacingMultiple: 1.15 });
card(s, 4.95, 4.85, 4.3, 1.2); T(s, "라벨 품질 (사람)", { x: 5.18, y: 4.97, w: 3.9, h: 0.3, fontSize: 12.5, bold: true, color: K.ink });
T(s, `학습 쌍: ${pairsTxt}\n평가셋: ${evalTxt}`, { x: 5.18, y: 5.32, w: 3.95, h: 0.7, fontSize: 11, color: K.sub, lineSpacingMultiple: 1.15 });
keynum(s, 9.5, 1.6, 3.28, 2.6, "197문항", "자체 평가셋 · dev 50 / test 147\n모든 질문에 회사·사업연도 표기, 근거 청크 ID 표시");
card(s, 9.5, 4.4, 3.28, 1.65); T(s, "공개 사항", { x: 9.72, y: 4.55, w: 2.9, h: 0.3, fontSize: 12, bold: true, color: K.red });
T(s, "모든 문항이 LLM 생성. 사람이 원문을 안 보고 쓴 문항은 13주차에 새 test로 추가", { x: 9.72, y: 4.9, w: 2.9, h: 1.1, fontSize: 11, color: K.ink, lineSpacingMultiple: 1.15 });
bottomNote(s, "피드백 반영", "생성 LLM(Qwen·EEVE) ≠ 답변 LLM(EXAONE), LLM 데이터에 generated_by 표시");
s.addNotes("[1:00] 데이터는 확정됐습니다. 검색 대상은 10개사 3년치 사업보고서 원문 30건, 3만 4천 청크입니다. 학습 쌍은 평가와 겹치지 않는 다른 20개사에서 4,646개를 만들었고, 회사 단위로 나눠 겹침이 0인지 학습할 때마다 코드로 확인합니다. 평가셋은 공개 FinDartBench와 자체 평가셋 197문항 두 가지입니다. 자체 평가셋은 모두 LLM으로 만든 문항이고, 그중 30문항을 우리가 직접 검수했습니다. 학습 쌍도 99쌍을 세 명이 함께 검수해 사용 가능 비율이 58.6%였습니다. 사람이 직접 쓴 문항은 13주차에 새 test로 추가합니다.");

// ---------- 5. 실험 설정 ----------
s = pres.addSlide({ masterName: "CONTENT" }); header(s, 5, "SETUP", "실험 설정", "baseline vs fine-tuning");
const steps = [["baseline", "BM25\n공백 토큰"], ["한국어 전처리", "BM25\nKiwi 형태소"], ["사전학습", "KURE-v1\n공개"], ["직접 학습", "KURE-v1\nfine-tuning"], ["규칙", "RRF +\n회사·연도 필터"], ["직접 학습", "재정렬기\ncross-encoder"]];
steps.forEach((st, i) => {
  const x = 0.55 + i * 2.07, dark = st[0] === "직접 학습";
  s.addShape(pres.shapes.ROUNDED_RECTANGLE, { x, y: 1.6, w: 1.8, h: 1.25, rectRadius: 0.12, fill: { color: dark ? K.dark : "FFFFFF" }, line: { color: dark ? K.dark : K.line, width: 0.5 } });
  T(s, st[0], { x, y: 1.7, w: 1.8, h: 0.3, fontSize: 10.5, color: dark ? K.org : K.sub, align: "center", bold: true });
  T(s, st[1], { x, y: 2.0, w: 1.8, h: 0.75, fontSize: 13, bold: true, color: dark ? "F2F0EC" : K.ink, align: "center", valign: "middle" });
  if (i < 5) s.addShape(pres.shapes.RIGHT_ARROW, { x: x + 1.84, y: 2.1, w: 0.2, h: 0.25, fill: { color: "A8A29A" }, line: { color: "A8A29A" } });
});
table(s, [
  ["", "임베딩 (bi-encoder)", "재정렬기 (cross-encoder)"],
  ["모델", "KURE-v1 (568M) · 비교 klue/bert-base", "klue/roberta-base · klue/bert-base · bge-reranker-v2-m3"],
  ["입력", "질문 / 청크 각각 최대 512토큰", "[질문; 청크] 최대 512토큰, top-20 다시 정렬"],
  ["학습", "대조 학습 (in-batch + 같은 회사·다른 사업연도 hard negative)", "BCE: 정답 1, negative 3개 0"],
  ["설정", "batch 32 · lr 1e-5 · 1 epoch · seed 42/43/44", "batch 16 · lr 2e-5 · 1 epoch · seed 42/43/44"],
  ["자원", "RTX 4090 1장, 135–230초, 13.2GB", "81–260초, 4.7–15.2GB"],
], { x: 0.55, y: 3.15, w: 8.7, colW: [1.0, 3.85, 3.85], fs: 11, rowH: 0.42 });
keynum(s, 9.5, 3.15, 3.28, 2.85, "dev로만 선택", "negative 구성 5가지, 재정렬기, 규칙 모두 dev로 고르고 test는 마지막에.\n지표: R@5 · MRR@10, 같은 문항 짝지은 부트스트랩 CI + McNemar");
bottomNote(s, "같은 조건", "학습 데이터 4,646쌍 · 코퍼스 34,427청크 · 지표를 모든 모델에 똑같이 적용");
s.addNotes("[0:40] 비교는 한 단계씩 더하는 방식입니다. 공백 BM25, 형태소 BM25, 공개 KURE, 그리고 직접 학습한 KURE, 규칙인 하이브리드와 회사·연도 필터, 마지막으로 직접 학습한 재정렬기입니다. 어두운 칸이 직접 학습한 모델입니다. 둘 다 같은 4,646쌍으로 1 epoch, seed 세 개로 학습했고, 모든 선택은 dev로만 했습니다.");

// ---------- 6. 결과 ----------
s = pres.addSlide({ masterName: "CONTENT" }); header(s, 6, "RESULTS", "결과: 공개 모델 vs fine-tuning", "결과표 하나");
const O = { fill: "FBE9DF" };
table(s, [
  ["모델", "종류", "FinDartBench\ntest R@5", "자체 평가셋\ntest R@5"],
  ["BM25 공백 토큰", "baseline", { text: "0.445", align: "center" }, { text: "—", align: "center" }],
  ["BM25 Kiwi 형태소", "baseline", { text: "0.615", align: "center" }, { text: "0.164", align: "center" }],
  ["KURE-v1 공개", "baseline", { text: "0.597", align: "center" }, { text: "0.201", align: "center" }],
  [{ text: "KURE-v1 fine-tuning", bold: true }, "fine-tuning", { text: "0.615", align: "center", bold: true }, { text: "0.313", align: "center", bold: true }],
  ["공개 KURE + RRF + 필터", "baseline + 규칙", { text: "0.705", align: "center" }, { text: "0.299", align: "center" }],
  [{ text: "fine-tuning + RRF + 필터", bold: true }, "fine-tuning + 규칙", { text: "0.714", align: "center", bold: true }, { text: "0.455", align: "center", bold: true }],
  [{ text: "  + KLUE 재정렬기 학습", fill: "FFFFFF" }, "fine-tuning", { text: "—", align: "center" }, { text: "0.575", align: "center" }],
  [{ text: "  + bge 재정렬기 학습 (최종)", bold: true, fill: O.fill }, { text: "fine-tuning", fill: O.fill }, { text: "0.757", align: "center", bold: true, fill: O.fill }, { text: "0.657", align: "center", bold: true, fill: O.fill }],
], { x: 0.55, y: 1.6, w: 7.3, colW: [2.9, 1.8, 1.3, 1.3], fs: 11.5, rowH: 0.43 });
T(s, "원문 코퍼스 34,427청크 · FinDartBench test 1,947문항 · 자체 평가셋 v1 test 134문항(첫 사용) · seed 42 (43·44도 같은 결론)", { x: 0.55, y: 5.6, w: 7.3, h: 0.45, fontSize: 9.5, color: K.sub });
card(s, 8.1, 1.6, 4.68, 2.75); T(s, "같은 문항 짝지은 비교 (fine-tuning − 공개, R@5)", { x: 8.32, y: 1.75, w: 4.3, h: 0.3, fontSize: 12, bold: true, color: K.ink });
T(s, [
  { text: "FinDartBench 전체  ", options: { color: K.sub } }, { text: "+1.8%p [−2.2, +5.9]  유의하지 않음", options: { bold: true, breakLine: true } },
  { text: "   연도 있는 질문  ", options: { color: K.sub } }, { text: "+8.6%p [+2.4, +15.1]", options: { bold: true, color: K.green, breakLine: true } },
  { text: "   연도 없는 질문  ", options: { color: K.sub } }, { text: "−1.3%p [−5.8, +3.1]", options: { bold: true, color: K.red, breakLine: true } },
  { text: " ", options: { fontSize: 6, breakLine: true } },
  { text: "자체 평가셋 (회사·연도 항상 있음)  ", options: { color: K.sub } }, { text: "+15.7%p [+9.0, +22.4]", options: { bold: true, color: K.green, breakLine: true } },
  { text: "KLUE 재정렬기 학습 (FDB, seed 3개)  ", options: { color: K.sub } }, { text: "+4.2%p, 모두 유의", options: { bold: true, color: K.green } },
], { x: 8.32, y: 2.15, w: 4.3, h: 2.1, fontSize: 11.5, lineSpacingMultiple: 1.3 });
keynum(s, 8.1, 4.5, 4.68, 1.55, "+15.7%p", "자체 평가셋 R@5 (RRF + 필터 기준)");
bottomNote(s, "읽는 법", "fine-tuning 이득은 질문에 회사·연도가 있을 때 생긴다. FinDartBench는 연도 없는 질문이 69%라 평균에서 묻힌다");
s.addNotes("[1:20] 결과표 하나로 보여 드립니다. 왼쪽 열은 공개 벤치마크, 오른쪽 열은 자체 평가셋이고 둘 다 test입니다. 형태소 분석만으로 0.445에서 0.615로 크게 오릅니다. 공개 KURE와 fine-tuning을 같은 문항으로 짝지어 비교하면, 공개 벤치마크 전체로는 1.8%p로 신뢰구간이 0을 포함해 유의하지 않습니다. 하지만 연도가 있는 질문만 보면 8.6%p로 유의하고, 연도가 없는 질문은 차이가 없습니다. 모든 질문에 회사·연도가 있는 자체 평가셋에서는 15.7%p로 유의합니다. 즉 이득은 회사·연도가 질문에 있을 때 생깁니다. 직접 학습한 재정렬기는 seed 세 개 모두 유의하게 올립니다.");

// ---------- 7. 수치 검증 ----------
s = pres.addSlide({ masterName: "CONTENT" }); header(s, 7, "NUMERIC", "수치 검증: LLM 단독 / RAG / RAG + OpenDART", "대표 실험 2");
card(s, 0.55, 1.6, 7.3, 4.4, "FFFFFF");
s.addChart(pres.charts.BAR, [{ name: "test 정답률", labels: ["LLM 단독", "RAG (top-5)", "RAG + OpenDART"], values: [0.0, 4.2, 95.8] }], {
  x: 0.75, y: 1.75, w: 6.9, h: 4.1, barDir: "col", chartColors: ["B5AEA4", "6B655E", "E8622C"], varyColors: true,
  showValue: true, dataLabelPosition: "outEnd", dataLabelFormatCode: '0.0"%"', dataLabelFontFace: "+mn-lt", dataLabelFontSize: 13, dataLabelColor: "1F1C19",
  valAxisMaxVal: 100, valAxisMinVal: 0, valAxisLabelFormatCode: '0"%"', valAxisLabelFontFace: "+mn-lt", catAxisLabelFontFace: "+mn-lt", catAxisLabelFontSize: 12, valAxisLabelFontSize: 10,
  valGridLine: { color: "E6E2DC", size: 0.5 }, catGridLine: { style: "none" }, showLegend: false,
  showTitle: true, title: "수치 정답률 (공식 값 ±0.5%), 자체 평가셋 test 수치 96문항", titleFontFace: "+mn-lt", titleFontSize: 12, titleColor: "1F1C19",
});
card(s, 8.1, 1.6, 4.68, 2.75); T(s, "왜 RAG가 틀리나 (RAG가 답한 53개)", { x: 8.32, y: 1.75, w: 4.3, h: 0.3, fontSize: 12, bold: true, color: K.ink });
T(s, [
  { text: "정답 4 · 자릿수만 틀림 24 · 숫자 자체가 틀림 25", options: { bold: true, breakLine: true } },
  { text: "표의 '(단위: 백만원)'을 반영하지 않거나, 증감·차이 계산을 틀린다. 계산이 필요한 기업 비교는 단위를 무시해도 0/36", options: { color: K.sub, breakLine: true } },
  { text: " ", options: { fontSize: 6, breakLine: true } },
  { text: "근거 없는 금액 비율  RAG 40.4% → 0%", options: { bold: true, color: K.green, breakLine: true } },
  { text: "dev 26문항: 0% / 19.2% / 100%", options: { color: K.sub } },
], { x: 8.32, y: 2.15, w: 4.3, h: 2.1, fontSize: 11.5, lineSpacingMultiple: 1.2 });
card(s, 8.1, 4.5, 4.68, 1.5); T(s, "주의", { x: 8.32, y: 4.62, w: 4.3, h: 0.3, fontSize: 12, bold: true, color: K.red });
T(s, "문항이 OpenDART 6개 계정 템플릿이라 조회기에 유리한 조건. 95.8%는 LLM 실력이 아니라 조회 + Python 계산의 정확도", { x: 8.32, y: 4.95, w: 4.3, h: 1.0, fontSize: 11, color: K.ink, lineSpacingMultiple: 1.15 });
bottomNote(s, "설계", "LLM은 조회 계획(회사·연도·계정·연산)과 설명만, 숫자는 OpenDART 값 + Python 계산. 답 없는 질문 13개는 세 방식 모두 거절");
s.addNotes("[0:50] 두 번째 대표 실험인 수치 검증입니다. 같은 7.8B LLM으로 질문만 주면 0%, 검색한 문단을 주면 4.2%, LLM이 조회 계획만 세우고 숫자는 OpenDART와 Python이 계산하면 95.8%입니다. RAG가 틀리는 이유는 절반이 백만원 같은 표 단위를 반영하지 않은 것이고, 절반은 계산이나 다른 칸을 읽은 것입니다. 다만 문항이 OpenDART가 다루는 6개 계정이라 유리한 조건이고, 95.8%는 LLM 실력이 아니라 조회기의 정확도입니다.");

// ---------- 8. 오류 분석 ----------
s = pres.addSlide({ masterName: "CONTENT" }); header(s, 8, "ERROR ANALYSIS", "오류 분석: 틀린 30건", "실패 469건 중 30건");
table(s, [
  ["유형", "건수", "프로포절 때"],
  ["회사 없는 질문 → 다른 회사", { text: "11", align: "center", bold: true }, { text: "6", align: "center" }],
  ["연도 없는 질문 → 같은 회사 다른 사업연도", { text: "8", align: "center", bold: true }, { text: "9", align: "center" }],
  ["같은 보고서의 인접·유사 청크", { text: "5", align: "center" }, { text: "—", align: "center" }],
  ["정답 라벨 불완전 (1위도 답을 담음)", { text: "4", align: "center" }, { text: "3", align: "center" }],
  ["모호 (여러 연도를 묻는 질문)", { text: "2", align: "center" }, { text: "—", align: "center" }],
], { x: 0.55, y: 1.6, w: 6.2, colW: [4.2, 1.0, 1.0], fs: 12, rowH: 0.48 });
T(s, "최종 시스템, FinDartBench test에서 정답이 top-5 밖인 469건 중 seed 42로 30건", { x: 0.55, y: 4.6, w: 6.2, h: 0.3, fontSize: 9.5, color: K.sub });
[["회사 없는 질문", "\"이자율 100bp 변동 시 법인세비용차감전순손익 영향은?\"", "모든 회사 주석에 같은 문구 → 고를 근거가 없음"],
 ["연도 없는 질문", "\"향후 투자계획에서 2025년 예상투자액 합계는?\"", "'2025년'은 계획 연도라 필터가 못 잡고, FY2023 보고서의 같은 표가 1위"]]
  .forEach((e, i) => {
    card(s, 7.0, 1.6 + i * 1.5, 5.78, 1.35, "FFFFFF");
    T(s, e[0], { x: 7.22, y: 1.7 + i * 1.5, w: 5.4, h: 0.28, fontSize: 11, bold: true, color: K.org });
    T(s, e[1], { x: 7.22, y: 2.0 + i * 1.5, w: 5.4, h: 0.4, fontSize: 12, bold: true, color: K.ink });
    T(s, "→ " + e[2], { x: 7.22, y: 2.42 + i * 1.5, w: 5.4, h: 0.45, fontSize: 11, color: K.sub });
  });
keynum(s, 7.0, 4.65, 5.78, 1.4, "19 / 30", "질문에 회사나 연도가 없어서 틀린 경우");
bottomNote(s, "다음 실험", "회사·연도가 없으면 '최신 보고서 기준 + 되묻기', 라벨 불완전 6건은 정답 청크를 여러 개 허용");
s.addNotes("[0:50] 최종 시스템이 틀린 469건 중 30건을 읽고 분류했습니다. 가장 많은 것은 회사가 없는 질문 11건, 연도가 없는 질문 8건입니다. 예를 들어 '이자율 100bp 변동 시 영향은?'은 모든 회사 주석에 같은 문구가 있어서 어느 회사인지 고를 근거가 없습니다. 프로포절 때 1위였던 다른 사업연도 혼동은 필터로 줄었고, 이제 남은 실패는 대부분 질문 자체에 회사나 연도가 없는 경우입니다.");

// ---------- 9. 배운 점 · 바뀐 계획 ----------
s = pres.addSlide({ masterName: "CONTENT" }); header(s, 9, "LESSONS", "배운 점과 바뀐 계획", "결과 → 결정");
[["fine-tuning 이득은 질문에 회사·연도가 있을 때만", "연도 있음 +8.6%p / 연도 없음 −1.3%p. 학습 질문 45%를 연도 생략형으로 바꿔도 그대로", "연도 없는 질문은 학습이 아니라 '최신 연도 + 되묻기' 규칙으로"],
 ["재정렬기도 우리 쌍으로 학습해야 효과", "학습 없는 bge 0.455 → 0.455, 우리 쌍으로 학습 0.657 (자체 test R@5)", "학습한 재정렬기를 두 번째 직접 학습 모델로 유지 (plan B)"],
 ["LLM은 표 숫자를 그대로 못 쓴다", "RAG 정답률 4.2%, 절반이 단위 문제", "숫자는 OpenDART + Python, LLM은 설명만"]]
  .forEach((r, i) => {
    const y = 1.6 + i * 1.12;
    card(s, 0.55, y, 8.7, 0.98, "FFFFFF");
    T(s, r[0], { x: 0.8, y: y + 0.1, w: 4.6, h: 0.35, fontSize: 13, bold: true, color: K.ink });
    T(s, r[1], { x: 0.8, y: y + 0.47, w: 4.6, h: 0.45, fontSize: 10.5, color: K.sub });
    s.addShape(pres.shapes.RIGHT_ARROW, { x: 5.5, y: y + 0.35, w: 0.35, h: 0.3, fill: { color: K.org }, line: { color: K.org } });
    T(s, r[2], { x: 6.0, y: y, w: 3.1, h: 0.98, fontSize: 12, bold: true, color: K.org, valign: "middle" });
  });
card(s, 0.55, 5.0, 8.7, 1.05); T(s, "공개 사항", { x: 0.8, y: 5.1, w: 3, h: 0.3, fontSize: 12, bold: true, color: K.red });
T(s, "자체 평가셋 test를 3번 썼다(첫 사용 결과만 핵심 숫자로). 그 뒤 고친 v2(질문 분해, R@5 0.888)는 test에서 본 약점을 고친 것이라 참고용. 최종 평가는 팀이 쓴 처음 보는 문항으로 한다.", { x: 0.8, y: 5.4, w: 8.3, h: 0.6, fontSize: 11, color: K.ink });
keynum(s, 9.5, 1.6, 3.28, 4.45, "+8.6 vs −1.3", "연도 있는 질문 vs 연도 없는 질문 (%p)\nfine-tuning − 공개, FinDartBench test dense R@5");
s.addNotes("[0:40] 결과에서 배운 것 세 가지와 그래서 바꾸는 것입니다. 첫째, fine-tuning 이득은 질문에 회사와 연도가 있을 때만 생깁니다. 연도가 없는 질문은 학습으로 풀 문제가 아니라 최신 연도로 답하고 되묻는 규칙으로 처리합니다. 둘째, 재정렬기는 학습하지 않으면 효과가 없고 우리 쌍으로 학습해야 오릅니다. 그래서 재정렬기를 두 번째 직접 학습 모델로 둡니다. 셋째, LLM은 표 숫자를 그대로 못 써서 숫자는 OpenDART로 정합니다. 그리고 자체 평가셋 test를 여러 번 썼기 때문에, 최종 평가는 팀이 새로 쓴 처음 보는 문항으로 하겠습니다.");

// ---------- 10. 남은 계획 ----------
s = pres.addSlide({ masterName: "CONTENT" }); header(s, 10, "PLAN", "남은 계획 (13–16주차)", "역할 · plan B");
table(s, [
  ["주차", "할 일", "담당"],
  ["13주 (11/23–)", "원문을 안 보고 쓴 사람 문항 60개 → 30문항 교차 검수 κ ≥ 0.6 → 새 test로 고정", "백윤우 · 공동"],
  ["13주", "학습 쌍 검수 결과로 생성 필터 보강, 필요하면 재학습 (dev로만)", "임병건"],
  ["14주 (11/30–)", "수치 질문의 재무제표 섹션 우선, 연도 없는 질문 '최신 연도 + 되묻기' (dev 비교)", "임병건"],
  ["14주", "조회 계획에 기수·별칭 규칙, 계정 확대, 서술형 답변 사람 평가 30문항", "박채영"],
  ["15주 (12/7–)", "새 test 1회로 최종 평가 · 오류 분석 재실시 · 데모 · 최종 발표", "공동"],
  ["16주", "최종 보고서, 코드·split·config 제출 (제출 코드로 숫자 재현)", "공동"],
], { x: 0.55, y: 1.6, w: 8.7, colW: [1.5, 5.7, 1.5], fs: 11, rowH: 0.55 });
card(s, 9.5, 1.6, 3.28, 2.1); T(s, "역할", { x: 9.72, y: 1.75, w: 2.9, h: 0.3, fontSize: 12, bold: true, color: K.ink });
T(s, [{ text: "임병건 ", options: { bold: true } }, { text: "학습 쌍 · 모델 학습 · 서버", options: { breakLine: true } }, { text: "백윤우 ", options: { bold: true } }, { text: "평가셋 · 오류 분석 · 발표", options: { breakLine: true } }, { text: "박채영 ", options: { bold: true } }, { text: "수치 검증 · 생성 · 발표 자료", options: {} }], { x: 9.72, y: 2.1, w: 2.95, h: 1.5, fontSize: 11.5, color: K.ink, lineSpacingMultiple: 1.35 });
s.addShape(pres.shapes.ROUNDED_RECTANGLE, { x: 9.5, y: 3.9, w: 3.28, h: 2.15, rectRadius: 0.12, fill: { color: K.dark }, line: { color: K.dark } });
T(s, "plan B", { x: 9.72, y: 4.05, w: 2.9, h: 0.3, fontSize: 12, bold: true, color: K.org });
T(s, "사람 문항 test에서도 임베딩 이득이 유의하지 않으면, 이득이 확인된 학습한 재정렬기를 대표 직접 학습 모델로 하고 RAG 시스템에 집중", { x: 9.72, y: 4.4, w: 2.9, h: 1.6, fontSize: 11, color: "F2F0EC", lineSpacingMultiple: 1.15 });
bottomNote(s, "목표", "처음 보는 사람 문항에서 fine-tuning vs 공개 모델의 차이를 다시 확인한다");
s.addNotes("[0:30] 남은 4주 계획입니다. 13주차에 원문을 보지 않고 쓴 사람 문항 60개로 처음 보는 test를 만들고, 15주차에 그 test로 한 번만 최종 평가합니다. 사람 문항에서도 임베딩 이득이 유의하지 않으면, 이득이 확인된 재정렬기를 대표 모델로 하고 시스템 쪽에 집중합니다. 감사합니다.");

// ---------- Backups ----------
function backup(title, fn, notes) {
  const b = pres.addSlide({ masterName: "CONTENT" });
  b.addShape(pres.shapes.RECTANGLE, { x: 0.55, y: 0.42, w: 0.11, h: 0.11, fill: { color: "A8A29A" }, line: { color: "A8A29A" } });
  T(b, "APPENDIX", { x: 0.75, y: 0.36, w: 6, h: 0.25, fontSize: 10, color: K.mute, charSpacing: 4 });
  T(b, title, { x: 0.55, y: 0.62, w: 12, h: 0.75, fontSize: 26, bold: true, color: K.ink });
  fn(b); if (notes) b.addNotes(notes);
}
backup("[부록] 예상 질문", (b) => table(b, [
  ["질문", "답"],
  ["baseline과 정말 같은 데이터·split인가?", "모든 모델이 같은 34,427청크, 같은 test 문항. 학습 쌍 4,646개도 모든 학습 모델에 동일"],
  ["숫자는 dev인가 test인가? 몇 개?", "선택은 dev(FinDartBench 672, 자체 43). 보고 숫자는 test 1,947 / 134(첫 사용). dev 값은 보고서 표 1에 같이"],
  ["두 모델 차이가 우연 범위를 넘나?", "FinDartBench 전체 +1.8%p는 CI가 0 포함 → 판단 불가. 자체 +15.7%p, 연도 있는 질문 +8.6%p는 CI가 0 위"],
  ["가장 많이 틀리는 유형과 이유는?", "회사·연도가 없는 질문 19/30. 질문에 단서가 없어 같은 문구의 다른 회사·연도 문단과 구분 불가"],
  ["라벨은 믿을 만한가? 일치도는?", `학습 쌍: ${pairsTxt}. 평가셋: ${evalTxt}. 변형 문항은 따로 점수 보고`],
  ["최종까지 무엇을 하면 결론이 나오나?", "처음 보는 사람 문항 test 1회. 안 되면 재정렬기를 대표 모델로 (plan B)"],
  ["왜 KLUE-BERT가 아니라 KURE?", "같은 쌍으로 학습해도 KLUE-BERT dense R@5 0.27 < KURE 0.31 (학습 전 0.04). 재정렬기는 KLUE-BERT도 학습"],
], { x: 0.55, y: 1.6, w: 12.23, colW: [3.8, 8.43], fs: 11.5, rowH: 0.6 }));
backup("[부록] v2 개선과 test 사용 이력", (b) => {
  table(b, [
    ["시스템", "자체 R@5", "기업 비교 R@5", "FDB R@1", "FDB R@5"],
    ["v1 최종 (fine-tuning + RRF + 필터 + bge 학습)", "0.657", "0.200", "0.399", "0.757"],
    ["v2 (+ 두 회사 질문 분해, 후보 40, 연도 규칙)", "0.888", "0.825", "0.592", "0.818"],
  ], { x: 0.55, y: 1.6, w: 12.23, colW: [5.6, 1.6, 1.8, 1.6, 1.63], fs: 12, rowH: 0.5 });
  table(b, [
    ["평가셋", "사용", "내용", "선택에 썼나"],
    ["자체 v1 test", "1회 (11/18)", "핵심 숫자 (+15.7%p)", "아니오, dev 43으로 선택"],
    ["", "2회 (11/19)", "v2: 1회 test에서 본 약점(기업 비교)을 고친 뒤 측정 → 낙관적", "아니오"],
    ["", "3회 (11/19)", "KLUE-BERT 비교 (분석)", "아니오"],
  ], { x: 0.55, y: 3.4, w: 12.23, colW: [2.0, 1.8, 6.0, 2.43], fs: 12, rowH: 0.5 });
  T(b, "FinDartBench R@1 0.399 → 0.592는 대부분 '연도 없으면 최신 연도' 규칙 덕분이고, dev의 연도 없는 질문 96%의 정답이 2024 보고서라는 데이터셋 성질에 기댄다.", { x: 0.55, y: 5.6, w: 12.23, h: 0.5, fontSize: 11, color: K.sub });
});
backup("[부록] KLUE-BERT 비교와 hard negative 구성 (dev)", (b) => {
  table(b, [
    ["모델 (R@5)", "dense 자체", "dense FDB"],
    ["klue/bert-base 학습 없음", "0.037", "0.185"],
    ["klue/bert-base fine-tuning", "0.269", "0.449"],
    ["KURE-v1 공개", "0.201", "0.600"],
    ["KURE-v1 fine-tuning", "0.313", "0.614"],
  ], { x: 0.55, y: 1.6, w: 5.9, colW: [3.3, 1.3, 1.3], fs: 12, rowH: 0.5 });
  table(b, [
    ["negative 구성 (FDB dev 672)", "dense R@5", "dense MRR"],
    ["공개 KURE-v1", "0.668", "0.491"],
    ["in-batch만", "0.686", "0.484"],
    ["무작위 다른 회사", "0.696", "0.481"],
    ["같은 회사·다른 사업연도 hard (선택)", "0.716", "0.494"],
    ["hard 절반", "0.696", "0.491"],
    ["다른 섹션만", "0.680", "0.484"],
  ], { x: 6.85, y: 1.6, w: 5.93, colW: [3.5, 1.2, 1.23], fs: 12, rowH: 0.5 });
  T(b, "KLUE-BERT는 학습 없이는 검색에 거의 못 쓰지만 학습하면 7배. KLUE-BERT 재정렬기는 재정렬 없음 대비 자체 test R@5 +30.6%p. 자체 test 3번째 사용(분석).", { x: 0.55, y: 5.6, w: 12.23, h: 0.5, fontSize: 11, color: K.sub });
});

(async () => {
  const out = process.argv[2] || "slides.pptx";
  await pres.writeFile({ fileName: out });
  await applyTheme(out, THEME);
  console.log("ok");
})();

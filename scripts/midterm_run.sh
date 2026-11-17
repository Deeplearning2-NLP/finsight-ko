#!/usr/bin/env bash
# Midterm pipeline (M1 -> M3 -> M6 -> dev selection -> test once). Resumable: every finished step leaves
# outputs/midterm/.done_<step>; re-running skips those. Run inside tmux so a VPN drop does not stop it:
#   tmux new -d -s midterm 'bash scripts/midterm_run.sh'      # progress: outputs/midterm/logs/pipeline.log
set -uo pipefail
ROOT=/mnt/urp_jwl_shared/bk/finsight-ko
cd "$ROOT"
source .venv/bin/activate
export HF_HOME=/mnt/urp_jwl_shared/bk/hf_cache TOKENIZERS_PARALLELISM=false
M=outputs/midterm
LOG=$M/logs
mkdir -p "$LOG"
VLLM=/mnt/urp_jwl_shared/bk/vllm-venv/bin/python
say() { echo "[$(date -u +%F' '%T)] $*" | tee -a "$LOG/pipeline.log"; }

wait_gpu() {  # wait until GPU $1 has < 2 GB in use (do not collide with other users' jobs)
  while true; do
    used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i "$1" | tr -d ' ')
    [ "${used:-99999}" -lt 2000 ] && return 0
    sleep 60
  done
}

step() {  # step <name> <gpu|-> <command...>
  local name=$1 gpu=$2; shift 2
  if [ -f "$M/.done_$name" ]; then say "skip $name (done)"; return 0; fi
  [ "$gpu" != "-" ] && wait_gpu "$gpu"
  say "start $name (gpu $gpu)"
  if CUDA_VISIBLE_DEVICES=${gpu/-/} "$@" > "$LOG/$name.log" 2>&1; then
    touch "$M/.done_$name"; say "done  $name"
  else
    say "FAIL  $name (see $LOG/$name.log)"; return 1
  fi
}

train_and_encode() {  # <gpu> <neg>
  step "train_$2" "$1" python scripts/train_embedding_m.py --neg "$2" &&
  step "encode_$2" "$1" python scripts/eval_m.py --encode "$M/kure-m-$2" "emb_m_$2.npy"
}

# ---- M1: format-diversified training pairs
step m1_chunks - python scripts/m1_chunks.py || exit 1
step m1_generate 0 env VLLM_USE_FLASHINFER_SAMPLER=0 $VLLM scripts/pairs_pilot_generate.py \
     --in m1_source_chunks.jsonl --prefix m1 || exit 1
step m1_build - python scripts/m1_build.py || exit 1

# ---- M3 (negative mixes) + M6 (rerankers) on four GPUs
( train_and_encode 0 none;  train_and_encode 0 hard50 ) &
( train_and_encode 1 easy;  train_and_encode 1 section ) &
( train_and_encode 2 hard;  step rr_klue 2 python scripts/train_reranker_m.py --base klue/roberta-base --tag klue ) &
( step rr_bge 3 python scripts/train_reranker_m.py --base BAAI/bge-reranker-v2-m3 --tag bge ) &
wait

# ---- dev selection (test is not touched here)
MODELS_DEV=("KURE-v1 공개=nlpai-lab/KURE-v1=emb_kure_base.npy"
            "S1 (a) 연도 포함 질문만=outputs/proposal/kure-s1-a=emb_s1_a.npy")
for n in none easy hard hard50 section; do
  [ -f "$M/.done_encode_$n" ] && MODELS_DEV+=("M3 $n=$M/kure-m-$n=emb_m_$n.npy")
done
step dev_embed 0 python scripts/eval_m.py --split dev --tag embed --models "${MODELS_DEV[@]}" || exit 1
step select_embed - python - <<'EOF' || exit 1
import json
d = json.load(open("outputs/midterm/eval_dev_embed.json"))["results"]
cands = {k.split(" | ")[0]: v["MRR@10"] for k, v in d.items() if k.startswith("M3 ") and k.endswith(" | dense")}
best = max(cands, key=cands.get)
json.dump({"criterion": "dev dense MRR@10", "dev_dense_mrr": cands, "best_embedding": best},
          open("outputs/midterm/selection_embed.json", "w"), ensure_ascii=False, indent=1)
print(best, cands)
EOF
BEST=$(python -c "import json;print(json.load(open('$M/selection_embed.json'))['best_embedding'].split(' ',1)[1])")
say "best embedding variant (dev): $BEST"
RR=("bge-zero=BAAI/bge-reranker-v2-m3")
[ -f "$M/.done_rr_klue" ] && RR+=("klue-ft=$M/reranker-klue")
[ -f "$M/.done_rr_bge" ] && RR+=("bge-ft=$M/reranker-bge")
step dev_rerank 0 python scripts/eval_m.py --split dev --tag rerank \
     --models "M3 $BEST=$M/kure-m-$BEST=emb_m_$BEST.npy" --reranker "${RR[@]}" || exit 1
step select_rerank - python - <<'EOF' || exit 1
import json
d = json.load(open("outputs/midterm/eval_dev_rerank.json"))["results"]
c = {k.rsplit("+", 1)[1]: v["MRR@10"] for k, v in d.items() if "RRF+filter+" in k}
base = [v["MRR@10"] for k, v in d.items() if k.endswith(" | RRF+filter")][0]
json.dump({"criterion": "dev MRR@10 of RRF+filter+reranker", "dev_mrr": c, "dev_mrr_no_reranker": base,
           "best_reranker": max(c, key=c.get)}, open("outputs/midterm/selection_rerank.json", "w"), ensure_ascii=False, indent=1)
print(c, base)
EOF

# ---- test, once (all configurations fixed above on dev)
step test_final 0 python scripts/eval_m.py --split test --tag final \
     --models "KURE-v1 공개=nlpai-lab/KURE-v1=emb_kure_base.npy" "M3 $BEST=$M/kure-m-$BEST=emb_m_$BEST.npy" \
              "S1 (a) 연도 포함 질문만=outputs/proposal/kure-s1-a=emb_s1_a.npy" \
     --reranker "${RR[@]}" || exit 1
say "PIPELINE FINISHED"

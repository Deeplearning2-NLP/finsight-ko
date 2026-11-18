#!/usr/bin/env bash
# Seed robustness (analysis only, after the final test; nothing is re-selected from it):
# M3 hard / M3 none embeddings and the KLUE reranker with seeds 43, 44 (seed 42 = the reported runs).
# Resumable: outputs/midterm/.done_seed_<step>. Log: outputs/midterm/logs/seed_pipeline.log
set -u
cd "$(dirname "$0")/.." && source .venv/bin/activate
export HF_HOME=/mnt/urp_jwl_shared/bk/hf_cache
M=outputs/midterm; L=$M/logs; mkdir -p $L
log(){ echo "[$(date '+%F %T')] $*" | tee -a $L/seed_pipeline.log; }
step(){ local name=$1 gpu=$2; shift 2
  [ -f $M/.done_seed_$name ] && { log "skip $name"; return 0; }
  log "start $name (gpu $gpu)"
  if CUDA_VISIBLE_DEVICES=$gpu "$@" > $L/seed_$name.log 2>&1; then touch $M/.done_seed_$name; log "done  $name"
  else log "FAIL  $name (see $L/seed_$name.log)"; return 1; fi; }
emb(){ local neg=$1 s=$2 gpu=$3
  step train_${neg}_s$s $gpu python scripts/train_embedding_m.py --neg $neg --seed $s --tag ${neg}_s$s && \
  step encode_${neg}_s$s $gpu python scripts/eval_m.py --encode $M/kure-m-${neg}_s$s emb_m_${neg}_s$s.npy; }
( emb hard 43 0; emb hard 44 0 ) &
( emb none 43 1; emb none 44 1 ) &
( step rr_klue_s43 2 python scripts/train_reranker_m.py --base klue/roberta-base --tag klue_s43 --seed 43; \
  step rr_klue_s44 2 python scripts/train_reranker_m.py --base klue/roberta-base --tag klue_s44 --seed 44 ) &
wait
MODELS=("KURE-v1 공개=nlpai-lab/KURE-v1=emb_kure_base.npy")
for n in hard none; do
  MODELS+=("M3 ${n} s42=$M/kure-m-${n}=emb_m_${n}.npy")
  for s in 43 44; do MODELS+=("M3 ${n} s$s=$M/kure-m-${n}_s$s=emb_m_${n}_s$s.npy"); done
done
for sp in dev test; do
  step eval_emb_$sp 0 python scripts/eval_m.py --split $sp --tag seeds_emb --models "${MODELS[@]}" &
  step eval_rr_$sp 1 python scripts/eval_m.py --split $sp --tag seeds_rr --models "KURE-v1 공개=nlpai-lab/KURE-v1=emb_kure_base.npy" \
       --reranker klue-s42=$M/reranker-klue klue-s43=$M/reranker-klue_s43 klue-s44=$M/reranker-klue_s44 &
  wait
done
log "SEED PIPELINE FINISHED"

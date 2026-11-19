#!/bin/bash
# KLUE-BERT (klue/bert-base) as (1) bi-encoder trained on the same M1 pairs (hard negatives, like M3 hard) and
# (2) cross-encoder reranker. Resumable (.done_kb_*). Log: outputs/midterm/logs/klue_bert.log
cd /mnt/urp_jwl_shared/bk/finsight-ko
export HF_HOME=/mnt/urp_jwl_shared/bk/hf_cache TOKENIZERS_PARALLELISM=false
L=outputs/midterm/logs/klue_bert.log; D=outputs/midterm/logs
log(){ echo "[$(date '+%F %T')] $*" | tee -a $L; }
step(){ n=$1; g=$2; shift 2; [ -f $D/.done_kb_$n ] && { log "skip $n"; return; }; log "start $n"; CUDA_VISIBLE_DEVICES=$g "$@" >> $L 2>&1 && touch $D/.done_kb_$n && log "done $n" || { log "FAIL $n"; return 1; }; }
( step bi 1 .venv/bin/python scripts/train_embedding_m.py --neg hard --base klue/bert-base --tag kluebert_hard && \
  step enc 1 .venv/bin/python scripts/eval_m.py --encode outputs/midterm/kure-m-kluebert_hard emb_m_kluebert_hard.npy ) &
step rr 2 .venv/bin/python scripts/train_reranker_m.py --base klue/bert-base --tag kluebert &
wait; log "KLUE BERT FINISHED"

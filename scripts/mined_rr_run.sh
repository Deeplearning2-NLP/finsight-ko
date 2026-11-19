#!/bin/bash
cd /mnt/urp_jwl_shared/bk/finsight-ko
L=outputs/midterm/logs/mined_rr.log; D=outputs/midterm/logs
log(){ echo "[$(date '+%F %T')] $*" | tee -a $L; }
step(){ n=$1; g=$2; shift 2; [ -f $D/.done_mr_$n ] && { log "skip $n"; return; }; log "start $n"; CUDA_VISIBLE_DEVICES=$g "$@" >> $L 2>&1 && touch $D/.done_mr_$n && log "done $n" || { log "FAIL $n"; exit 1; }; }
step mine 1 .venv/bin/python scripts/mine_negatives_m.py
step bge 1 .venv/bin/python scripts/train_reranker_m.py --base BAAI/bge-reranker-v2-m3 --tag bge_mined --mined data/train/pairs_m1_mined.jsonl &
step klue 2 .venv/bin/python scripts/train_reranker_m.py --base klue/roberta-base --tag klue_mined --mined data/train/pairs_m1_mined.jsonl &
wait; log "MINED RR FINISHED"

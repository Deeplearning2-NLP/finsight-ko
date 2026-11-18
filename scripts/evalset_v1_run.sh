#!/bin/bash
# evalset v1: dev selection -> test once. Resumable via .done_* markers. Log: outputs/midterm/logs/evalset_v1.log
cd /mnt/urp_jwl_shared/bk/finsight-ko
L=outputs/midterm/logs/evalset_v1.log; D=outputs/midterm/logs
export CUDA_VISIBLE_DEVICES=${GPU:-0}
log(){ echo "[$(date '+%F %T')] $*" | tee -a $L; }
step(){ n=$1; shift; [ -f $D/.done_ev1_$n ] && { log "skip $n"; return; }; log "start $n"; "$@" >> $L 2>&1 && touch $D/.done_ev1_$n && log "done $n" || { log "FAIL $n"; exit 1; }; }
step dev  .venv/bin/python scripts/eval_evalset.py --file data/eval/evalset_v1.jsonl --split dev --tag v1_dev
step sel  .venv/bin/python scripts/select_evalset_dev.py
step test .venv/bin/python scripts/eval_evalset.py --file data/eval/evalset_v1.jsonl --split test --tag v1_test
log "EVALSET V1 FINISHED"

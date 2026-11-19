#!/bin/bash
# v2: wait for mined rerankers, dev stages (v1 A-D, FinDartBench E), then each test once. Resumable (.done_v2_*).
cd /mnt/urp_jwl_shared/bk/finsight-ko
L=outputs/midterm/logs/v2.log; D=outputs/midterm/logs
export CUDA_VISIBLE_DEVICES=${GPU:-0}
log(){ echo "[$(date '+%F %T')] $*" | tee -a $L; }
step(){ n=$1; shift; [ -f $D/.done_v2_$n ] && { log "skip $n"; return; }; log "start $n"; "$@" >> $L 2>&1 && touch $D/.done_v2_$n && log "done $n" || { log "FAIL $n"; exit 1; }; }
until grep -q "MINED RR FINISHED\|FAIL" $D/mined_rr.log 2>/dev/null; do sleep 30; done
grep -q FAIL $D/mined_rr.log && log "mined rerankers failed; continuing without them"
step v1dev  .venv/bin/python scripts/eval_v2.py --data v1  --split dev  --stages
step fdbdev .venv/bin/python scripts/eval_v2.py --data fdb --split dev  --stages
step v1test .venv/bin/python scripts/eval_v2.py --data v1  --split test --final
step fdbtest .venv/bin/python scripts/eval_v2.py --data fdb --split test --final
log "V2 FINISHED"

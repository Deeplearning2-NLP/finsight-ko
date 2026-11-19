#!/bin/bash
# G1 dev pipeline (no test). Resumable via .done_g1_*. Log: outputs/midterm/logs/g1.log
cd /mnt/urp_jwl_shared/bk/finsight-ko
export HF_HOME=/mnt/urp_jwl_shared/bk/hf_cache TOKENIZERS_PARALLELISM=false VLLM_USE_FLASHINFER_SAMPLER=0
L=outputs/midterm/logs/g1.log; D=outputs/midterm/logs; V=/mnt/urp_jwl_shared/bk/vllm-venv/bin/python
log(){ echo "[$(date '+%F %T')] $*" | tee -a $L; }
step(){ n=$1; shift; [ -f $D/.done_g1_$n ] && { log "skip $n"; return; }; log "start $n"; "$@" >> $L 2>&1 && touch $D/.done_g1_$n && log "done $n" || { log "FAIL $n"; exit 1; }; }
export CUDA_VISIBLE_DEVICES=${GPU:-0}
step enc_AB  .venv/bin/python scripts/unit_fix_encode.py AB
step enc_ABC .venv/bin/python scripts/unit_fix_encode.py ABC
step ret_orig .venv/bin/python scripts/g1_dev_eval.py
step ret_AB  env FINSIGHT_CORPUS=unitfix_AB  .venv/bin/python scripts/g1_dev_eval.py
step ret_ABC env FINSIGHT_CORPUS=unitfix_ABC .venv/bin/python scripts/g1_dev_eval.py
for T in AB ABC; do
  step ctx_$T env FINSIGHT_CORPUS=unitfix_$T .venv/bin/python scripts/numeric_qa.py contexts
  step gen_$T env FINSIGHT_CORPUS=unitfix_$T $V scripts/numeric_qa.py generate --split dev
  step grade_$T env FINSIGHT_CORPUS=unitfix_$T .venv/bin/python scripts/numeric_qa.py grade --split dev
done
log "G1 FINISHED"

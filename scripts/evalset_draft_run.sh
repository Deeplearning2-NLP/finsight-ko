#!/usr/bin/env bash
# Wait for the EEVE download and for any GPU with < 2 GB in use, then generate + build the eval-set draft.
cd /mnt/urp_jwl_shared/bk/finsight-ko
export HF_HOME=/mnt/urp_jwl_shared/bk/hf_cache VLLM_USE_FLASHINFER_SAMPLER=0
L=outputs/midterm/logs
while tmux has-session -t eeve_dl 2>/dev/null; do sleep 15; done
while true; do
  g=$(nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits | awk -F', ' '$2<2000{print $1; exit}')
  [ -n "$g" ] && break; sleep 60
done
echo "[$(date -u +%T)] evalset on GPU $g" >> $L/pipeline.log
CUDA_VISIBLE_DEVICES=$g /mnt/urp_jwl_shared/bk/vllm-venv/bin/python scripts/evalset_draft.py --stage generate > $L/evalset_generate.log 2>&1 &&
  source .venv/bin/activate && python scripts/evalset_draft.py --stage build > $L/evalset_build.log 2>&1
echo "[$(date -u +%T)] evalset finished (exit $?)" >> $L/pipeline.log

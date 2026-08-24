#!/bin/bash
# O3: does the augmentation anatomy survive on the strong (labs) baseline?
cd .
set -o pipefail
pick() { need=$1
  while true; do
    b=$(nvidia-smi --query-gpu=index,memory.free --format=csv,noheader,nounits | sort -t, -k2 -rn | head -1)
    i=${b%%,*}; f=${b##*, }; [ "$f" -ge "$need" ] && { echo $i; return; }; sleep 120
  done; }
R() { need=$1; shift
  for t in 1 2 3 4 5; do
    g=$(pick $need); CUDA_VISIBLE_DEVICES=$g "$@" 2>&1 | tee -a mimic/o3_full.log | grep -E "^\[seed|label injection|persona aug"
    [ ${PIPESTATUS[0]} -eq 0 ] && return 0; sleep 240
  done; return 1; }
T="--textemb mimic/out/note_emb.npy --textids mimic/out/note_emb_ids.parquet"
G="--dxemb mimic/out/dx_text_emb_gloss.npy --dxids mimic/out/dx_text_ids_gloss.parquet"
L="--labsfeat mimic/out/labs48.parquet"
B="--mode mtl $T $G $L --dx-mode strata --fixed-split"
echo "=== [1] bracket on the labs baseline (floor / ceiling) ==="
for s in 42 43 44 45 46; do
  R 10000 python3 -u mimic/train_mtl.py --seed $s $B --inject-labels mimic/out/inject_prior.parquet  --tag o3-labs-prior  || exit 1
  R 10000 python3 -u mimic/train_mtl.py --seed $s $B --inject-labels mimic/out/inject_oracle.parquet --tag o3-labs-oracle || exit 1
done
echo "=== [2] ColdLLM simulator (recalibrated) on the labs baseline ==="
for s in 42 43 44 45 46; do
  R 10000 python3 -u mimic/train_mtl.py --seed $s $B --inject-labels mimic/out/inject_sim_cal.parquet --tag o3-labs-sim || exit 1
done
echo "=== [3] input-channel agent evidence on the labs baseline ==="
for s in 42 43 44 45 46; do
  R 10000 python3 -u mimic/train_mtl.py --seed $s --mode mtl $T $G $L --dx-mode strata --fixed-split \
    --agentemb mimic/out/agent_emb_agent.npy --agentids mimic/out/agent_ids_agent.parquet --tag o3-labs-agent || exit 1
done
echo "=== [4] identity treatment on the labs baseline (dropout vs deletion) ==="
for s in 42 43 44 45 46; do
  R 10000 python3 -u mimic/train_mtl.py --seed $s --mode mtl $T $G $L --dx-mode input --dx-dropout 0.5 --fixed-split --tag o3-labs-dxdrop || exit 1
done
echo "=== O3 DONE ==="

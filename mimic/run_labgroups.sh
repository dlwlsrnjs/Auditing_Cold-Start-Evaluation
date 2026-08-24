#!/bin/bash
# Which physiological family carries the laboratory gain? If one group dominates, the result is
# fragile (and possibly a single near-terminal marker); if no group does, it is broad severity.
cd .
set -o pipefail
until grep -qE "BASELINES DONE" mimic/baselines.log 2>/dev/null; do sleep 300; done
pick() { need=$1
  while true; do
    b=$(nvidia-smi --query-gpu=index,memory.free --format=csv,noheader,nounits | sort -t, -k2 -rn | head -1)
    i=${b%%,*}; f=${b##*, }; [ "$f" -ge "$need" ] && { echo $i; return; }; sleep 120
  done; }
R() { need=$1; shift
  for t in 1 2 3 4 5; do
    g=$(pick $need); CUDA_VISIBLE_DEVICES=$g "$@" 2>&1 | tee -a mimic/labgroups_full.log | grep -E "^\[seed|lab group"
    [ ${PIPESTATUS[0]} -eq 0 ] && return 0; sleep 240
  done; return 1; }
T="--textemb mimic/out/note_emb.npy --textids mimic/out/note_emb_ids.parquet"
G="--dxemb mimic/out/dx_text_emb_gloss.npy --dxids mimic/out/dx_text_ids_gloss.parquet"
B="--mode mtl $T $G --dx-mode strata --fixed-split --labsfeat mimic/out/labs48.parquet"
for grp in renal heme coag hepatic perfusion; do
  for s in 42 43 44 45 46; do
    R 10000 python3 -u mimic/train_mtl.py --seed $s $B --lab-group $grp --tag lg-$grp || exit 1
  done
done
echo "=== LABGROUPS DONE ==="

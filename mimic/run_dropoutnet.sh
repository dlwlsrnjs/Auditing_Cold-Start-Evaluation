#!/bin/bash
cd .
set -o pipefail
pick() { need=$1
  while true; do
    best=$(nvidia-smi --query-gpu=index,memory.free --format=csv,noheader,nounits | sort -t, -k2 -rn | head -1)
    idx=${best%%,*}; free=${best##*, }
    [ "$free" -ge "$need" ] && { echo $idx; return; }; sleep 120
  done; }
R() { need=$1; shift
  for t in 1 2 3 4 5 6; do
    g=$(pick $need); CUDA_VISIBLE_DEVICES=$g "$@" 2>&1 | tee -a mimic/dropoutnet_full.log | grep -E "^\[seed"
    rc=${PIPESTATUS[0]}; [ $rc -eq 0 ] && return 0; sleep 300
  done; return 1; }
T="--textemb mimic/out/note_emb.npy --textids mimic/out/note_emb_ids.parquet"
G="--dxemb mimic/out/dx_text_emb_gloss.npy --dxids mimic/out/dx_text_ids_gloss.parquet"
# per-seed split, matching the RQ2 table; identity RETAINED but stochastically dropped
for p in 0.3 0.5 0.7; do
  for s in 42 43 44 45 46; do
    R 10000 python3 -u mimic/train_mtl.py --seed $s --mode mtl $T $G \
      --dx-mode input --dx-dropout $p --tag dxdrop${p} || { echo "dxdrop$p seed $s FAILED"; exit 1; }
  done
done
echo "=== DROPOUTNET DONE ==="

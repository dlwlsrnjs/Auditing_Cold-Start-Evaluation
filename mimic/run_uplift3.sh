#!/bin/bash
cd .
set -o pipefail
pick() {
  need=$1
  while true; do
    best=$(nvidia-smi --query-gpu=index,memory.free --format=csv,noheader,nounits | sort -t, -k2 -rn | head -1)
    idx=${best%%,*}; free=${best##*, }
    [ "$free" -ge "$need" ] && { echo $idx; return; }
    sleep 120
  done
}
R() {
  need=$1; shift
  for t in 1 2 3 4 5 6; do
    g=$(pick $need)
    CUDA_VISIBLE_DEVICES=$g "$@" 2>&1 | tee -a mimic/uplift3_full.log | grep -E "^\[seed|labs block|persona aug|\[saved\]"
    rc=${PIPESTATUS[0]}
    [ $rc -eq 0 ] && return 0
    echo "  [retry $t] exit=$rc" >> mimic/uplift3_full.log
    sleep 300
  done
  return 1
}
echo "=== [2] gte re-embedding ==="
R 12000 python3 -u mimic/embed_gte.py --which all || { echo "GTE EMBED FAILED"; exit 1; }
echo "=== [3] gte configs, 5 seeds ==="
T2="--textemb mimic/out/note_emb_gte.npy --textids mimic/out/note_emb_ids.parquet"
G2="--dxemb mimic/out/dx_text_emb_gte.npy --dxids mimic/out/dx_text_ids_gloss.parquet"
L="--labsfeat mimic/out/labs48.parquet"
P2="--personas mimic/out/personas_llm.parquet --personaemb mimic/out/persona_emb_gte.npy"
BASE="--mode mtl --dx-mode strata --fixed-split"
for s in 42 43 44 45 46; do
  R 10000 python3 -u mimic/train_mtl.py --seed $s $BASE $T2 $G2 $L --tag rm2-gte-labs || { echo "rm2-gte-labs seed $s FAILED"; exit 1; }
  R 10000 python3 -u mimic/train_mtl.py --seed $s $BASE $T2 $G2 $L $P2 --tag rm2-gte-labs-persona || { echo "rm2-gte-labs-persona seed $s FAILED"; exit 1; }
done
echo "=== UPLIFT3 DONE ==="

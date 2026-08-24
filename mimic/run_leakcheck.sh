#!/bin/bash
cd .
set -o pipefail
until grep -q "\[saved\] labs48_24h" mimic/build_labs24.log 2>/dev/null; do sleep 30; done
pick() { need=$1
  while true; do
    b=$(nvidia-smi --query-gpu=index,memory.free --format=csv,noheader,nounits | sort -t, -k2 -rn | head -1)
    i=${b%%,*}; f=${b##*, }; [ "$f" -ge "$need" ] && { echo $i; return; }; sleep 120
  done; }
R() { need=$1; shift
  for t in 1 2 3 4 5; do
    g=$(pick $need); CUDA_VISIBLE_DEVICES=$g "$@" 2>&1 | tee -a mimic/leakcheck_full.log | grep -E "^\[seed|exclude-early|labs block"
    [ ${PIPESTATUS[0]} -eq 0 ] && return 0; sleep 240
  done; return 1; }
T="--textemb mimic/out/note_emb.npy --textids mimic/out/note_emb_ids.parquet"
G="--dxemb mimic/out/dx_text_emb_gloss.npy --dxids mimic/out/dx_text_ids_gloss.parquet"
P="--personas mimic/out/personas_llm.parquet --personaemb mimic/out/persona_emb_llm.npy"
BASE="--mode mtl --dx-mode strata --fixed-split"
echo "=== (a) 24h lab window ==="
for s in 42 43 44 45 46; do
  R 10000 python3 -u mimic/train_mtl.py --seed $s $BASE $T $G --labsfeat mimic/out/labs48_24h.parquet --tag leak-labs24 || exit 1
  R 10000 python3 -u mimic/train_mtl.py --seed $s $BASE $T $G --labsfeat mimic/out/labs48_24h.parquet $P --tag leak-labs24-persona || exit 1
done
echo "=== (b) exclude deaths within 48h ==="
for s in 42 43 44 45 46; do
  R 10000 python3 -u mimic/train_mtl.py --seed $s $BASE $T $G --labsfeat mimic/out/labs48.parquet --exclude-early-deaths 48 --tag leak-excl48 || exit 1
  R 10000 python3 -u mimic/train_mtl.py --seed $s $BASE $T $G --labsfeat mimic/out/labs48.parquet --exclude-early-deaths 48 $P --tag leak-excl48-persona || exit 1
done
echo "=== LEAKCHECK DONE ==="

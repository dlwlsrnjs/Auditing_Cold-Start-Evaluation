#!/bin/bash
# Cold-start baseline comparators the reviewer asked for (O5):
#   ALDI-style warm->cold distillation, and a SMOTE-style non-LLM synthetic-row control.
cd .
set -o pipefail
until grep -qE "O3 DONE" mimic/o3.log 2>/dev/null; do sleep 300; done
pick() { need=$1
  while true; do
    b=$(nvidia-smi --query-gpu=index,memory.free --format=csv,noheader,nounits | sort -t, -k2 -rn | head -1)
    i=${b%%,*}; f=${b##*, }; [ "$f" -ge "$need" ] && { echo $i; return; }; sleep 120
  done; }
R() { need=$1; shift
  for t in 1 2 3 4 5; do
    g=$(pick $need); CUDA_VISIBLE_DEVICES=$g "$@" 2>&1 | tee -a mimic/baselines_full.log | grep -E "^\[seed|smote control"
    [ ${PIPESTATUS[0]} -eq 0 ] && return 0; sleep 240
  done; return 1; }
T="--textemb mimic/out/note_emb.npy --textids mimic/out/note_emb_ids.parquet"
G="--dxemb mimic/out/dx_text_emb_gloss.npy --dxids mimic/out/dx_text_ids_gloss.parquet"
B="--mode mtl $T $G --fixed-split"
echo "=== [1] ALDI-style distillation (identity retained, cold codes use the distilled projection) ==="
for s in 42 43 44 45 46; do
  R 10000 python3 -u mimic/train_mtl.py --seed $s $B --dx-mode input --aldi 1.0 --tag bl-aldi || exit 1
done
echo "=== [2] SMOTE-style non-LLM synthetic rows, matched to the persona row budget ==="
for s in 42 43 44 45 46; do
  R 10000 python3 -u mimic/train_mtl.py --seed $s $B --dx-mode strata --smote 51020 --tag bl-smote || exit 1
done
echo "=== BASELINES DONE ==="

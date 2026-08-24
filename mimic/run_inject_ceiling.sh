#!/bin/bash
cd .
export CUDA_VISIBLE_DEVICES=0
T="--textemb mimic/out/note_emb.npy --textids mimic/out/note_emb_ids.parquet"
G="--dxemb mimic/out/dx_text_emb_gloss.npy --dxids mimic/out/dx_text_ids_gloss.parquet"
BASE="--mode mtl $T $G --dx-mode strata --fixed-split"
for s in 42 43 44 45 46; do
  python3 -u mimic/train_mtl.py --seed $s $BASE --inject-labels mimic/out/inject_oracle.parquet --tag fx-inj-oracle 2>&1 | grep -E "^\[seed|label injection|mortality|readmit|LOS"
  python3 -u mimic/train_mtl.py --seed $s $BASE --inject-labels mimic/out/inject_prior.parquet  --tag fx-inj-prior  2>&1 | grep -E "^\[seed|mortality|readmit|LOS"
done
echo "=== INJECT CEILING DONE ==="

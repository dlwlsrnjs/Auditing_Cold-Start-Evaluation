#!/bin/bash
# Matched control for the ALDI comparison: identity retained (dx-mode input), fixed split,
# no distillation. Without it, bl-aldi vs rm-base confounds distillation with the identity
# treatment (strata vs input).
cd .
T="--textemb mimic/out/note_emb.npy --textids mimic/out/note_emb_ids.parquet"
G="--dxemb mimic/out/dx_text_emb_gloss.npy --dxids mimic/out/dx_text_ids_gloss.parquet"
run() { gpu=$1; shift; CUDA_VISIBLE_DEVICES=$gpu "$@" 2>&1 | tee -a mimic/aldictl_full.log | grep -E "^\[seed"; }
for s in 42 44 46; do
  run 0 python3 -u mimic/train_mtl.py --seed $s --mode mtl $T $G --dx-mode input --fixed-split --tag bl-identity
done >> mimic/aldictl.log 2>&1 &
for s in 43 45; do
  run 1 python3 -u mimic/train_mtl.py --seed $s --mode mtl $T $G --dx-mode input --fixed-split --tag bl-identity
done >> mimic/aldictl.log 2>&1 &
wait
echo "=== ALDICTL DONE ===" >> mimic/aldictl.log

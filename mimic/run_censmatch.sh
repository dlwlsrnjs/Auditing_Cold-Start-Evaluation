#!/bin/bash
# Round-8 B1: the naive and corrected censoring arms were scored on different test rows
# (105,360 at 20.0% prevalence vs 63,062 at 33.5%), because the naive basis refills censored
# labels with 0 before the evaluation mask is derived. Re-run both arms with --eval-uncensored
# so they are scored on identical observed rows and differ only in the training label.
cd .
T="--textemb mimic/out/note_emb.npy --textids mimic/out/note_emb_ids.parquet"
G="--dxemb mimic/out/dx_text_emb_gloss.npy --dxids mimic/out/dx_text_ids_gloss.parquet"
B="--mode mtl $T $G --dx-mode strata --eval-uncensored"
pick() { while true; do
  b=$(nvidia-smi --query-gpu=index,memory.free --format=csv,noheader,nounits | sort -t, -k2 -rn | head -1)
  i=${b%%,*}; f=${b##*, }; [ "$f" -ge 11000 ] && { echo $i; return; }; sleep 120; done; }
for s in 42 43 44 45 46; do
  g=$(pick); CUDA_VISIBLE_DEVICES=$g python3 -u mimic/train_mtl.py --seed $s $B --tag cm-corrected \
    2>&1 | tee -a mimic/censmatch_full.log | grep -E "^\[seed|NAIVE"
  g=$(pick); CUDA_VISIBLE_DEVICES=$g python3 -u mimic/train_mtl.py --seed $s $B --readmit-basis naive --tag cm-naive \
    2>&1 | tee -a mimic/censmatch_full.log | grep -E "^\[seed|NAIVE"
done >> mimic/censmatch.log 2>&1
echo "=== CENSMATCH DONE ===" >> mimic/censmatch.log

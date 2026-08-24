#!/bin/bash
# Score our model on the note-present subset (39,830 test admissions) so the ClinicalBERT
# comparison uses a common denominator (reviewer N19).
cd .
T="--textemb mimic/out/note_emb.npy --textids mimic/out/note_emb_ids.parquet"
G="--dxemb mimic/out/dx_text_emb_gloss.npy --dxids mimic/out/dx_text_ids_gloss.parquet"
B="--mode mtl $T $G --dx-mode strata --fixed-split --labsfeat mimic/out/labs48.parquet"
pick() { nvidia-smi --query-gpu=index,memory.free --format=csv,noheader,nounits | sort -t, -k2 -rn | head -1 | cut -d, -f1; }
for s in 42 43 44 45 46; do
  g=$(pick); CUDA_VISIBLE_DEVICES=$g python3 -u mimic/train_mtl.py --seed $s $B --tag nd-labs \
    2>&1 | tee -a mimic/notedenom_full.log | grep -E "^\[seed|note_present"
done >> mimic/notedenom.log 2>&1
echo "=== NOTEDENOM DONE ===" >> mimic/notedenom.log

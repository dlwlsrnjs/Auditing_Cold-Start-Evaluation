#!/bin/bash
cd .
T="--textemb mimic/out/note_emb.npy --textids mimic/out/note_emb_ids.parquet"
G="--dxemb mimic/out/dx_text_emb_gloss.npy --dxids mimic/out/dx_text_ids_gloss.parquet"
while ps -eo args | grep -q "[t]rain_mtl.py --seed 44"; do sleep 15; done
for s in 45 46; do
  CUDA_VISIBLE_DEVICES=1 python3 -u mimic/train_mtl.py --seed $s --mode mtl $T $G \
    --dx-mode strata --fixed-split --inject-labels mimic/out/inject_permanchor.parquet \
    --tag rm-permanchor 2>&1 | grep -E "^\[seed|mortality|Traceback|Error|CUDA"
done
echo "=== ANCHOR DONE ==="

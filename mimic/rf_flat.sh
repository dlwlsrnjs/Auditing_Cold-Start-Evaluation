#!/bin/bash
cd .
T="--textemb mimic/out/note_emb.npy --textids mimic/out/note_emb_ids.parquet"
G="--dxemb mimic/out/dx_text_emb_gloss.npy --dxids mimic/out/dx_text_ids_gloss.parquet"
for s in 45 46; do
  CUDA_VISIBLE_DEVICES=0 python3 -u mimic/train_mtl.py --seed $s --mode mtl $T $G \
    --dx-mode strata --fixed-split --personas mimic/out/personas_llm_flatlabel.parquet \
    --personaemb mimic/out/persona_emb_llm.npy --tag rm-persona-flatlabel 2>&1 \
    | grep -E "^\[seed|mortality|Traceback|Error|CUDA"
done
echo "=== FLAT DONE ==="

#!/bin/bash
# B1: permuted-findings control. Same persona rows, same labels, same codes --
# only the findings-text embedding is reassigned across codes within ICD chapter.
# If the persona gain survives this, the channel is not carrying text content.
cd .
T="--textemb mimic/out/note_emb.npy --textids mimic/out/note_emb_ids.parquet"
G="--dxemb mimic/out/dx_text_emb_gloss.npy --dxids mimic/out/dx_text_ids_gloss.parquet"
BASE="--mode mtl $T $G --dx-mode strata --fixed-split"
run() {
  for t in 1 2 3 4 5 6; do
    best=$(nvidia-smi --query-gpu=index,memory.free --format=csv,noheader,nounits | sort -t, -k2 -rn | head -1)
    idx=${best%%,*}; free=${best##*, }
    [ "$free" -ge 12000 ] || { echo "waiting for GPU (free=${free}MiB)"; sleep 120; continue; }
    CUDA_VISIBLE_DEVICES=$idx python3 -u mimic/train_mtl.py "$@" 2>&1 \
      | grep -E "^\[seed|mortality|persona augmentation|Traceback|Error|CUDA" && return 0
    sleep 300
  done
  echo "FAILED: $*"; return 1
}
for s in 42 43 44 45 46; do
  run --seed $s $BASE --personas mimic/out/personas_llm.parquet \
      --personaemb mimic/out/persona_emb_llm_shuf.npy --tag rm-persona-shuf || exit 1
done
echo "=== SHUFTEXT DONE ==="

#!/bin/bash
# B1 follow-up: permutation-draw variance. Three further independent draws of the
# within-chapter text permutation, 5 seeds each, so the intact-vs-permuted contrast
# can carry a draw component instead of being conditional on one realisation.
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
      | grep -E "^\[seed|mortality|Traceback|Error|CUDA" && return 0
    sleep 300
  done
  echo "FAILED: $*"; return 1
}
for d in 2 3 4; do
  echo "=== permutation draw $d ==="
  for s in 42 43 44 45 46; do
    run --seed $s $BASE --personas mimic/out/personas_llm.parquet \
        --personaemb mimic/out/persona_emb_llm_shuf${d}.npy --tag rm-persona-shuf${d} || exit 1
  done
done
echo "=== SHUFREPS DONE ==="

#!/bin/bash
# 외부 리뷰 M2/M3 대응.
#  M2: design 6 과 동일한 51,020 행에서 라벨만 warm ICD-3 기저율로 평탄화.
#      design 7 과 달리 행수·코드범위·예산이 전부 동일하므로 label-only intervention.
#  M3: prior anchor 를 그룹 내 순열로 대체. 참라벨의 정확한 유병률과 양성 수를 유지하고
#      admission-specific 정보만 제거하므로, 열화가 정보 손실만 반영한다.
cd .
T="--textemb mimic/out/note_emb.npy --textids mimic/out/note_emb_ids.parquet"
G="--dxemb mimic/out/dx_text_emb_gloss.npy --dxids mimic/out/dx_text_ids_gloss.parquet"
BASE="--mode mtl $T $G --dx-mode strata --fixed-split"
run() { for t in 1 2 3 4 5 6; do
  b=$(nvidia-smi --query-gpu=index,memory.free --format=csv,noheader,nounits | sort -t, -k2 -rn | head -1)
  i=${b%%,*}; f=${b##*, }; [ "$f" -ge 12000 ] || { echo "waiting GPU (${f}MiB)"; sleep 120; continue; }
  CUDA_VISIBLE_DEVICES=$i python3 -u mimic/train_mtl.py "$@" 2>&1 \
    | grep -E "^\[seed|mortality|Traceback|Error|CUDA" && return 0
  sleep 300; done; echo "FAILED: $*"; return 1; }
for s in 42 43 44 45 46; do
  run --seed $s $BASE --personas mimic/out/personas_llm_flatlabel.parquet \
      --personaemb mimic/out/persona_emb_llm.npy --tag rm-persona-flatlabel || exit 1
  run --seed $s $BASE --inject-labels mimic/out/inject_permanchor.parquet \
      --tag rm-permanchor || exit 1
done
echo "=== REVIEWFIX DONE ==="

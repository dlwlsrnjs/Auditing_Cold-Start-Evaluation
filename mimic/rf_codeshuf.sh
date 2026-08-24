#!/bin/bash
# 외부 리뷰 §4B 세 번째 조건: design 6 과 동일한 51,020 행에서 생성 라벨을
# 코드 내에서만 순열. 코드별 유병률·양성 수는 정확히 보존되고 persona↔label
# 짝만 파괴되므로, 이득이 코드 수준 라벨에서 오는지 persona 수준 라벨에서
# 오는지 분리한다. (flatlabel 은 코드 수준까지 지우므로 이 둘이 짝을 이룬다.)
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
  run --seed $s $BASE --personas mimic/out/personas_llm_codeshuf.parquet \
      --personaemb mimic/out/persona_emb_llm.npy --tag rm-persona-codeshuf || exit 1
done
echo "=== CODESHUF DONE ==="

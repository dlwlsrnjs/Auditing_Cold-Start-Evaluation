#!/bin/bash
# lab_n 과 lab_abn_frac 을 분리한다. 적대적 검증 리뷰어 지적:
#   lab_abn_frac 는 MIMIC 의 reference-range flag 에서 나오므로 analyte 값 파생이다.
#   따라서 두 변수를 묶어 "measurement intensity" 라 부른 것은 과장이다.
#   lab_n 만 순수 care-process 변수이고, 그것만으로 얼마가 나오는지가 실제 질문.
cd .
T="--textemb mimic/out/note_emb.npy --textids mimic/out/note_emb_ids.parquet"
G="--dxemb mimic/out/dx_text_emb_gloss.npy --dxids mimic/out/dx_text_ids_gloss.parquet"
BASE="--mode mtl $T $G --dx-mode strata --fixed-split --labsfeat mimic/out/labs48.parquet"
run() { for t in 1 2 3 4 5 6; do
  b=$(nvidia-smi --query-gpu=index,memory.free --format=csv,noheader,nounits | sort -t, -k2 -rn | head -1)
  i=${b%%,*}; f=${b##*, }; [ "$f" -ge 10000 ] || { sleep 120; continue; }
  CUDA_VISIBLE_DEVICES=$i python3 -u mimic/train_mtl.py "$@" 2>&1 \
    | grep -E "^\[seed|mortality|lab group|Traceback|Error" && return 0
  sleep 300; done; echo "FAILED: $*"; return 1; }
for s in 42 43 44 45 46; do
  run --seed $s $BASE --lab-group labn    --tag lg-labn    || exit 1
  run --seed $s $BASE --lab-group abnfrac --tag lg-abnfrac || exit 1
done
echo "=== LABN DONE ==="

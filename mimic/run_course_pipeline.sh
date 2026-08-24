#!/bin/bash
# Course-SFT pipeline with GPU auto-pick + wait-for-memory + stage gating.
cd .
set -o pipefail
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

pick_gpu() {  # wait until some GPU has >= $1 MiB free; echo its index
  need=$1
  while true; do
    best=$(nvidia-smi --query-gpu=index,memory.free --format=csv,noheader,nounits \
           | sort -t, -k2 -rn | head -1)
    idx=${best%%,*}; free=${best##*, }
    if [ "$free" -ge "$need" ]; then echo $idx; return; fi
    echo "waiting for GPU ($free MiB free, need $need)" >&2; sleep 120
  done
}

retry() {  # retry a command up to 5 times, re-picking the GPU each time
  need=$1; shift
  for t in 1 2 3 4 5 6 7 8 9 10; do
    g=$(pick_gpu $need)
    echo "--- attempt $t on GPU $g: $*" >&2
    CUDA_VISIBLE_DEVICES=$g "$@" && return 0
    echo "--- attempt $t failed; waiting 5m" >&2; sleep 300
  done
  return 1
}

echo "=== [0] numerical selftest ==="
if [ ! -f mimic/out/sft_config.txt ]; then
  retry 20000 python3 -u mimic/sft_selftest.py || { echo "SELFTEST FAILED"; exit 1; }
fi
read DT ATTN < mimic/out/sft_config.txt
echo "selftest chose dtype=$DT attn=$ATTN"
echo "=== [1] SFT ==="
if [ ! -f mimic/out/course_lora/adapter_model.safetensors ]; then
  retry 24000 python3 -u mimic/sft_course.py --sample 60000 --epochs 1 --bs 4 --accum 8 --lr 5e-5 --dtype $DT --attn $ATTN || { echo "SFT FAILED"; exit 1; }
fi
echo "=== [2] generation ==="
[ -f mimic/out/course_emb_sft.npy ] || retry 16000 python3 -u mimic/gen_course.py --mode sft || { echo "GEN-SFT FAILED"; exit 1; }
[ -f mimic/out/course_emb_zs.npy ]  || retry 16000 python3 -u mimic/gen_course.py --mode zs  || { echo "GEN-ZS FAILED"; exit 1; }
echo "=== [3] fixed-split runs ==="
T="--textemb mimic/out/note_emb.npy --textids mimic/out/note_emb_ids.parquet"
G="--dxemb mimic/out/dx_text_emb_gloss.npy --dxids mimic/out/dx_text_ids_gloss.parquet"
BASE="--mode mtl $T $G --dx-mode strata --fixed-split"
for s in 42 43 44 45 46; do
  retry 10000 python3 -u mimic/train_mtl.py --seed $s $BASE --agentemb mimic/out/course_emb_sft.npy --agentids mimic/out/course_ids_sft.parquet --tag fx-course-sft 2>&1 | grep -E "^\[seed|mortality|readmit|LOS"
  retry 10000 python3 -u mimic/train_mtl.py --seed $s $BASE --agentemb mimic/out/course_emb_zs.npy  --agentids mimic/out/course_ids_zs.parquet  --tag fx-course-zs  2>&1 | grep -E "^\[seed|mortality|readmit|LOS"
done
echo "=== COURSE PIPELINE DONE ==="

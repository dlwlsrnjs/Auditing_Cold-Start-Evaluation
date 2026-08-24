#!/bin/bash
cd .
set -o pipefail
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
pick_gpu() {
  need=$1
  while true; do
    best=$(nvidia-smi --query-gpu=index,memory.free --format=csv,noheader,nounits | sort -t, -k2 -rn | head -1)
    idx=${best%%,*}; free=${best##*, }
    [ "$free" -ge "$need" ] && { echo $idx; return; }
    echo "waiting for GPU ($free MiB free, need $need)" >&2; sleep 120
  done
}
retry() {
  need=$1; shift
  for t in 1 2 3 4 5 6 7 8 9 10; do
    g=$(pick_gpu $need)
    echo "--- attempt $t on GPU $g: $*" >&2
    CUDA_VISIBLE_DEVICES=$g "$@" && return 0
    echo "--- attempt $t failed; waiting 5m" >&2; sleep 300
  done
  return 1
}
T="--textemb mimic/out/note_emb.npy --textids mimic/out/note_emb_ids.parquet"
G="--dxemb mimic/out/dx_text_emb_gloss.npy --dxids mimic/out/dx_text_ids_gloss.parquet"
BASE="--mode mtl $T $G --dx-mode strata --fixed-split"

echo "=== [1] simulator full judge (76,909 rows, soft-mort) ==="
retry 13000 python3 -u mimic/gen_sim.py --bs 128 || { echo "SIM GEN FAILED"; exit 1; }
for s in 42 43 44 45 46; do
  retry 10000 python3 -u mimic/train_mtl.py --seed $s $BASE --inject-labels mimic/out/inject_sim.parquet --tag fx-inj-sim 2>&1 | grep -E "^\[seed|label injection|mortality|readmit|LOS"
done
echo "=== [2] patho personas full ==="
rm -f mimic/out/personas_patho.parquet mimic/out/persona_emb_patho.npy   # pilot files
retry 14000 python3 -u mimic/gen_personas.py --mode patho || { echo "PATHO GEN FAILED"; exit 1; }
for s in 42 43 44 45 46; do
  retry 10000 python3 -u mimic/train_mtl.py --seed $s $BASE --personas mimic/out/personas_patho.parquet --personaemb mimic/out/persona_emb_patho.npy --tag fx-persona-patho 2>&1 | grep -E "^\[seed|persona aug|mortality|readmit|LOS"
done
echo "=== [3] plain personas (ablation for patho) ==="
retry 14000 python3 -u mimic/gen_personas.py --mode llm || { echo "PLAIN GEN FAILED"; exit 1; }
for s in 42 43 44 45 46; do
  retry 10000 python3 -u mimic/train_mtl.py --seed $s $BASE --personas mimic/out/personas_llm.parquet --personaemb mimic/out/persona_emb_llm.npy --tag fx-persona-llm 2>&1 | grep -E "^\[seed|mortality|readmit|LOS"
  retry 10000 python3 -u mimic/train_mtl.py --seed $s $BASE --personas mimic/out/personas_naive.parquet --personaemb mimic/out/persona_emb_naive.npy --tag fx-persona-naive 2>&1 | grep -E "^\[seed|mortality|readmit|LOS"
done
echo "=== STAGE3 DONE ==="

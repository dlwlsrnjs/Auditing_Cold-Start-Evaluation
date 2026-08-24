#!/bin/bash
# Stage-2 queue: waits for the persona pipeline, then patho personas (dual cold axes),
# then the ColdLLM-style simulator (SFT -> label injection into the oracle-prior bracket).
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
until grep -qE "PERSONA PIPELINE DONE|FAILED" mimic/persona_pipeline.log 2>/dev/null; do sleep 300; done
T="--textemb mimic/out/note_emb.npy --textids mimic/out/note_emb_ids.parquet"
G="--dxemb mimic/out/dx_text_emb_gloss.npy --dxids mimic/out/dx_text_ids_gloss.parquet"
BASE="--mode mtl $T $G --dx-mode strata --fixed-split"

echo "=== [A] patho personas (rare-disease + rare-patient axes) ==="
[ -f mimic/out/persona_emb_patho.npy ] || retry 16000 python3 -u mimic/gen_personas.py --mode patho || { echo "PATHO GEN FAILED"; exit 1; }
for s in 42 43 44 45 46; do
  retry 10000 python3 -u mimic/train_mtl.py --seed $s $BASE --personas mimic/out/personas_patho.parquet --personaemb mimic/out/persona_emb_patho.npy --tag fx-persona-patho 2>&1 | grep -E "^\[seed|persona aug|mortality|readmit|LOS"
done

echo "=== [B] ColdLLM simulator: SFT -> label injection ==="
[ -f mimic/out/sim_lora/adapter_model.safetensors ] || retry 24000 python3 -u mimic/sft_sim.py || { echo "SIM SFT FAILED"; exit 1; }
[ -f mimic/out/inject_sim.parquet ] || retry 16000 python3 -u mimic/gen_sim.py || { echo "SIM GEN FAILED"; exit 1; }
for s in 42 43 44 45 46; do
  retry 10000 python3 -u mimic/train_mtl.py --seed $s $BASE --inject-labels mimic/out/inject_sim.parquet --tag fx-inj-sim 2>&1 | grep -E "^\[seed|label injection|mortality|readmit|LOS"
done
echo "=== STAGE2 DONE ==="

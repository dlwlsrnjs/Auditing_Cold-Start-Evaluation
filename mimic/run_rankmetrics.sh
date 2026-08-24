#!/bin/bash
cd .
until grep -q "PILOT-SEV-DONE" mimic/pilot_sev.log 2>/dev/null; do sleep 120; done
T="--textemb mimic/out/note_emb.npy --textids mimic/out/note_emb_ids.parquet"
G="--dxemb mimic/out/dx_text_emb_gloss.npy --dxids mimic/out/dx_text_ids_gloss.parquet"
BASE="--mode mtl $T $G --dx-mode strata --fixed-split"
run() {
  for t in 1 2 3 4 5 6; do
    best=$(nvidia-smi --query-gpu=index,memory.free --format=csv,noheader,nounits | sort -t, -k2 -rn | head -1)
    idx=${best%%,*}; free=${best##*, }
    [ "$free" -ge 10000 ] || { sleep 120; continue; }
    CUDA_VISIBLE_DEVICES=$idx python3 -u mimic/train_mtl.py "$@" 2>&1 | grep -E "^\[seed|mortality" && return 0
    sleep 300
  done
}
for s in 42 43 44 45 46; do
  run --seed $s $BASE --tag rm-base
  run --seed $s $BASE --personas mimic/out/personas_llm.parquet --personaemb mimic/out/persona_emb_llm.npy --tag rm-persona
  run --seed $s $BASE --personas mimic/out/personas_naive.parquet --personaemb mimic/out/persona_emb_naive.npy --tag rm-naive
  run --seed $s $BASE --inject-labels mimic/out/inject_oracle.parquet --tag rm-oracle
done
echo "=== RANKMETRICS DONE ==="

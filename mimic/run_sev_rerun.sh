#!/bin/bash
cd .
set -o pipefail
pick() {
  need=$1
  while true; do
    best=$(nvidia-smi --query-gpu=index,memory.free --format=csv,noheader,nounits | sort -t, -k2 -rn | head -1)
    idx=${best%%,*}; free=${best##*, }
    [ "$free" -ge "$need" ] && { echo $idx; return; }
    sleep 120
  done
}
T="--textemb mimic/out/note_emb.npy --textids mimic/out/note_emb_ids.parquet"
G="--dxemb mimic/out/dx_text_emb_gloss.npy --dxids mimic/out/dx_text_ids_gloss.parquet"
BASE="--mode mtl $T $G --dx-mode strata --fixed-split"
fail=0
for s in 42 43 44 45 46; do
  for cfg in sevw sev_raw; do
    ok=0
    for t in 1 2 3 4 5 6; do
      g=$(pick 10000)
      CUDA_VISIBLE_DEVICES=$g python3 -u mimic/train_mtl.py --seed $s $BASE \
        --personas mimic/out/personas_${cfg}.parquet --personaemb mimic/out/persona_emb_${cfg}.npy \
        --tag rm-persona-${cfg} 2>&1 | tee -a mimic/sev_rerun_full.log | grep -qE "^\[seed" && { ok=1; break; }
      sleep 300
    done
    [ $ok -eq 1 ] || { echo "seed=$s cfg=$cfg FAILED after 6 attempts" >> mimic/sev_rerun_full.log; fail=1; }
  done
done
[ $fail -eq 0 ] && echo "=== SEV RERUN DONE ===" || echo "=== SEV RERUN DONE WITH FAILURES ==="

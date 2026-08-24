#!/bin/bash
# Re-run design 5 (recalibrated simulator injection) with the offset that matches
# the observed rate on the injected rows exactly (-2.2257), replacing the -2.0553
# used in the original run. Same configuration as run_simcal.sh otherwise.
cd .
T="--textemb mimic/out/note_emb.npy --textids mimic/out/note_emb_ids.parquet"
G="--dxemb mimic/out/dx_text_emb_gloss.npy --dxids mimic/out/dx_text_ids_gloss.parquet"
for s in 42 43 44 45 46; do
  for t in 1 2 3 4 5; do
    best=$(nvidia-smi --query-gpu=index,memory.free --format=csv,noheader,nounits | sort -t, -k2 -rn | head -1)
    idx=${best%%,*}; free=${best##*, }
    [ "$free" -ge 10000 ] || { sleep 120; continue; }
    CUDA_VISIBLE_DEVICES=$idx python3 -u mimic/train_mtl.py --seed $s --mode mtl $T $G \
      --dx-mode strata --fixed-split --inject-labels mimic/out/inject_sim_cal_exact.parquet \
      --tag fx-inj-simcal-exact 2>&1 | grep -E "^\[seed|mortality" && break
    sleep 180
  done
done
echo "=== SIMCAL-EXACT DONE ==="

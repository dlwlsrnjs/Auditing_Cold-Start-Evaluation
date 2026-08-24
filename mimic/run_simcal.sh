#!/bin/bash
cd .
T="--textemb mimic/out/note_emb.npy --textids mimic/out/note_emb_ids.parquet"
G="--dxemb mimic/out/dx_text_emb_gloss.npy --dxids mimic/out/dx_text_ids_gloss.parquet"
for s in 42 43 44 45 46; do
  for t in 1 2 3 4 5; do
    best=$(nvidia-smi --query-gpu=index,memory.free --format=csv,noheader,nounits | sort -t, -k2 -rn | head -1)
    idx=${best%%,*}; free=${best##*, }
    [ "$free" -ge 10000 ] || { sleep 120; continue; }
    CUDA_VISIBLE_DEVICES=$idx python3 -u mimic/train_mtl.py --seed $s --mode mtl $T $G \
      --dx-mode strata --fixed-split --inject-labels mimic/out/inject_sim_cal.parquet \
      --tag fx-inj-simcal 2>&1 | grep -E "^\[seed|mortality|readmit" && break
    sleep 300
  done
done
echo "=== SIMCAL DONE ==="

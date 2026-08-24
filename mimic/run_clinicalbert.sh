#!/bin/bash
cd .
pick() {
  need=$1
  while true; do
    best=$(nvidia-smi --query-gpu=index,memory.free --format=csv,noheader,nounits | sort -t, -k2 -rn | head -1)
    idx=${best%%,*}; free=${best##*, }
    [ "$free" -ge "$need" ] && { echo $idx; return; }
    sleep 120
  done
}
for s in 42 43 44 45 46; do
  for t in 1 2 3 4 5; do
    g=$(pick 8000)
    CUDA_VISIBLE_DEVICES=$g python3 -u mimic/baseline_clinicalbert.py --seed $s --fixed-split --tag clinicalbert-ft \
      2>&1 | grep -E "^\[seed|text-covered|ep[0-9]" && break
    sleep 300
  done
done
echo "=== CLINICALBERT DONE ==="

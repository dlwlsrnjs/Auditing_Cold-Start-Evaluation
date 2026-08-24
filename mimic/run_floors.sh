#!/bin/bash
# Round-5 P6: the bracket's ceilings are 15-seed but its floors were 5-seed and from a
# different tag series, so the two ends are not comparable as printed. Re-run both floors
# at 15 seeds under the same protocol as sat-*.
cd .
Q=mimic/.floor_queue; LK=mimic/.floor_queue.lock
T="--textemb mimic/out/note_emb.npy --textids mimic/out/note_emb_ids.parquet"
G="--dxemb mimic/out/dx_text_emb_gloss.npy --dxids mimic/out/dx_text_ids_gloss.parquet"
B="--mode mtl $T $G --dx-mode strata --fixed-split"
{ for s in $(seq 42 56); do
    echo "python3 -u mimic/train_mtl.py --seed $s $B --labsfeat mimic/out/labs48.parquet --inject-labels mimic/out/inject_prior.parquet --tag sat-prior"
    echo "python3 -u mimic/train_mtl.py --seed $s $B --inject-labels mimic/out/inject_prior.parquet --tag sat-txt-prior"
  done; } > $Q
touch $LK; echo "queued $(wc -l < $Q)"
pick() { while true; do
    b=$(nvidia-smi --query-gpu=index,memory.free --format=csv,noheader,nounits | sort -t, -k2 -rn | head -1)
    i=${b%%,*}; f=${b##*, }; [ "$f" -ge 11000 ] && { echo $i; return; }; sleep 120; done; }
worker() { while true; do
    job=$(flock $LK -c "head -1 $Q; sed -i 1d $Q")
    [ -z "$job" ] && break
    for try in 1 2 3; do
      g=$(pick); CUDA_VISIBLE_DEVICES=$g $job 2>&1 | tee -a mimic/floors_full.log | grep -E "^\[seed|label injection" >> mimic/floors.log
      [ ${PIPESTATUS[0]} -eq 0 ] && break; sleep 180
    done; done; }
worker & worker & wait
echo "=== FLOORS DONE ===" >> mimic/floors.log

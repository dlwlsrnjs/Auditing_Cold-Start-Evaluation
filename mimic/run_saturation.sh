#!/bin/bash
# The paper's strongest claim -- "the label-injection channel is saturated on the laboratory
# baseline" -- rests on a 5-seed null (p=0.93) whose 95% CI is [-2.12, +2.28] pt, i.e. it still
# admits 59% of the text-only ceiling. A null is not evidence of absence at n=5, which is the
# exact error this paper criticises. Re-run both arms at 15 seeds so the interval can carry the
# claim (projected half-width ~1.0 pt), under fresh tags so existing analyses stay stable.
cd .
Q=mimic/.sat_queue; LK=mimic/.sat_queue.lock
T="--textemb mimic/out/note_emb.npy --textids mimic/out/note_emb_ids.parquet"
G="--dxemb mimic/out/dx_text_emb_gloss.npy --dxids mimic/out/dx_text_ids_gloss.parquet"
B="--mode mtl $T $G --dx-mode strata --fixed-split --labsfeat mimic/out/labs48.parquet"
{ for s in $(seq 42 56); do
    echo "python3 -u mimic/train_mtl.py --seed $s $B --tag sat-c1"
    echo "python3 -u mimic/train_mtl.py --seed $s $B --inject-labels mimic/out/inject_oracle.parquet --tag sat-oracle"
  done; } > $Q
touch $LK; echo "queued $(wc -l < $Q) runs"
pick() { while true; do
    b=$(nvidia-smi --query-gpu=index,memory.free --format=csv,noheader,nounits | sort -t, -k2 -rn | head -1)
    i=${b%%,*}; f=${b##*, }; [ "$f" -ge 11000 ] && { echo $i; return; }; sleep 120
  done; }
worker() { while true; do
    job=$(flock $LK -c "head -1 $Q; sed -i 1d $Q")
    [ -z "$job" ] && break
    for try in 1 2 3; do
      g=$(pick); CUDA_VISIBLE_DEVICES=$g $job 2>&1 | tee -a mimic/saturation_full.log | grep -E "^\[seed|label injection" >> mimic/saturation.log
      [ ${PIPESTATUS[0]} -eq 0 ] && break
      echo "RETRY $job" >> mimic/saturation.log; sleep 180
    done
  done; }
worker & worker & wait
echo "=== SATURATION DONE ===" >> mimic/saturation.log

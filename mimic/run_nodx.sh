#!/bin/bash
# Reviewer point: the diagnosis-text group carries discharge-time information. Showing that it
# is present in both arms is not sufficient, since a treatment effect can depend on it. Re-run
# the key comparisons with the diagnosis-text group removed entirely.
cd .
Q=mimic/.nodx_queue; LK=mimic/.nodx_queue.lock
T="--textemb mimic/out/note_emb.npy --textids mimic/out/note_emb_ids.parquet"
C="--cohort mimic/out/cohort_cens.parquet"
B="--mode mtl $T --dx-mode strata --fixed-split"
L="--labsfeat mimic/out/labs48.parquet"
P="--personas mimic/out/personas_llm.parquet --personaemb mimic/out/persona_emb_llm.npy"
{ for s in 42 43 44 45 46; do
    echo "python3 -u mimic/train_mtl.py --seed $s $B $C --tag nodx-base"
    echo "python3 -u mimic/train_mtl.py --seed $s $B $C $P --tag nodx-persona"
    echo "python3 -u mimic/train_mtl.py --seed $s $B $C $L --tag nodx-labs"
    echo "python3 -u mimic/train_mtl.py --seed $s $B $C $L $P --tag nodx-labs-persona"
  done; } > $Q
touch $LK; echo "queued $(wc -l < $Q)"
pick() { while true; do
  b=$(nvidia-smi --query-gpu=index,memory.free --format=csv,noheader,nounits | sort -t, -k2 -rn | head -1)
  i=${b%%,*}; f=${b##*, }; [ "$f" -ge 11000 ] && { echo $i; return; }; sleep 120; done; }
worker() { while true; do
  job=$(flock $LK -c "head -1 $Q; sed -i 1d $Q")
  [ -z "$job" ] && break
  for try in 1 2 3; do
    g=$(pick); CUDA_VISIBLE_DEVICES=$g $job 2>&1 | tee -a mimic/nodx_full.log | grep -E "^\[seed" >> mimic/nodx.log
    [ ${PIPESTATUS[0]} -eq 0 ] && break; sleep 180
  done; done; }
worker & worker & wait
echo "=== NODX DONE ===" >> mimic/nodx.log

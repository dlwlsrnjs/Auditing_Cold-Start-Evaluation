#!/bin/bash
# The persona code needs a diagnosis-text vector for each synthetic row, so the group cannot be
# removed outright for those arms. We instead swap the gloss embedding for the ICD title alone,
# which removes the LLM-written component while keeping the channel, and re-run the four
# comparisons the conclusions rest on.
cd .
Q=mimic/.to_queue; LK=mimic/.to_queue.lock
T="--textemb mimic/out/note_emb.npy --textids mimic/out/note_emb_ids.parquet"
G="--dxemb mimic/out/dx_text_emb.npy --dxids mimic/out/dx_text_ids.parquet"
C="--cohort mimic/out/cohort_cens.parquet"
B="--mode mtl $T $G --dx-mode strata --fixed-split"
L="--labsfeat mimic/out/labs48.parquet"
P="--personas mimic/out/personas_llm.parquet --personaemb mimic/out/persona_emb_llm.npy"
{ for s in 42 43 44 45 46; do
    echo "python3 -u mimic/train_mtl.py --seed $s $B $C --tag to-base"
    echo "python3 -u mimic/train_mtl.py --seed $s $B $C $P --tag to-persona"
    echo "python3 -u mimic/train_mtl.py --seed $s $B $C $L --tag to-labs"
    echo "python3 -u mimic/train_mtl.py --seed $s $B $C $L $P --tag to-labs-persona"
  done; } > $Q
touch $LK; echo "queued $(wc -l < $Q)"
pick() { while true; do
  b=$(nvidia-smi --query-gpu=index,memory.free --format=csv,noheader,nounits | sort -t, -k2 -rn | head -1)
  i=${b%%,*}; f=${b##*, }; [ "$f" -ge 11000 ] && { echo $i; return; }; sleep 60; done; }
worker() { while true; do
  job=$(flock $LK -c "head -1 $Q; sed -i 1d $Q")
  [ -z "$job" ] && break
  for try in 1 2 3; do
    g=$(pick); CUDA_VISIBLE_DEVICES=$g $job 2>&1 | tee -a mimic/titleonly_full.log | grep -E "^\[seed" >> mimic/titleonly.log
    [ ${PIPESTATUS[0]} -eq 0 ] && break; sleep 120
  done; done; }
worker & worker & wait
echo "=== TITLEONLY DONE ===" >> mimic/titleonly.log

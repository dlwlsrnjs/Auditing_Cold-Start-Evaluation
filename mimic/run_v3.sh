#!/bin/bash
# Rebuild the two contested analyses on corrected foundations.
#  Q1: corrected censoring definition (cohort_cens) AND matched evaluation rows.
#  Q3: landmark mortality -- score only admissions still in hospital at the 48h landmark,
#      so no outcome falls inside the feature window.
cd .
Q=mimic/.v3_queue; LK=mimic/.v3_queue.lock
T="--textemb mimic/out/note_emb.npy --textids mimic/out/note_emb_ids.parquet"
G="--dxemb mimic/out/dx_text_emb_gloss.npy --dxids mimic/out/dx_text_ids_gloss.parquet"
C="--cohort mimic/out/cohort_cens.parquet"
B="--mode mtl $T $G --dx-mode strata"
L="--labsfeat mimic/out/labs48.parquet"
P="--personas mimic/out/personas_llm.parquet --personaemb mimic/out/persona_emb_llm.npy"
{ for s in 42 43 44 45 46; do
    # Q1 on the corrected cohort, both arms scored on identical observed rows
    echo "python3 -u mimic/train_mtl.py --seed $s $B $C --eval-uncensored --tag v3-corrected"
    echo "python3 -u mimic/train_mtl.py --seed $s $B $C --eval-uncensored --readmit-basis naive --tag v3-naive"
    # landmark mortality: admissions surviving in hospital past the 48h landmark
    echo "python3 -u mimic/train_mtl.py --seed $s $B $C --fixed-split --exclude-early-deaths 48 --tag v3-lm-base"
    echo "python3 -u mimic/train_mtl.py --seed $s $B $C --fixed-split --exclude-early-deaths 48 $L --tag v3-lm-labs"
    echo "python3 -u mimic/train_mtl.py --seed $s $B $C --fixed-split --exclude-early-deaths 48 $L $P --tag v3-lm-labs-persona"
    echo "python3 -u mimic/train_mtl.py --seed $s $B $C --fixed-split --exclude-early-deaths 48 $P --tag v3-lm-persona"
  done; } > $Q
touch $LK; echo "queued $(wc -l < $Q)"
pick() { while true; do
  b=$(nvidia-smi --query-gpu=index,memory.free --format=csv,noheader,nounits | sort -t, -k2 -rn | head -1)
  i=${b%%,*}; f=${b##*, }; [ "$f" -ge 11000 ] && { echo $i; return; }; sleep 120; done; }
worker() { while true; do
  job=$(flock $LK -c "head -1 $Q; sed -i 1d $Q")
  [ -z "$job" ] && break
  for try in 1 2 3; do
    g=$(pick); CUDA_VISIBLE_DEVICES=$g $job 2>&1 | tee -a mimic/v3_full.log | grep -E "^\[seed|exclude-early|NAIVE" >> mimic/v3.log
    [ ${PIPESTATUS[0]} -eq 0 ] && break; sleep 180
  done; done; }
worker & worker & wait
echo "=== V3 DONE ===" >> mimic/v3.log

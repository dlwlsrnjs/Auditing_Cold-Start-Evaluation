#!/bin/bash
# Resume the O3 -> baselines -> labgroups queue that died when the container restarted
# (2026-08-18 16:58 KST). Two GPUs are free, so instead of one serial chain we pull from a
# shared job queue with one worker pinned per GPU.
cd .
Q=mimic/.resume_queue
LK=mimic/.resume_queue.lock

T="--textemb mimic/out/note_emb.npy --textids mimic/out/note_emb_ids.parquet"
G="--dxemb mimic/out/dx_text_emb_gloss.npy --dxids mimic/out/dx_text_ids_gloss.parquet"
L="--labsfeat mimic/out/labs48.parquet"

{
# [o3] only survivor of the interrupted run: identity-dropout seed 46
echo "o3 python3 -u mimic/train_mtl.py --seed 46 --mode mtl $T $G $L --dx-mode input --dx-dropout 0.5 --fixed-split --tag o3-labs-dxdrop"
# [baselines] ALDI-style warm->cold distillation, then SMOTE-style non-LLM synthetic rows
for s in 42 43 44 45 46; do
  echo "baselines python3 -u mimic/train_mtl.py --seed $s --mode mtl $T $G --fixed-split --dx-mode input --aldi 1.0 --tag bl-aldi"
done
for s in 42 43 44 45 46; do
  echo "baselines python3 -u mimic/train_mtl.py --seed $s --mode mtl $T $G --fixed-split --dx-mode strata --smote 51020 --tag bl-smote"
done
# [labgroups] which physiological family carries the laboratory gain
for grp in renal heme coag hepatic perfusion; do
  for s in 42 43 44 45 46; do
    echo "labgroups python3 -u mimic/train_mtl.py --seed $s --mode mtl $T $G --dx-mode strata --fixed-split $L --lab-group $grp --tag lg-$grp"
  done
done
} > $Q
echo "queued $(wc -l < $Q) runs"

worker() {
  gpu=$1
  while true; do
    job=$(flock $LK -c "head -1 $Q; sed -i 1d $Q")
    [ -z "$job" ] && break
    sec=${job%% *}; cmd=${job#* }
    for try in 1 2 3; do
      CUDA_VISIBLE_DEVICES=$gpu $cmd 2>&1 | tee -a mimic/${sec}_full.log \
        | grep -E "^\[seed|lab group|label injection|smote control|aldi" >> mimic/${sec}.log
      [ ${PIPESTATUS[0]} -eq 0 ] && break
      echo "RETRY($try) gpu$gpu: $cmd" >> mimic/resume.log; sleep 120
    done
  done
}
touch $LK
worker 0 & worker 1 &
wait
# markers the downstream watchers in the old scripts look for
echo "=== O3 DONE ==="        >> mimic/o3.log
echo "=== BASELINES DONE ===" >> mimic/baselines.log
echo "=== LABGROUPS DONE ===" >> mimic/labgroups.log
echo "=== RESUME DONE ==="    >> mimic/resume.log

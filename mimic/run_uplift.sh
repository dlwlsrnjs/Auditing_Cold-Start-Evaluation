#!/bin/bash
# Uplift queue: waits for sev runs + labs build, then (1) labs sanity, (2) labs 5-seed,
# (3) gte re-embed, (4) final combined configs.
cd .
set -o pipefail
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
pick() {
  need=$1
  while true; do
    best=$(nvidia-smi --query-gpu=index,memory.free --format=csv,noheader,nounits | sort -t, -k2 -rn | head -1)
    idx=${best%%,*}; free=${best##*, }
    [ "$free" -ge "$need" ] && { echo $idx; return; }
    sleep 120
  done
}
R() {
  need=$1; shift
  for t in 1 2 3 4 5 6 7 8; do
    g=$(pick $need)
    CUDA_VISIBLE_DEVICES=$g "$@" && return 0
    sleep 300
  done
  return 1
}
until grep -q "SEV FULL DONE" mimic/sev_full.log 2>/dev/null; do sleep 300; done
until grep -q "\[saved\] labs48" mimic/build_labs.log 2>/dev/null; do sleep 120; done
T="--textemb mimic/out/note_emb.npy --textids mimic/out/note_emb_ids.parquet"
G="--dxemb mimic/out/dx_text_emb_gloss.npy --dxids mimic/out/dx_text_ids_gloss.parquet"
L="--labsfeat mimic/out/labs48.parquet"
P="--personas mimic/out/personas_llm.parquet --personaemb mimic/out/persona_emb_llm.npy"
BASE="--mode mtl --dx-mode strata --fixed-split"
echo "=== [1] labs sanity (seed 42) ==="
R 10000 python3 -u mimic/train_mtl.py --seed 42 $BASE $T $G $L --tag labs-sanity 2>&1 | grep -E "^\[seed|labs block|mortality|recall"
echo "=== [2] labs & labs+persona, 5 seeds ==="
for s in 42 43 44 45 46; do
  R 10000 python3 -u mimic/train_mtl.py --seed $s $BASE $T $G $L --tag rm2-labs 2>&1 | grep -E "^\[seed|mortality"
  R 10000 python3 -u mimic/train_mtl.py --seed $s $BASE $T $G $L $P --tag rm2-labs-persona 2>&1 | grep -E "^\[seed|mortality"
done
echo "=== [3] gte re-embedding ==="
[ -f mimic/out/note_emb_gte.npy ] || R 12000 python3 -u mimic/embed_gte.py --which all || { echo "GTE EMBED FAILED"; exit 1; }
echo "=== [4] gte configs, 5 seeds ==="
T2="--textemb mimic/out/note_emb_gte.npy --textids mimic/out/note_emb_ids.parquet"
G2="--dxemb mimic/out/dx_text_emb_gte.npy --dxids mimic/out/dx_text_ids_gloss.parquet"
P2="--personas mimic/out/personas_llm.parquet --personaemb mimic/out/persona_emb_gte.npy"
for s in 42 43 44 45 46; do
  R 10000 python3 -u mimic/train_mtl.py --seed $s $BASE $T2 $G2 $L --tag rm2-gte-labs 2>&1 | grep -E "^\[seed|mortality"
  R 10000 python3 -u mimic/train_mtl.py --seed $s $BASE $T2 $G2 $L $P2 --tag rm2-gte-labs-persona 2>&1 | grep -E "^\[seed|mortality"
done
echo "=== UPLIFT DONE ==="

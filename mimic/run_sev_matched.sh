#!/bin/bash
# Budget-matched rerun of designs 8-9: four severity personas per rare code (one per
# tier) instead of eight, so the arm is row-matched to C2 the way design 6 is.
# Waits for the permutation replicates to release the GPUs first.
cd .
until grep -q "SHUFREPS DONE" mimic/shufreps.log 2>/dev/null; do sleep 120; done
echo "=== [1] budget-matched sev generation (12,762 codes x 4) ==="
pick() { while true; do
  b=$(nvidia-smi --query-gpu=index,memory.free --format=csv,noheader,nounits | sort -t, -k2 -rn | head -1)
  i=${b%%,*}; f=${b##*, }; [ "$f" -ge 30000 ] && { echo $i; return; }; sleep 120; done; }
g=$(pick)
CUDA_VISIBLE_DEVICES=$g python3 -u mimic/gen_personas.py --mode sev --sev-k 4 \
  --out-suffix 4 --bs 96 || { echo "SEV4 GEN FAILED"; exit 1; }
[ -f mimic/out/persona_emb_sev4.npy ] || { echo "SEV4 EMB MISSING"; exit 1; }
echo "=== [2] post-process: same funnel masking and prevalence weights ==="
python3 - <<'PY'
import pandas as pd, numpy as np
per=pd.read_parquet("mimic/out/personas_sev4.parquet")
per["tier"]=per.persona_k % 4
per.loc[per.y_mort==1,"y_readmit"]=np.nan
per.to_parquet("mimic/out/personas_sev4_raw.parquet",index=False)
share=per.tier.value_counts(normalize=True).sort_index().values
target=np.array([.65,.25,.08,.02])
w=target/np.maximum(share,1e-9); w=w/(per.tier.map(dict(enumerate(w))).mean())
per["conf"]=per.tier.map(dict(enumerate(w))).astype(float)
per.to_parquet("mimic/out/personas_sev4w.parquet",index=False)
print(f"rows {len(per):,}  mort={per.y_mort.mean():.3f}  weights={np.round(w,3)}")
PY
cp mimic/out/persona_emb_sev4.npy mimic/out/persona_emb_sev4w.npy
cp mimic/out/persona_emb_sev4.npy mimic/out/persona_emb_sev4_raw.npy
echo "=== [3] runs: budget-matched sev-weighted + sev-raw, 5 seeds ==="
T="--textemb mimic/out/note_emb.npy --textids mimic/out/note_emb_ids.parquet"
G="--dxemb mimic/out/dx_text_emb_gloss.npy --dxids mimic/out/dx_text_ids_gloss.parquet"
BASE="--mode mtl $T $G --dx-mode strata --fixed-split"
run() { for t in 1 2 3 4 5 6; do
  b=$(nvidia-smi --query-gpu=index,memory.free --format=csv,noheader,nounits | sort -t, -k2 -rn | head -1)
  i=${b%%,*}; f=${b##*, }; [ "$f" -ge 12000 ] || { sleep 120; continue; }
  CUDA_VISIBLE_DEVICES=$i python3 -u mimic/train_mtl.py "$@" 2>&1 \
    | grep -E "^\[seed|mortality|Traceback|Error|CUDA" && return 0
  sleep 300; done; echo "FAILED: $*"; return 1; }
for s in 42 43 44 45 46; do
  run --seed $s $BASE --personas mimic/out/personas_sev4w.parquet \
      --personaemb mimic/out/persona_emb_sev4w.npy --tag rm-persona-sev4w || exit 1
  run --seed $s $BASE --personas mimic/out/personas_sev4_raw.parquet \
      --personaemb mimic/out/persona_emb_sev4_raw.npy --tag rm-persona-sev4raw || exit 1
done
echo "=== SEV MATCHED DONE ==="

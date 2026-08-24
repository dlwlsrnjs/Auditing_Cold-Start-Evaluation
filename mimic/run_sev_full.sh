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
echo "=== [1] full sev generation (12,762 codes x 8) ==="
rm -f mimic/out/personas_sev.parquet mimic/out/persona_emb_sev.npy
for t in 1 2 3 4 5 6 7 8; do
  g=$(pick 14000)
  CUDA_VISIBLE_DEVICES=$g python3 -u mimic/gen_personas.py --mode sev --bs 96 && break
  sleep 300
done
[ -f mimic/out/persona_emb_sev.npy ] || { echo "SEV GEN FAILED"; exit 1; }
echo "=== [2] post-process: prevalence weights + funnel masking ==="
python3 - <<'PY'
import pandas as pd, numpy as np
per=pd.read_parquet("mimic/out/personas_sev.parquet")
per["tier"]=per.persona_k % 4
# funnel consistency: the dead cannot be readmitted
per.loc[per.y_mort==1,"y_readmit"]=np.nan
raw=per.copy(); raw.to_parquet("mimic/out/personas_sev_raw.parquet",index=False)
# prevalence weights: tier mix in reality ~ [.65,.25,.08,.02]; sample shares from data
share=per.tier.value_counts(normalize=True).sort_index().values
target=np.array([.65,.25,.08,.02])
w=target/np.maximum(share,1e-9); w=w/ (per.tier.map(dict(enumerate(w))).mean())
per["conf"]=per.tier.map(dict(enumerate(w))).astype(float)
per.to_parquet("mimic/out/personas_sevw.parquet",index=False)
print("weights per tier:",np.round(w,3),"weighted mort prior:",
      float((per.y_mort*per.conf).sum()/per.conf.sum()).__round__(4))
PY
cp mimic/out/persona_emb_sev.npy mimic/out/persona_emb_sevw.npy
cp mimic/out/persona_emb_sev.npy mimic/out/persona_emb_sev_raw.npy
echo "=== [3] runs: sev-weighted + sev-raw, 5 seeds ==="
T="--textemb mimic/out/note_emb.npy --textids mimic/out/note_emb_ids.parquet"
G="--dxemb mimic/out/dx_text_emb_gloss.npy --dxids mimic/out/dx_text_ids_gloss.parquet"
BASE="--mode mtl $T $G --dx-mode strata --fixed-split"
for s in 42 43 44 45 46; do
  for cfg in "sevw" "sev_raw"; do
    for t in 1 2 3 4 5 6; do
      g=$(pick 10000)
      CUDA_VISIBLE_DEVICES=$g python3 -u mimic/train_mtl.py --seed $s $BASE \
        --personas mimic/out/personas_${cfg}.parquet --personaemb mimic/out/persona_emb_${cfg}.npy \
        --tag rm-persona-${cfg} 2>&1 | grep -E "^\[seed|mortality" && break
      sleep 300
    done
  done
done
echo "=== SEV FULL DONE ==="

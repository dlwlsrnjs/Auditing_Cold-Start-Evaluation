#!/bin/bash
cd .
export CUDA_VISIBLE_DEVICES=0
python3 -u mimic/agent_evidence.py --mode agent  --train-sample 25000 --bs 96
python3 -u mimic/agent_evidence.py --mode noretr --train-sample 25000 --bs 96
echo "=== GEN DONE ==="
T="--textemb mimic/out/note_emb.npy --textids mimic/out/note_emb_ids.parquet"
G="--dxemb mimic/out/dx_text_emb_gloss.npy --dxids mimic/out/dx_text_ids_gloss.parquet"
BASE="--mode mtl $T $G --dx-mode strata --fixed-split"
for s in 42 43 44 45 46; do
  python3 -u mimic/train_mtl.py --seed $s $BASE --tag fx-base 2>&1 | grep -E "^\[seed|mortality|readmit|LOS"
  python3 -u mimic/train_mtl.py --seed $s $BASE --knn --tag fx-knn 2>&1 | grep -E "^\[seed|mortality|readmit|LOS"
  python3 -u mimic/train_mtl.py --seed $s $BASE --agentemb mimic/out/agent_emb_agent.npy --agentids mimic/out/agent_ids_agent.parquet --tag fx-agent 2>&1 | grep -E "^\[seed|mortality|readmit|LOS"
  python3 -u mimic/train_mtl.py --seed $s $BASE --agentemb mimic/out/agent_emb_noretr.npy --agentids mimic/out/agent_ids_noretr.parquet --tag fx-noretr 2>&1 | grep -E "^\[seed|mortality|readmit|LOS"
  python3 -u mimic/train_mtl.py --seed $s $BASE --knn --agentemb mimic/out/agent_emb_agent.npy --agentids mimic/out/agent_ids_agent.parquet --tag fx-knn-agent 2>&1 | grep -E "^\[seed|mortality|readmit|LOS"
done
echo "=== AGENT GRID DONE ==="

#!/bin/bash
# 리뷰 라운드 3 대응 3건.
#  A. hospice  : hospital_expire_flag 가 0 인 호스피스 퇴원자를 사망 양성으로 재라벨.
#                호스피스 비율이 warm-user 세그먼트에서 2배 높으므로(1.28/1.25% vs
#                0.63/0.57%), "cold user 가 더 어려운 세그먼트가 아니다"(75/76)가
#                라벨 오염의 산물인지 검사한다. C1 설정 그대로, 라벨만 다르다.
#  B. lg-count : lab_n, lab_abn_frac 두 전역 요약만으로 학습. 검사 '강도'가 생리학적
#                중증도와 별개로 얼마나 운반하는지 — analyte family 분해가 이 둘을
#                빼놓고 돌았으므로 그 분해가 답하지 못한 질문이다.
#  C. chapshuf : 생성 라벨을 ICD 챕터 안에서 코드 간 순열. 주입 양성 수와 주변분포는
#                정확히 유지되고 code<->rate 대응만 파괴된다. 이득이 살아남으면
#                "운반자는 per-code rate" 주장이 반증된다.
cd .
T="--textemb mimic/out/note_emb.npy --textids mimic/out/note_emb_ids.parquet"
G="--dxemb mimic/out/dx_text_emb_gloss.npy --dxids mimic/out/dx_text_ids_gloss.parquet"
BASE="--mode mtl $T $G --dx-mode strata --fixed-split"
run() { for t in 1 2 3 4 5 6; do
  b=$(nvidia-smi --query-gpu=index,memory.free --format=csv,noheader,nounits | sort -t, -k2 -rn | head -1)
  i=${b%%,*}; f=${b##*, }; [ "$f" -ge 10000 ] || { echo "waiting GPU (${f}MiB)"; sleep 120; continue; }
  CUDA_VISIBLE_DEVICES=$i python3 -u mimic/train_mtl.py "$@" 2>&1 \
    | grep -E "^\[seed|mortality|hospice-positive|lab group|Traceback|Error|CUDA" && return 0
  sleep 300; done; echo "FAILED: $*"; return 1; }
for s in 42 43 44 45 46; do
  run --seed $s $BASE --personas mimic/out/personas_llm_chapshuf.parquet \
      --personaemb mimic/out/persona_emb_llm.npy --tag rm-persona-chapshuf || exit 1
done
echo "=== CHAPSHUF DONE ==="
for s in 42 43 44 45 46; do
  run --seed $s $BASE --hospice-positive --tag hosp-base || exit 1
  run --seed $s $BASE --hospice-positive --personas mimic/out/personas_llm.parquet \
      --personaemb mimic/out/persona_emb_llm.npy --tag hosp-persona || exit 1
done
echo "=== HOSPICE DONE ==="
for s in 42 43 44 45 46; do
  run --seed $s $BASE --labsfeat mimic/out/labs48.parquet --lab-group count --tag lg-count || exit 1
done
echo "=== LGCOUNT DONE ==="

#!/usr/bin/env python3
"""Automated content audit of a generated persona corpus, no clinician required.

(1) Code specificity: retrieve each persona's own diagnosis code among the rare codes by
    cosine similarity between its findings embedding and the code's title+gloss embedding.
    Chance is 1/n_codes; chapter agreement of the top hit is compared with the
    chapter-size-weighted chance rate.
(2) Label calibration: corpus mortality / readmission / LOS against the real cold-item rates.

    python3 mimic/audit_persona_content.py --personas mimic/out/personas_llm.parquet \
        --emb mimic/out/persona_emb_llm.npy
"""
import argparse, numpy as np, pandas as pd
ap = argparse.ArgumentParser()
ap.add_argument("--personas", default="mimic/out/personas_llm.parquet")
ap.add_argument("--emb",      default="mimic/out/persona_emb_llm.npy")
ap.add_argument("--dxemb",    default="mimic/out/dx_text_emb_gloss.npy")
ap.add_argument("--dxids",    default="mimic/out/dx_text_ids_gloss.parquet")
ap.add_argument("--cohort",   default="mimic/out/cohort_cens.parquet")
A = ap.parse_args()

fe = np.load(A.emb).astype(np.float32)
de = np.load(A.dxemb).astype(np.float32)
gid = pd.read_parquet(A.dxids)
per = pd.read_parquet(A.personas)
coh = pd.read_parquet(A.cohort)
rare = set(coh[coh.cold_item_rare_dx == 1].primary_icd.astype(str))
m = gid.primary_icd.astype(str).isin(rare).values
D, codes = de[m], gid.primary_icd.astype(str).values[m]
chap = np.array([c[0] for c in codes])
pos = {c: i for i, c in enumerate(codes)}
tgt = per.primary_icd.astype(str).map(pos).values
ok = ~pd.isna(tgt); tgt = tgt[ok].astype(int); F = fe[ok]

ranks = np.empty(len(F), np.int32); top1 = np.empty(len(F), np.int32)
for a in range(0, len(F), 2000):
    b = min(a + 2000, len(F)); S = F[a:b] @ D.T
    own = S[np.arange(b - a), tgt[a:b]][:, None]
    ranks[a:b] = (S > own).sum(1) + 1; top1[a:b] = S.argmax(1)
cnt = pd.Series(chap).value_counts(normalize=True)
print(f"codes {len(codes):,}  personas {len(F):,}")
print(f"top-1 {(ranks==1).mean()*100:.2f}% (chance {100/len(codes):.3f}%)  "
      f"top-10 {(ranks<=10).mean()*100:.2f}%  median rank {int(np.median(ranks)):,}")
print(f"chapter agreement of top hit {(chap[top1]==chap[tgt]).mean()*100:.1f}% "
      f"(chance {float((cnt**2).sum())*100:.1f}%)")
ci = coh[coh.cold_item_rare_dx == 1]
print(f"labels: mort {per.y_mort.mean()*100:.2f}% ({per.y_mort.mean()/ci.y_mortality.mean():.2f}x real), "
      f"readmit {per.y_readmit.mean()*100:.2f}% ({per.y_readmit.mean()/ci.y_readmit_30d.mean():.2f}x)")

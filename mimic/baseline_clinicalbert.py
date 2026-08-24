#!/usr/bin/env python3
"""SOTA baseline: Huang et al. 2019 ClinicalBERT recipe, adapted to our cohort/split/labels.

Fine-tunes Bio_ClinicalBERT end-to-end on first-48h note text for mortality / readmission
(binary heads). This is the direct comparison our paper's frozen-embedding pipeline needs:
does full fine-tuning of a domain LM beat frozen mean-pool + structured fusion?

Differences from Huang et al. (kept faithful where it matters, adapted where our protocol
requires it): same base model family and per-admission note truncation/concatenation idea;
adapted to our strict cold-item split, our label definitions (censoring-corrected readmission),
and our note availability (36.7% -> text-only rows are the ones this baseline can even use).

Usage: python3 mimic/baseline_clinicalbert.py --seed 42
"""
import argparse, json, time
import numpy as np, pandas as pd, torch, torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import AutoModel, AutoTokenizer
from sklearn.metrics import roc_auc_score, average_precision_score

ap = argparse.ArgumentParser()
ap.add_argument("--cohort", default="mimic/out/cohort.parquet")
ap.add_argument("--notes", default="mimic/out/notes_early.parquet")
ap.add_argument("--out", default="mimic/out")
ap.add_argument("--model", default="emilyalsentzer/Bio_ClinicalBERT")
ap.add_argument("--seed", type=int, default=42)
ap.add_argument("--fixed-split", action="store_true")
ap.add_argument("--epochs", type=int, default=3)
ap.add_argument("--bs", type=int, default=16)
ap.add_argument("--accum", type=int, default=2)
ap.add_argument("--lr", type=float, default=2e-5)
ap.add_argument("--maxlen", type=int, default=512)
ap.add_argument("--device", default="cuda:0")
ap.add_argument("--tag", default="clinicalbert")
A = ap.parse_args()
t0 = time.time()
torch.manual_seed(A.seed); np.random.seed(A.seed)

d = pd.read_parquet(A.cohort).reset_index(drop=True)
rng = np.random.default_rng(0 if A.fixed_split else A.seed)
pids = d.subject_id.unique(); rng.shuffle(pids)
n = len(pids); tr_p = set(pids[:int(.7 * n)]); va_p = set(pids[int(.7 * n):int(.8 * n)])
d["split"] = np.where(d.subject_id.isin(tr_p), "train",
             np.where(d.subject_id.isin(va_p), "val", "test"))
d.loc[(d.split == "train") & (d.cold_item_rare_dx == 1), "split"] = "drop"

notes = pd.read_parquet(A.notes).sort_values(["hadm_id", "charttime"])
note_txt = notes.groupby("hadm_id").text.apply(lambda s: "\n".join(s)).str.slice(0, 4000)
d = d[d.hadm_id.isin(note_txt.index)].reset_index(drop=True)   # text-only baseline: needs a note
print(f"text-covered admissions: {len(d):,}  split={d.split.value_counts().to_dict()}", flush=True)

tok = AutoTokenizer.from_pretrained(A.model)

class DS(Dataset):
    def __init__(s, sub): s.sub = sub.reset_index(drop=True)
    def __len__(s): return len(s.sub)
    def __getitem__(s, i):
        r = s.sub.iloc[i]
        txt = note_txt.loc[r.hadm_id]
        return txt, float(r.y_mortality), float(r.y_readmit_30d) if pd.notna(r.y_readmit_30d) else -1.0

def collate(batch):
    txts, ym, yr = zip(*batch)
    enc = tok(list(txts), padding=True, truncation=True, max_length=A.maxlen, return_tensors="pt")
    return enc, torch.tensor(ym, dtype=torch.float32), torch.tensor(yr, dtype=torch.float32)

tr_ds = DS(d[d.split == "train"]); va_ds = DS(d[d.split == "val"]); te_ds = DS(d[d.split == "test"])
tr_dl = DataLoader(tr_ds, batch_size=A.bs, shuffle=True, collate_fn=collate, num_workers=4)
va_dl = DataLoader(va_ds, batch_size=32, shuffle=False, collate_fn=collate, num_workers=2)
te_dl = DataLoader(te_ds, batch_size=32, shuffle=False, collate_fn=collate, num_workers=2)

class ClinicalBERTHeads(nn.Module):
    def __init__(s):
        super().__init__()
        s.bert = AutoModel.from_pretrained(A.model)
        h = s.bert.config.hidden_size
        s.mort = nn.Linear(h, 1); s.readmit = nn.Linear(h, 1)
    def forward(s, enc):
        out = s.bert(**enc).last_hidden_state[:, 0]     # [CLS]
        return s.mort(out).squeeze(-1), s.readmit(out).squeeze(-1)

net = ClinicalBERTHeads().to(A.device)
opt = torch.optim.AdamW(net.parameters(), lr=A.lr)
bce = nn.BCEWithLogitsLoss(reduction="none")

@torch.no_grad()
def evaluate(dl):
    net.eval(); pm, pr, ym, yr = [], [], [], []
    for enc, ymb, yrb in dl:
        enc = {k: v.to(A.device) for k, v in enc.items()}
        lm, lr_ = net(enc)
        pm.append(torch.sigmoid(lm).cpu().numpy()); pr.append(torch.sigmoid(lr_).cpu().numpy())
        ym.append(ymb.numpy()); yr.append(yrb.numpy())
    pm, pr, ym, yr = map(np.concatenate, (pm, pr, ym, yr))
    m = yr >= 0
    r = {"mort_auroc": roc_auc_score(ym, pm) if len(set(ym)) > 1 else float("nan"),
         "mort_auprc": average_precision_score(ym, pm) if len(set(ym)) > 1 else float("nan"),
         "readmit_auroc": roc_auc_score(yr[m], pr[m]) if m.sum() and len(set(yr[m])) > 1 else float("nan"),
         "readmit_auprc": average_precision_score(yr[m], pr[m]) if m.sum() and len(set(yr[m])) > 1 else float("nan")}
    return r

best_auc, best_state = -1, None
step = 0
for ep in range(A.epochs):
    net.train()
    for enc, ymb, yrb in tr_dl:
        enc = {k: v.to(A.device) for k, v in enc.items()}
        ymb = ymb.to(A.device); yrb = yrb.to(A.device)
        lm, lr_ = net(enc)
        loss = bce(lm, ymb).mean()
        mmask = (yrb >= 0).float()
        if mmask.sum() > 0:
            loss = loss + (bce(lr_, yrb.clamp(min=0)) * mmask).sum() / mmask.sum()
        (loss / A.accum).backward(); step += 1
        if step % A.accum == 0:
            torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0); opt.step(); opt.zero_grad()
    v = evaluate(va_dl)
    print(f"  ep{ep} val mort_auroc={v['mort_auroc']:.4f} readmit_auroc={v['readmit_auroc']:.4f} "
          f"({(time.time()-t0)/60:.1f}m)", flush=True)
    if v["mort_auroc"] > best_auc:
        best_auc = v["mort_auroc"]; best_state = {k: t.clone() for k, t in net.state_dict().items()}

net.load_state_dict(best_state)
test = evaluate(te_dl)
row = {"seed": A.seed, "mode": "clinicalbert", "tag": A.tag, "n_test": len(te_ds), **test}
open(f"{A.out}/results_mtl.jsonl", "a").write(json.dumps(row) + "\n")
print(f"[seed {A.seed}] ClinicalBERT-FT  mort_auroc={test['mort_auroc']:.4f} "
      f"mort_auprc={test['mort_auprc']:.4f} readmit_auroc={test['readmit_auroc']:.4f}  "
      f"({(time.time()-t0)/60:.0f}m)", flush=True)

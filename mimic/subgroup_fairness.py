#!/usr/bin/env python3
"""Disaggregated performance by demographic and administrative subgroup.

race, insurance, language, marital_status and gender enter the model as
categorical inputs (train_mtl.py CATS), so any disparity across those groups is
a property of the reported system. This script reports, per subgroup:
  - C1 mortality AUROC / AUPRC / recall at a 10% review budget
  - the persona augmentation's delta on the same subgroup, paired on seed
It uses the saved per-admission risks of the bootstrap series (bs-base = C1,
bs-persona = design 6), 5 seeds each on the fixed split.
"""
import glob, re, sys
import numpy as np, pandas as pd
from sklearn.metrics import roc_auc_score, average_precision_score

ROOT = "mimic/out"
MIN_N, MIN_POS = 1000, 25

def load(tag):
    out = {}
    for f in sorted(glob.glob(f"{ROOT}/preds/{tag}_s*.parquet")):
        s = int(re.search(r"_s(\d+)", f).group(1))
        out[s] = pd.read_parquet(f).set_index("hadm_id")
    return out

def recall_at(y, r, budget=0.10):
    """Recall inside the group's own top-decile: the convention the paper's
    primary endpoint uses (train_mtl.py _rank_metrics scores each block on its
    own pool). It forces every group to be flagged at exactly the budget, so it
    cannot show an allocation disparity -- for that we also report the flagged
    share and TPR at ONE global threshold, which is what a review budget means
    in deployment."""
    k = max(1, int(round(len(r) * budget)))
    idx = np.argsort(-r)[:k]
    return y[idx].sum() / max(1, y.sum())

def metrics(y, r):
    if y.sum() < 2 or y.sum() == len(y): return None
    return (roc_auc_score(y, r) * 100, average_precision_score(y, r) * 100,
            recall_at(y, r) * 100)

# "declined to answer" (1.7% mortality) is a consent category and is kept apart
# from "unknown / unable to obtain" (12.5%), which is an acuity artifact -- merging
# them manufactures the high-prevalence row the prose would otherwise lean on.
RACE = {"WHITE": "White", "BLACK/AFRICAN AMERICAN": "Black", "OTHER": "Other",
        "HISPANIC/LATINO": "Hispanic", "ASIAN": "Asian",
        "UNKNOWN": "Unknown", "UNABLE TO OBTAIN": "Unknown",
        "PATIENT DECLINED TO ANSWER": "Declined"}
def race_of(v):
    v = str(v)
    if v in RACE: return RACE[v]
    for k, lab in [("WHITE", "White"), ("BLACK", "Black"), ("HISPANIC", "Hispanic"),
                   ("ASIAN", "Asian"), ("AMERICAN INDIAN", "Other"),
                   ("NATIVE HAWAIIAN", "Other"), ("PORTUGUESE", "White"),
                   ("SOUTH AMERICAN", "Hispanic")]:
        if v.startswith(k): return lab
    return "Other"

c = pd.read_parquet(f"{ROOT}/cohort_cens.parquet",
                    columns=["hadm_id", "race", "insurance", "language",
                             "marital_status", "gender"]).set_index("hadm_id")
c["race"] = c["race"].map(race_of)
c["language"] = np.where(c["language"].astype(str).eq("English"), "English", "Non-English")
c["insurance"] = c["insurance"].fillna("(missing)")
c["marital_status"] = c["marital_status"].fillna("(missing)")

base, pers = load("bs-base"), load("bs-persona")
seeds = sorted(set(base) & set(pers))
print(f"seeds: {seeds}", file=sys.stderr)
ref = base[seeds[0]]
meta = c.reindex(ref.index)

rows = []
for attr in ["race", "insurance", "language", "marital_status", "gender"]:
    for lvl, sub in meta.groupby(attr, sort=False):
        idx = sub.index
        y = ref.loc[idx, "y_mort"].values.astype(int)
        if len(idx) < MIN_N or y.sum() < MIN_POS: continue
        b = np.array([metrics(y, base[s].loc[idx, "risk"].values) for s in seeds], float)
        p = np.array([metrics(y, pers[s].loc[idx, "risk"].values) for s in seeds], float)
        d = p - b
        # deployment view: ONE global top-decile cut, plus subgroup calibration
        g_tpr, g_flag, cal = [], [], []
        for s in seeds:
            r = base[s]["risk"]; thr = np.quantile(r.values, 1 - 0.10)
            sel = r.loc[idx].values >= thr
            g_tpr.append(100 * y[sel].sum() / max(1, y.sum()))
            g_flag.append(100 * sel.mean())
            cal.append((1 / (1 + np.exp(-r.loc[idx].values))).mean() / max(y.mean(), 1e-9))
        rows.append(dict(attr=attr, level=lvl, n=len(idx), deaths=int(y.sum()),
                         prev=100 * y.mean(),
                         auroc=b[:, 0].mean(), auprc=b[:, 1].mean(), r10=b[:, 2].mean(),
                         cal=np.mean(cal), flag=np.mean(g_flag), tpr=np.mean(g_tpr),
                         d_auroc=d[:, 0].mean(), d_auprc=d[:, 1].mean(), d_r10=d[:, 2].mean()))
t = pd.DataFrame(rows)
pd.set_option("display.width", 200, "display.max_rows", 100)
print(t.to_string(index=False, float_format=lambda v: f"{v:7.2f}"))
print(f"\nsubgroups reported: {len(t)} (>= {MIN_N} admissions and >= {MIN_POS} deaths)")
print(f"persona AUROC delta: min {t.d_auroc.min():+.2f}, max {t.d_auroc.max():+.2f}, "
      f"negative in {(t.d_auroc < 0).sum()} of {len(t)}")
print(f"persona R@10% delta: min {t.d_r10.min():+.2f}, max {t.d_r10.max():+.2f}, "
      f"negative in {(t.d_r10 < 0).sum()} of {len(t)}")
print(f"C1 AUROC spread across subgroups: {t.auroc.min():.2f} to {t.auroc.max():.2f}")
print(f"at one global 10% cut: flagged {t.flag.min():.1f}%--{t.flag.max():.1f}%, "
      f"TPR {t.tpr.min():.1f}--{t.tpr.max():.1f}")
print(f"calibration (predicted/observed): {t.cal.min():.2f}--{t.cal.max():.2f}")
print(f"|delta AUROC| >= 0.05 in {(t.d_auroc.abs() >= 0.05).sum()} of {len(t)}; "
      f"negative and >= 0.05 in {((t.d_auroc <= -0.05)).sum()}")

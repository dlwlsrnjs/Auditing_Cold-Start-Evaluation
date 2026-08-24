#!/usr/bin/env python3
"""First-48h lab features per admission — the severity signal the model has been missing.

Streams labevents.csv.gz (2.6GB gz), keeps rows inside [admittime, admittime+48h] for cohort
admissions, and aggregates the 24 highest-coverage analytes into last/min/max features
(+ abnormal-flag count and n_labs). Diagnosis-agnostic severity information -> expected to help
cold-dx blocks disproportionately.

Output: mimic/out/labs48.parquet (hadm_id + ~74 features)
"""
import os, time
import numpy as np, pandas as pd

OUT = "mimic/out"
WIN = float(os.environ.get("LAB_WINDOW_H", "48"))
SUF = "" if WIN == 48 else f"_{int(WIN)}h"
LAB = "mimic/physionet.org/files/mimiciv/3.1/hosp/labevents.csv.gz"

# 24 core analytes (MIMIC-IV itemids), chosen for coverage + clinical severity relevance
ITEMS = {
    50912: "creatinine", 50971: "potassium", 50983: "sodium", 50902: "chloride",
    50882: "bicarbonate", 51006: "bun", 50931: "glucose", 50893: "calcium",
    51221: "hematocrit", 51222: "hemoglobin", 51265: "platelets", 51301: "wbc",
    51248: "mch", 51277: "rdw", 50960: "magnesium", 50970: "phosphate",
    51237: "inr", 51274: "pt", 51275: "ptt", 50813: "lactate",
    50885: "bilirubin_total", 50861: "alt", 50878: "ast", 50862: "albumin",
}
t0 = time.time()
adm = pd.read_parquet(f"{OUT}/cohort.parquet", columns=["hadm_id", "admittime"])
adm_t = adm.set_index("hadm_id").admittime
print(f"cohort admissions: {len(adm):,}", flush=True)

acc = {}   # (hadm_id) -> {name: [values in time order], "_abn": count, "_n": count}
reader = pd.read_csv(LAB, chunksize=2_000_000,
                     usecols=["hadm_id", "itemid", "charttime", "valuenum", "flag"])
seen = 0
for ch in reader:
    seen += len(ch)
    ch = ch.dropna(subset=["hadm_id", "valuenum"])
    ch = ch[ch.itemid.isin(ITEMS)]
    ch["hadm_id"] = ch.hadm_id.astype(int)
    ch = ch[ch.hadm_id.isin(adm_t.index)]
    if not len(ch):
        if seen % 20_000_000 == 0: print(f"  scanned {seen/1e6:.0f}M ({time.time()-t0:.0f}s)", flush=True)
        continue
    ch["charttime"] = pd.to_datetime(ch.charttime)
    at = adm_t.loc[ch.hadm_id].values
    dt = (ch.charttime.values - at) / np.timedelta64(1, "h")
    ch = ch[(dt >= -6) & (dt <= WIN)]                    # small pre-admission ED window included
    for h, it, v, fl in zip(ch.hadm_id, ch.itemid, ch.valuenum, ch.flag):
        a = acc.setdefault(h, {"_abn": 0, "_n": 0})
        a.setdefault(ITEMS[it], []).append(v)
        a["_n"] += 1
        if isinstance(fl, str): a["_abn"] += 1
    if seen % 20_000_000 == 0:
        print(f"  scanned {seen/1e6:.0f}M rows, kept adms={len(acc):,} ({time.time()-t0:.0f}s)", flush=True)

rows = []
for h, a in acc.items():
    r = {"hadm_id": h, "lab_n": a["_n"], "lab_abn_frac": a["_abn"] / max(a["_n"], 1)}
    for name in ITEMS.values():
        vs = a.get(name)
        if vs:
            r[f"{name}_last"] = vs[-1]; r[f"{name}_min"] = min(vs); r[f"{name}_max"] = max(vs)
    rows.append(r)
labs = pd.DataFrame(rows)
labs.to_parquet(f"{OUT}/labs48{SUF}.parquet", index=False)
cov = len(labs) / len(adm)
print(f"[saved] labs48{SUF}.parquet {labs.shape}  coverage={cov*100:.1f}% of admissions "
      f"({(time.time()-t0)/60:.0f}m)", flush=True)

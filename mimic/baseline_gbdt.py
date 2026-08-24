#!/usr/bin/env python3
"""XGBoost baseline on the same cohort, splits, and features as train_mtl.py.

The reviewer-expected strong tabular baseline: one GBDT per task head, trained on the identical
structured features (demographics, admission fields, history block, train-only dx frequency,
ICD-3 target-frequency encoding) plus optional mean-pooled note/dx-text embeddings reduced by PCA.
Same patient-level split + strict cold-item protocol; same cold-block breakdown output.

Usage: python3 mimic/baseline_gbdt.py --seed 42 [--use-text]
"""
import argparse, json, time
import numpy as np, pandas as pd, xgboost as xgb
from sklearn.metrics import roc_auc_score, average_precision_score
from sklearn.decomposition import PCA

ap = argparse.ArgumentParser()
ap.add_argument("--cohort", default="mimic/out/cohort.parquet")
ap.add_argument("--out", default="mimic/out")
ap.add_argument("--seed", type=int, default=42)
ap.add_argument("--use-text", action="store_true")
ap.add_argument("--pca", type=int, default=64)
ap.add_argument("--tag", default="")
A = ap.parse_args()
np.random.seed(A.seed); t0 = time.time()

d = pd.read_parquet(A.cohort)

# ---- identical split protocol to train_mtl.py ----
rng = np.random.default_rng(A.seed)
pids = d.subject_id.unique(); rng.shuffle(pids)
n = len(pids); tr_p = set(pids[:int(.7 * n)]); va_p = set(pids[int(.7 * n):int(.8 * n)])
d["split"] = np.where(d.subject_id.isin(tr_p), "train",
             np.where(d.subject_id.isin(va_p), "val", "test"))
d.loc[(d.split == "train") & (d.cold_item_rare_dx == 1), "split"] = "drop"

# ---- identical history block ----
d = d.sort_values(["subject_id", "admittime"]).reset_index(drop=True)
g = d.groupby("subject_id")
d["prev_dischtime"] = g.dischtime.shift(1)
d["days_since_last_discharge"] = (d.admittime - d.prev_dischtime).dt.total_seconds() / 86400.0
d["prior_los_mean"] = g.y_los_days.transform(lambda s: s.shift(1).expanding().mean())
HIST = ["n_prior_admissions", "days_since_last_discharge", "prior_los_mean"]

CATS = ["gender", "race", "insurance", "language", "marital_status",
        "admission_type", "admission_location"]
d["icd3"] = d.primary_icd.astype(str).str.strip().str[:3]
tr = d[d.split == "train"]
feats = [d.age.values.astype(np.float32)]
names = ["age"]
for c in HIST:
    feats.append(d[c].fillna(-1).values.astype(np.float32)); names.append(c)
for c in CATS:                                   # ordinal-encode on train vocab
    vocab = {v: i + 1 for i, v in enumerate(sorted(tr[c].dropna().unique()))}
    feats.append(d[c].map(vocab).fillna(0).values.astype(np.float32)); names.append(c)
tr_freq = tr.primary_icd.astype(str).value_counts()
feats.append(np.log1p(d.primary_icd.astype(str).map(tr_freq).fillna(0.0)).values.astype(np.float32))
names.append("dx_freq_train")
i3 = {v: i + 1 for i, v in enumerate(sorted(tr.icd3.dropna().unique()))}
feats.append(d.icd3.map(i3).fillna(0).values.astype(np.float32)); names.append("icd3_id")
# target-frequency encodings of icd3 computed on TRAIN only
for lbl, col in [("mort", "y_mortality"), ("los", "y_los_days")]:
    m = tr.groupby("icd3")[col].mean()
    feats.append(d.icd3.map(m).fillna(m.mean()).values.astype(np.float32)); names.append(f"icd3_{lbl}_te")

X = np.stack(feats, 1)
if A.use_text:
    for emb_p, ids_p, key in [
        (f"{A.out}/note_emb.npy", f"{A.out}/note_emb_ids.parquet", "hadm_id"),
        (f"{A.out}/dx_text_emb_gloss.npy", f"{A.out}/dx_text_ids_gloss.parquet", "primary_icd")]:
        emb = np.load(emb_p); ids = pd.read_parquet(ids_p)[key]
        if key == "primary_icd":
            pos = {str(c): i for i, c in enumerate(ids.astype(str).values)}
            idx = d.primary_icd.astype(str).map(pos)
        else:
            pos = {h: i for i, h in enumerate(ids.values)}
            idx = d.hadm_id.map(pos)
        M = np.zeros((len(d), emb.shape[1]), np.float32)
        ok = idx.notna().values; M[ok] = emb[idx[ok].astype(int).values]
        pca = PCA(n_components=A.pca, random_state=A.seed).fit(M[d.split == "train"])
        X = np.concatenate([X, pca.transform(M).astype(np.float32),
                            ok.astype(np.float32)[:, None]], 1)

itr = np.where(d.split == "train")[0]; iva = np.where(d.split == "val")[0]
ite = np.where(d.split == "test")[0]
print(f"features={X.shape[1]} train={len(itr):,} test={len(ite):,} text={A.use_text}", flush=True)

def _auc(y, p): return float(roc_auc_score(y, p)) if len(np.unique(y)) > 1 else float("nan")
def _ap(y, p): return float(average_precision_score(y, p)) if len(np.unique(y)) > 1 else float("nan")

res = {}
PAR = dict(max_depth=7, n_estimators=600, learning_rate=0.06, subsample=0.9,
           colsample_bytree=0.8, tree_method="hist", random_state=A.seed, n_jobs=16,
           early_stopping_rounds=40)
# mortality
ym = d.y_mortality.values
m1 = xgb.XGBClassifier(**PAR, eval_metric="auc",
                       scale_pos_weight=(ym[itr] == 0).sum() / max((ym[itr] == 1).sum(), 1))
m1.fit(X[itr], ym[itr], eval_set=[(X[iva], ym[iva])], verbose=False)
pm = m1.predict_proba(X)[:, 1]
res["mort_auroc"], res["mort_auprc"] = _auc(ym[ite], pm[ite]), _ap(ym[ite], pm[ite])
# readmission (observed rows only)
yr = d.y_readmit_30d.values; mr = ~np.isnan(yr)
jtr = itr[mr[itr]]; jva = iva[mr[iva]]; jte = ite[mr[ite]]
m2 = xgb.XGBClassifier(**PAR, eval_metric="auc")
m2.fit(X[jtr], yr[jtr], eval_set=[(X[jva], yr[jva])], verbose=False)
pr = m2.predict_proba(X)[:, 1]
res["readmit_auroc"], res["readmit_auprc"] = _auc(yr[jte], pr[jte]), _ap(yr[jte], pr[jte])
# LOS
yl = np.log1p(d.y_los_days.values)
m3 = xgb.XGBRegressor(**{k: v for k, v in PAR.items() if k != "early_stopping_rounds"},
                      early_stopping_rounds=40, eval_metric="rmse")
m3.fit(X[itr], yl[itr], eval_set=[(X[iva], yl[iva])], verbose=False)
pl = np.expm1(m3.predict(X)); lt = np.expm1(yl)
res["los_rmse"] = float(np.sqrt(np.mean((lt[ite] - pl[ite]) ** 2)))
res["los_mae"] = float(np.mean(np.abs(lt[ite] - pl[ite])))

blocks = {}
sub = d.iloc[ite]
for un, u in [("warmU", 0), ("coldU", 1)]:
    for iname, i in [("warmD", 0), ("coldD", 1)]:
        s = ite[np.where((sub.cold_user_first_admission.values == u) &
                         (sub.cold_item_rare_dx.values == i))[0]]
        if len(s) < 50: continue
        sm = s[~np.isnan(yr[s])]
        blocks[f"{un}_{iname}"] = dict(
            n=int(len(s)), mort_auroc=_auc(ym[s], pm[s]),
            readmit_auroc=_auc(yr[sm], pr[sm]) if len(sm) else float("nan"),
            los_mae=float(np.mean(np.abs(lt[s] - pl[s]))))

row = {"seed": A.seed, "mode": "gbdt", "text": A.use_text, "tag": A.tag or
       ("gbdt-text" if A.use_text else "gbdt"), **res, "blocks": blocks}
open(f"{A.out}/results_mtl.jsonl", "a").write(json.dumps(row) + "\n")
print(f"[seed {A.seed}] GBDT text={A.use_text}  mort={res['mort_auroc']:.4f}/"
      f"{res['mort_auprc']:.4f}  readmit={res['readmit_auroc']:.4f}  "
      f"los_mae={res['los_mae']:.3f}  ({time.time()-t0:.0f}s)", flush=True)
for k, v in blocks.items():
    print(f"    {k:<12} n={v['n']:>7,}  mort={v['mort_auroc']:.4f}  "
          f"readm={v['readmit_auroc']:.4f}  losMAE={v['los_mae']:.3f}")

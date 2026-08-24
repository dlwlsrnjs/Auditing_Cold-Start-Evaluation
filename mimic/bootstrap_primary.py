#!/usr/bin/env python3
"""Test-set bootstrap interval for the pre-specified primary endpoint (CC-segment R@10%).

Reviewer O5(ii): the paper's "95% CI [+2.2,+4.2]" is a 5-seed interval and measures
initialization variance only; a reader will read it as a sampling interval. This computes the
sampling-variance component by resampling TEST ADMISSIONS (paired across the two models, so the
same bootstrap sample scores both), which is the interval that actually bounds the claim.
"""
import argparse, glob, os
import numpy as np, pandas as pd

ap = argparse.ArgumentParser()
ap.add_argument("--a", required=True, help="glob of prediction parquets for arm A")
ap.add_argument("--b", required=True, help="glob of prediction parquets for arm B (reference)")
ap.add_argument("--segment", default="cc", choices=["cc", "wc", "cw", "ww", "all"])
ap.add_argument("--budget", type=float, default=0.10)
ap.add_argument("--boot", type=int, default=5000)
ap.add_argument("--seed", type=int, default=0)
A = ap.parse_args()

def load(pat):
    fs = sorted(glob.glob(pat))
    assert fs, f"no files match {pat}"
    dfs = [pd.read_parquet(f).sort_values("hadm_id").reset_index(drop=True) for f in fs]
    base = dfs[0]
    risk = np.mean([d.risk.values for d in dfs], 0)      # seed-averaged risk = the reported model
    return base.hadm_id.values, risk, base.y_mort.values, base.cold_user.values, base.cold_item.values, len(fs)

ha, ra, ya, cua, cia, na = load(A.a)
hb, rb, yb, cub, cib, nb = load(A.b)
assert np.array_equal(ha, hb), "the two arms are not on the same test rows"
assert np.array_equal(ya, yb)

mask = {"cc": (cua == 1) & (cia == 1), "wc": (cua == 0) & (cia == 1),
        "cw": (cua == 1) & (cia == 0), "ww": (cua == 0) & (cia == 0),
        "all": np.ones(len(ya), bool)}[A.segment]
y = ya[mask]; xa = ra[mask]; xb = rb[mask]
n, npos = len(y), int(y.sum())

def recall_at(risk, yy, budget):
    k = max(1, int(round(len(yy) * budget)))
    top = np.argpartition(-risk, k - 1)[:k]
    return yy[top].sum() / max(1, yy.sum())

pa, pb = recall_at(xa, y, A.budget), recall_at(xb, y, A.budget)
rng = np.random.default_rng(A.seed)
diffs = np.empty(A.boot)
for i in range(A.boot):
    idx = rng.integers(0, n, n)                 # paired resample: same rows score both arms
    yy = y[idx]
    if yy.sum() == 0: diffs[i] = 0.0; continue
    diffs[i] = recall_at(xa[idx], yy, A.budget) - recall_at(xb[idx], yy, A.budget)
diffs *= 100
lo, hi = np.percentile(diffs, [2.5, 97.5])
p_two = 2 * min((diffs <= 0).mean(), (diffs >= 0).mean())

print(f"segment={A.segment}  n={n:,} admissions, {npos} deaths, budget={A.budget:.0%} "
      f"({max(1,int(round(n*A.budget))):,} charts)")
print(f"arm A ({na} seeds averaged): R@{A.budget:.0%} = {pa:.4f}")
print(f"arm B ({nb} seeds averaged): R@{A.budget:.0%} = {pb:.4f}")
print(f"difference {100*(pa-pb):+.2f} pt")
print(f"test-set bootstrap 95% CI [{lo:+.2f}, {hi:+.2f}] pt over {A.boot:,} paired resamples, "
      f"two-sided p={p_two:.3f}")
print(f"in deaths at this budget: {100*(pa-pb)/100*npos:+.1f} of {npos} "
      f"(CI [{lo/100*npos:+.1f}, {hi/100*npos:+.1f}])")

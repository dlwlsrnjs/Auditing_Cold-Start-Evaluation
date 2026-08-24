#!/usr/bin/env python3
"""Rebuild the readmission label with an administratively correct censoring definition.

The original build flagged EVERY last admission of a surviving patient as censored. That
conflates two different things: an outcome that could not be observed because follow-up ran
out, and an outcome that was observed to be negative within the data source. MIMIC-IV shifts
each patient's dates into a random future year, so there is no shared calendar cutoff -- but
`patients.anchor_year_group` gives the real 3-year window containing the patient's anchor year,
and all of that patient's dates carry the same offset. That is enough to bound, conservatively,
how much real follow-up existed after each discharge.

A last admission is censored only if we cannot guarantee >= FU_YEARS of follow-up after
discharge under the least favourable placement of the anchor inside its group.
"""
import argparse
import numpy as np, pandas as pd

ap = argparse.ArgumentParser()
ap.add_argument("--cohort", default="mimic/out/cohort.parquet")
ap.add_argument("--patients", default="mimic/physionet.org/files/mimiciv/3.1/hosp/patients.csv.gz")
ap.add_argument("--out", default="mimic/out/cohort_cens.parquet")
ap.add_argument("--collection-end", type=int, default=2022)
ap.add_argument("--fu-years", type=float, default=1.0)
A = ap.parse_args()

d = pd.read_parquet(A.cohort)
p = pd.read_csv(A.patients, usecols=["subject_id", "anchor_year", "anchor_year_group"])
p["grp_end"] = p.anchor_year_group.str.split(" - ").str[1].astype(int)
d = d.merge(p[["subject_id", "anchor_year", "grp_end"]], on="subject_id", how="left")

# conservative: assume the real anchor sat at the END of its group, so the discharge is as late
# as the group allows and the remaining follow-up as short as possible
yrs_since_anchor = d.dischtime.dt.year - d.anchor_year
followup_yrs = A.collection_end - (d.grp_end + yrs_since_anchor)

alive = d.y_mortality == 0
old = d.readmit_censored.astype(bool)
new = (d.is_last_admission == 1) & alive & (followup_yrs < A.fu_years)

# rows that were called censored but in fact had the follow-up to observe a negative
recovered = old & ~new
d.loc[recovered, "y_readmit_30d"] = 0.0
if "y_readmit_30d_unplanned" in d.columns:
    d.loc[recovered, "y_readmit_30d_unplanned"] = 0.0
d.loc[new, ["y_readmit_30d"] + (["y_readmit_30d_unplanned"] if "y_readmit_30d_unplanned" in d.columns else [])] = np.nan

d["readmit_censored_old"] = old.astype(int)
d["readmit_censored"] = new.astype(int)
d["followup_years_min"] = followup_yrs
d = d.drop(columns=["anchor_year", "grp_end"])
d.to_parquet(A.out, index=False)

s = d[alive]
cu = s.cold_user_first_admission == 1
print(f"admissions {len(d):,}; survivors {len(s):,}")
print(f"censored: {int(old.sum()):,} ({old[alive].mean()*100:.1f}%) -> "
      f"{int(new.sum()):,} ({new[alive].mean()*100:.1f}%)")
print(f"  recovered as observed negatives: {int(recovered.sum()):,}")
print(f"censoring rate  cold-user {s.readmit_censored[cu].mean()*100:.1f}%  "
      f"warm-user {s.readmit_censored[~cu].mean()*100:.1f}%  "
      f"ratio {s.readmit_censored[cu].mean()/s.readmit_censored[~cu].mean():.2f}")
y = s.y_readmit_30d
print(f"readmission rate  cold {y[cu].mean()*100:.2f}%  warm {y[~cu].mean()*100:.2f}%  "
      f"gap {(y[~cu].mean()-y[cu].mean())*100:.2f}")
print(f"-> {A.out}")

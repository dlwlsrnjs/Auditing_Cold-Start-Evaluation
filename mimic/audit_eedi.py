#!/usr/bin/env python3
"""Audit 1 (label observability) on a public non-clinical log: Eedi/NeurIPS 2020
Education Challenge student-answer log.

Mirrors the paper's construction with no model training:
  entity   = student            (cf. patient)
  episode  = a student's activity day, >=1 answer      (cf. admission)
  item     = the first question answered that day      (cf. principal ICD code)
  label    = the student returns within 30 days        (cf. 30-day readmission)
  censored = the 30-day window does not close inside the log
  cold-user = the student's first activity day
  cold-item = the day's first question is in the rarest 20% of answer mass

Reports, per segment, the three quantities the audit needs -- censoring rate,
zero-imputed label rate, observed label rate -- and the closed form
  p_w q_w - p_c q_c = p_c q_w (r_p - r_q),   r_p = p_w/p_c,  r_q = q_c/q_w.
Also prices the mis-definition the paper warns about ("the record ends, so it
is censored") against the administrative one.
"""
import pandas as pd, numpy as np, sys

ROOT = "eedi/data"
W = 30  # forward window, days

print("loading answers ...", flush=True)
ans = pd.read_csv(f"{ROOT}/train_data/train_task_1_2.csv",
                  usecols=["QuestionId", "UserId", "AnswerId"],
                  dtype={"QuestionId": "int32", "UserId": "int32", "AnswerId": "int64"})
meta = pd.read_csv(f"{ROOT}/metadata/answer_metadata_task_1_2.csv",
                   usecols=["AnswerId", "DateAnswered"])
meta["ts"] = pd.to_datetime(meta["DateAnswered"], errors="coerce")
meta = meta.dropna(subset=["ts", "AnswerId"])
meta["AnswerId"] = meta["AnswerId"].astype("int64")
meta = meta[["AnswerId", "ts"]]
d = ans.merge(meta, on="AnswerId", how="inner")
del ans, meta
d["day"] = d["ts"].dt.floor("D")
print(f"  {len(d):,} timestamped answers, {d.UserId.nunique():,} students, "
      f"{d.QuestionId.nunique():,} questions", flush=True)
print(f"  log spans {d.day.min().date()} .. {d.day.max().date()}", flush=True)

# rare items: bottom 20% of answer mass, as in the paper's cold-item definition
cnt = d.QuestionId.value_counts().sort_values()
rare = set(cnt.index[cnt.cumsum() <= 0.20 * cnt.sum()])
print(f"  rare questions: {len(rare):,} of {len(cnt):,} "
      f"({len(rare)/len(cnt):.1%} of items, 20% of answer mass)", flush=True)

# episodes = student activity days; principal item = first question of the day
d = d.sort_values(["UserId", "ts"], kind="mergesort")
ep = (d.groupby(["UserId", "day"], sort=True)
        .agg(first_q=("QuestionId", "first")).reset_index())
ep = ep.sort_values(["UserId", "day"], kind="mergesort").reset_index(drop=True)
print(f"  {len(ep):,} episodes (student-days)", flush=True)

nxt = ep.groupby("UserId")["day"].shift(-1)
END = d.day.max()
ep["is_last"]    = nxt.isna()
ep["returned"]   = (nxt - ep["day"]).dt.days.le(W).fillna(False)
ep["window_open"] = (END - ep["day"]).dt.days < W          # administrative censoring
ep["cens_admin"] = ep["is_last"] & ep["window_open"]
ep["cens_naive"] = ep["is_last"]                            # "the record ends" rule
ep["cold_user"]  = ep.groupby("UserId").cumcount() == 0
ep["cold_item"]  = ep["first_q"].isin(rare)

def three(sub, cens):
    q = sub[cens].mean()
    obs = sub.loc[~sub[cens], "returned"].mean()
    zero = sub["returned"].where(~sub[cens], False).mean()
    return q, obs, zero

def audit(name, mask_cold, mask_warm, cens):
    c, w = ep[mask_cold], ep[mask_warm]
    q_c, p_c, z_c = three(c, cens)
    q_w, p_w, z_w = three(w, cens)
    gap_obs, gap_zero = (p_w - p_c) * 100, (z_w - z_c) * 100
    r_p, r_q = p_w / p_c, q_c / q_w
    # paper's convention: distortion D = p_w q_w - p_c q_c = p_c q_w (r_p - r_q),
    # and the zero-imputed gap is (observed gap - D).  D > 0 attenuates, D < 0 inflates.
    D_meas   = gap_obs - gap_zero
    D_closed = p_c * q_w * (r_p - r_q) * 100
    print(f"\n--- {name}  (censoring rule: {cens}) ---")
    print(f"  n cold {len(c):,}   n warm {len(w):,}")
    print(f"  censoring rate            cold {q_c:7.3%}   warm {q_w:7.3%}   r_q={r_q:.3f}")
    print(f"  return rate, observed     cold {p_c:7.3%}   warm {p_w:7.3%}   r_p={r_p:.3f}")
    print(f"  return rate, zero-imputed cold {z_c:7.3%}   warm {z_w:7.3%}")
    print(f"  warm-cold gap  observed {gap_obs:+.2f} pt   zero-imputed {gap_zero:+.2f} pt")
    print(f"  distortion D  measured {D_meas:+.2f} pt   closed form {D_closed:+.2f} pt"
          f"   ({'attenuates' if D_meas > 0 else 'inflates'} the gap)")
    print(f"  reported/observed = {gap_zero/gap_obs:+.3f}x")
    return dict(gap_obs=gap_obs, gap_zero=gap_zero, r_p=r_p, r_q=r_q)

for cens in ["cens_admin", "cens_naive"]:
    audit("user axis  (first activity day vs later)",
          ep.cold_user, ~ep.cold_user, cens)
    audit("item axis  (rare first question vs common)",
          ep.cold_item, ~ep.cold_item, cens)

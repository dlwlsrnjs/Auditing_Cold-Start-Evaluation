#!/usr/bin/env python3
"""Build the ColdHTL/MIMIC cohort from the LOCAL PhysioNet CSVs (no BigQuery).

Implements mimic/SETUP.md section B:
  unit           = hospital admission (hadm_id), adults, LOS > 0
  hybrid target  = y_mortality (primary binary)
                   -> y_readmit_30d (dependent binary, NULL if the patient died = the
                      conditional-observation funnel; only survivors can be readmitted)
                   -> y_los_days (continuous core)
  cold flags     = cold_user_first_admission (no prior admission = new user)
                   cold_item_rare_dx        (rare primary diagnosis = new/rare item)
  early text     = radiology notes charted within --window-hours of admission.
                   Discharge summaries are EXCLUDED from predictors: they are written at
                   discharge and leak mortality / LOS / disposition (SETUP.md section B).

Outputs (to --out):
  cohort.parquet        one row per admission: labels, cold flags, covariates
  notes_early.parquet   admission-linked early radiology text (the simulator's input)
  notes_discharge.parquet  only with --include-discharge; simulator SFT *target* text, never a predictor
  eda_report.txt        counts, label rates, funnel sizes, cold-block sizes

Usage:
  python3 mimic/build_cohort.py --root mimic/physionet.org/files --out mimic/out
"""
import argparse, os, sys, time
import numpy as np, pandas as pd

ap = argparse.ArgumentParser()
ap.add_argument("--root", default="mimic/physionet.org/files",
                help="directory holding mimiciv/3.1 and mimic-iv-note/2.2")
ap.add_argument("--out", default="mimic/out")
ap.add_argument("--min-age", type=int, default=18)
ap.add_argument("--window-hours", type=float, default=48.0, help="early-note window after admittime")
ap.add_argument("--readmit-days", type=int, default=30)
ap.add_argument("--rare-quantile", type=float, default=0.20,
                help="primary-dx codes whose admission frequency falls in this bottom quantile "
                     "of admissions are flagged cold_item_rare_dx")
ap.add_argument("--readmit-basis", choices=["uncensored", "all"], default="uncensored",
                help="uncensored (default): the readmission label is NaN on a patient's last "
                     "admission, where follow-up is unobserved. 'all' keeps the naive 0 there.")
ap.add_argument("--drop-censored-rows", action="store_true",
                help="additionally drop censored admissions from the cohort entirely "
                     "(also discards their fully-observed mortality/LOS labels)")
ap.add_argument("--max-note-chars", type=int, default=4000)
ap.add_argument("--chunksize", type=int, default=50000)
ap.add_argument("--include-discharge", action="store_true",
                help="also dump discharge summaries (SFT target only - NOT a predictor)")
ap.add_argument("--no-notes", action="store_true", help="labels only; skip the note pass")
A = ap.parse_args()

MIMIC = f"{A.root}/mimiciv/3.1"
NOTE = f"{A.root}/mimic-iv-note/2.2/note"
os.makedirs(A.out, exist_ok=True)
log_lines = []
def log(m):
    print(m, flush=True); log_lines.append(m)

def need(p):
    if not os.path.exists(p): sys.exit(f"missing required file: {p}\n(still downloading?)")
    return p

t0 = time.time()

# ----------------------------------------------------------------------------- admissions
adm = pd.read_csv(need(f"{MIMIC}/hosp/admissions.csv.gz"),
                  parse_dates=["admittime", "dischtime", "deathtime"],
                  usecols=["subject_id", "hadm_id", "admittime", "dischtime", "deathtime",
                           "admission_type", "admission_location", "discharge_location",
                           "insurance", "language", "marital_status", "race",
                           "edregtime", "hospital_expire_flag"])
pat = pd.read_csv(need(f"{MIMIC}/hosp/patients.csv.gz"),
                  usecols=["subject_id", "gender", "anchor_age", "anchor_year", "dod"],
                  parse_dates=["dod"])
log(f"raw admissions={len(adm):,}  patients={len(pat):,}")

df = adm.merge(pat, on="subject_id", how="inner")
# MIMIC-IV ages are anchored: age at admission = anchor_age + (admit year - anchor year)
df["age"] = df.anchor_age + (df.admittime.dt.year - df.anchor_year)
df["los_days"] = (df.dischtime - df.admittime).dt.total_seconds() / 86400.0

n0 = len(df)
df = df[df.age >= A.min_age]
n1 = len(df)
df = df[df.los_days > 0]
n2 = len(df)
log(f"filter adults(age>={A.min_age}): {n0:,} -> {n1:,}   LOS>0: {n1:,} -> {n2:,}")

df = df.sort_values(["subject_id", "admittime"]).reset_index(drop=True)

# ----------------------------------------------------------- hybrid target: the funnel
# 1) primary binary: in-hospital mortality
df["y_mortality"] = df.hospital_expire_flag.astype(int)

# 2) dependent binary: 30-day readmission. Defined ONLY for survivors -> NaN otherwise.
#    This is the sample-selection funnel that motivates the dependence chain.
df["next_admittime"] = df.groupby("subject_id").admittime.shift(-1)
df["next_admission_type"] = df.groupby("subject_id").admission_type.shift(-1)
gap = (df.next_admittime - df.dischtime).dt.total_seconds() / 86400.0
df["days_to_next_admit"] = gap
within = (gap >= 0) & (gap <= A.readmit_days)
df["y_readmit_30d"] = np.where(within, 1.0, 0.0)
# unplanned variant: elective / surgical-same-day readmissions are not the outcome of interest
elective = df.next_admission_type.astype(str).str.contains("ELECTIVE|SURGICAL SAME DAY", case=False, na=False)
df["y_readmit_30d_unplanned"] = np.where(within & ~elective, 1.0, 0.0)
died = df.y_mortality == 1
df.loc[died, ["y_readmit_30d", "y_readmit_30d_unplanned"]] = np.nan   # <- conditional observation

# competing risk: died after discharge inside the readmission window (readmission unobservable)
post = (df.dod - df.dischtime).dt.total_seconds() / 86400.0
df["died_within_window_post_discharge"] = ((post >= 0) & (post <= A.readmit_days)).astype(int)
df.loc[died, "died_within_window_post_discharge"] = 0

# RIGHT-CENSORING. A patient's LAST admission in the database has unknown follow-up: a 0 there may
# mean "not readmitted" or "readmitted after the record ends". This matters because it is not
# spread evenly across the cold blocks - a first admission is also the patient's *only* admission
# for single-admission patients, so cold_user rows are disproportionately censored and their
# readmission rate is mechanically deflated. Flag it instead of silently baking it into the label.
df["is_last_admission"] = (df.groupby("subject_id").cumcount(ascending=False) == 0).astype(int)
df["readmit_censored"] = ((df.is_last_admission == 1) & (~died)).astype(int)

# Chosen convention: UNCENSORED basis. The readmission head is a second conditional-observation
# channel on top of mortality - it is observed only when the patient survived AND the record
# extends past the window. Mortality and LOS remain fully observed on those rows, so the rows stay.
if A.readmit_basis == "uncensored":
    cens = df.readmit_censored == 1
    df.loc[cens, ["y_readmit_30d", "y_readmit_30d_unplanned"]] = np.nan
    log(f"readmit basis=uncensored: y_readmit set NaN on {int(cens.sum()):,} censored admissions "
        f"({cens.mean()*100:.1f}%); y_mortality / y_los_days kept on those rows")
if A.drop_censored_rows:
    n_before = len(df); df = df[df.readmit_censored == 0].reset_index(drop=True)
    log(f"--drop-censored-rows: {n_before:,} -> {len(df):,} admissions")

# 3) continuous core
df["y_los_days"] = df.los_days

# ----------------------------------------------------------------- cold-start flags
# cold user: the patient's first admission in the database (no prior history to learn from)
df["admission_rank"] = df.groupby("subject_id").cumcount() + 1
df["n_prior_admissions"] = df.admission_rank - 1
df["cold_user_first_admission"] = (df.admission_rank == 1).astype(int)

# cold item: rare primary diagnosis
dx = pd.read_csv(need(f"{MIMIC}/hosp/diagnoses_icd.csv.gz"),
                 usecols=["hadm_id", "seq_num", "icd_code", "icd_version"])
prim = dx[dx.seq_num == 1].drop_duplicates("hadm_id")
prim["dx_key"] = prim.icd_version.astype(str) + ":" + prim.icd_code.astype(str).str.strip()
df = df.merge(prim[["hadm_id", "dx_key", "icd_code", "icd_version"]]
              .rename(columns={"icd_code": "primary_icd", "icd_version": "primary_icd_version"}),
              on="hadm_id", how="left")
freq = df.dx_key.value_counts()
df["dx_freq"] = df.dx_key.map(freq)
# threshold on the ADMISSION distribution: the bottom `rare-quantile` share of admissions
thr = df.dx_freq.quantile(A.rare_quantile)
df["cold_item_rare_dx"] = ((df.dx_freq <= thr) & df.dx_key.notna()).astype(int)
log(f"primary dx present for {df.dx_key.notna().mean()*100:.1f}% of admissions; "
    f"rare-dx freq threshold <= {thr:.0f} admissions/code")

try:
    dd = pd.read_csv(f"{MIMIC}/hosp/d_icd_diagnoses.csv.gz")
    dd["dx_key"] = dd.icd_version.astype(str) + ":" + dd.icd_code.astype(str).str.strip()
    df = df.merge(dd[["dx_key", "long_title"]].drop_duplicates("dx_key"), on="dx_key", how="left")
    df = df.rename(columns={"long_title": "primary_dx_title"})
except FileNotFoundError:
    df["primary_dx_title"] = np.nan

# ------------------------------------------------------------------- early notes
notes_early = None
if not A.no_notes:
    win = pd.Timedelta(hours=A.window_hours)
    windows = df[["subject_id", "hadm_id", "admittime", "dischtime"]].copy()
    windows["win_end"] = windows.admittime + win
    # a note may fall inside more than one admission window only if admissions overlap; keep the
    # earliest matching admission so each note maps to at most one hadm_id
    windows = windows.sort_values(["subject_id", "admittime"])

    rr_path = need(f"{NOTE}/radiology.csv.gz")
    kept, seen, t1 = [], 0, time.time()
    reader = pd.read_csv(rr_path, chunksize=A.chunksize,
                         usecols=["note_id", "subject_id", "hadm_id", "note_type",
                                  "charttime", "storetime", "text"])
    for ch in reader:
        seen += len(ch)
        ch["charttime"] = pd.to_datetime(ch.charttime, errors="coerce")
        ch = ch[ch.charttime.notna()]
        ch = ch.rename(columns={"hadm_id": "note_hadm_id"})
        # 52.6% of radiology notes carry NO hadm_id (outpatient/ED), so we link on
        # subject_id + charttime falling inside the admission window rather than on hadm_id
        m = ch.merge(windows, on="subject_id", how="inner")
        m = m[(m.charttime >= m.admittime) & (m.charttime <= m.win_end)]
        if len(m):
            m = m.sort_values(["note_id", "admittime"]).drop_duplicates("note_id", keep="first")
            m["hours_from_admit"] = (m.charttime - m.admittime).dt.total_seconds() / 3600.0
            m["text"] = m.text.str.slice(0, A.max_note_chars)
            kept.append(m[["note_id", "subject_id", "hadm_id", "note_type", "charttime",
                           "hours_from_admit", "text"]])
        if seen % (A.chunksize * 10) == 0:
            log(f"  radiology scanned {seen:,} notes, kept {sum(len(k) for k in kept):,} "
                f"({time.time()-t1:.0f}s)")
    notes_early = (pd.concat(kept, ignore_index=True) if kept
                   else pd.DataFrame(columns=["note_id", "subject_id", "hadm_id"]))
    log(f"radiology: scanned {seen:,} notes -> {len(notes_early):,} inside the "
        f"{A.window_hours:g}h window, covering {notes_early.hadm_id.nunique():,} admissions")
    notes_early.to_parquet(f"{A.out}/notes_early.parquet", index=False)

    if A.include_discharge:
        ds_path = need(f"{NOTE}/discharge.csv.gz")
        ds = pd.read_csv(ds_path, usecols=["note_id", "subject_id", "hadm_id", "charttime", "text"])
        ds = ds[ds.hadm_id.isin(set(df.hadm_id))]
        ds["text"] = ds.text.str.slice(0, A.max_note_chars * 4)
        ds.to_parquet(f"{A.out}/notes_discharge.parquet", index=False)
        log(f"discharge summaries dumped: {len(ds):,} (SFT TARGET ONLY - never a predictor)")

# ------------------------------------------------------------------------ finalize
if notes_early is not None:
    cnt = notes_early.groupby("hadm_id").size().rename("n_early_notes")
    df = df.merge(cnt, on="hadm_id", how="left")
    df["n_early_notes"] = df.n_early_notes.fillna(0).astype(int)
    df["has_early_note"] = (df.n_early_notes > 0).astype(int)

keep = ["subject_id", "hadm_id", "admittime", "dischtime", "age", "gender", "race", "insurance",
        "language", "marital_status", "admission_type", "admission_location", "discharge_location",
        "y_mortality", "y_readmit_30d", "y_readmit_30d_unplanned", "y_los_days",
        "days_to_next_admit", "died_within_window_post_discharge",
        "is_last_admission", "readmit_censored",
        "cold_user_first_admission", "n_prior_admissions", "admission_rank",
        "cold_item_rare_dx", "primary_icd", "primary_icd_version", "primary_dx_title", "dx_freq"]
if notes_early is not None: keep += ["n_early_notes", "has_early_note"]
out = df[[c for c in keep if c in df.columns]].copy()
out.to_parquet(f"{A.out}/cohort.parquet", index=False)

# ------------------------------------------------------------------------ EDA report
log("")
log("=" * 70)
log(f"COHORT  admissions={len(out):,}  patients={out.subject_id.nunique():,}")
log("=" * 70)
log(f"  observation funnel (readmit basis = {A.readmit_basis}):")
log(f"    all admissions                      n={len(out):,}   <- y_mortality, y_los_days observed here")
surv = out[out.y_mortality == 0]
log(f"    |- survived                         n={len(surv):,}  ({len(surv)/len(out)*100:.1f}%)")
obs = out[out.y_readmit_30d.notna()]
log(f"       |- follow-up observed            n={len(obs):,}  ({len(obs)/len(out)*100:.1f}%)"
    f"   <- y_readmit_30d observed here")
log(f"          dropped as censored: {int(out.readmit_censored.sum()):,} last admissions")
log("")
log(f"  y_mortality           rate={out.y_mortality.mean()*100:.2f}%  (n={int(out.y_mortality.sum()):,})")
log(f"  y_readmit_30d         rate={out.y_readmit_30d.mean()*100:.2f}%  (n={int(out.y_readmit_30d.sum()):,} "
    f"of {len(obs):,} observed)")
log(f"  y_readmit_30d_unplan  rate={out.y_readmit_30d_unplanned.mean()*100:.2f}%")
log(f"  competing risk: died <= {A.readmit_days}d after discharge = "
    f"{out.died_within_window_post_discharge.mean()*100:.2f}% of survivors' rows")
log(f"  y_los_days            mean={out.y_los_days.mean():.2f} median={out.y_los_days.median():.2f} "
    f"p95={out.y_los_days.quantile(.95):.2f} max={out.y_los_days.max():.1f}")
log("")
log(f"  cold_user_first_admission  {out.cold_user_first_admission.mean()*100:.1f}%  "
    f"(n={int(out.cold_user_first_admission.sum()):,})")
log(f"  cold_item_rare_dx          {out.cold_item_rare_dx.mean()*100:.1f}%  "
    f"(n={int(out.cold_item_rare_dx.sum()):,})")
both = ((out.cold_user_first_admission == 1) & (out.cold_item_rare_dx == 1)).sum()
log(f"  both cold                  {both/len(out)*100:.1f}%  (n={both:,})")
log("")
log("  censoring (recorded, and excluded from the readmission head):")
log(f"     readmit_censored = patient's last admission, follow-up unknown: "
    f"{out.readmit_censored.mean()*100:.1f}% of admissions")
log(f"     censoring rate  cold-user={out[out.cold_user_first_admission==1].readmit_censored.mean()*100:.1f}%  "
    f"warm-user={out[out.cold_user_first_admission==0].readmit_censored.mean()*100:.1f}%  <- NOT balanced,")
log("     which is why the naive all-rows label mechanically deflates the cold-user readmission rate.")
log("")
log("  label rate by cold block (readmit on observed rows only):")
for uname, u in [("warm-user", 0), ("cold-user", 1)]:
    for iname, i in [("warm-dx", 0), ("cold-dx", 1)]:
        g = out[(out.cold_user_first_admission == u) & (out.cold_item_rare_dx == i)]
        if len(g):
            nr = int(g.y_readmit_30d.notna().sum())
            log(f"    {uname:10s} x {iname:8s} n={len(g):7,}  mort={g.y_mortality.mean()*100:5.2f}%  "
                f"readm={g.y_readmit_30d.mean()*100:5.2f}% (n_obs={nr:6,})  los={g.y_los_days.mean():5.2f}d")
if "has_early_note" in out:
    log("")
    log(f"  admissions with >=1 early radiology note ({A.window_hours:g}h): "
        f"{out.has_early_note.mean()*100:.1f}%  (n={int(out.has_early_note.sum()):,})")
    log(f"  mean early notes per covered admission: "
        f"{out[out.has_early_note==1].n_early_notes.mean():.2f}")
log("")
log(f"wrote {A.out}/cohort.parquet" + ("" if notes_early is None else f" + notes_early.parquet"))
log(f"elapsed {time.time()-t0:.0f}s")
open(f"{A.out}/eda_report.txt", "w").write("\n".join(log_lines) + "\n")

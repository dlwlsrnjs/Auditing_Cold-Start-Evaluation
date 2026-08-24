#!/usr/bin/env python3
"""Recompute every headline number in the paper from the released aggregate
metrics (mimic/out/results_mtl.jsonl) and print it next to the paper's value.

    python3 reproduce.py            # all
    python3 reproduce.py audit2     # one section

No MIMIC access is needed: results_mtl.jsonl holds per-run aggregate and
per-segment metrics only, with no identifiers and no row-level data.

Arms are compared PAIRED ON SEED and only WITHIN a tag series. Series differ in
what they vary (split, metric set, seed count), and their C1 baselines differ by
up to 0.31 aggregate AUROC points -- more than most effects here -- so a
cross-series delta is meaningless. The pairings below are the ones the paper
uses; several tag names are misleading, which is why they are listed explicitly.
"""
import json, sys, collections, statistics as st
from math import sqrt

try:
    from scipy import stats
    def pval(t, n): return float(stats.t.sf(abs(t), n - 1) * 2)
    def tcrit(n):   return float(stats.t.ppf(0.975, n - 1))
except ImportError:                                   # scipy optional
    def pval(t, n): return float("nan")
    def tcrit(n):   return 1.96

RES = "mimic/out/results_mtl.jsonl"
R = collections.defaultdict(dict)
for line in open(RES):
    d = json.loads(line)
    R[d["tag"]][d["seed"]] = d

CC = "coldU_coldD"                    # doubly-cold segment: the primary endpoint's

def val(d, metric):
    if metric == "primary":           # recall at a 10% review budget on CC
        return d["blocks"][CC]["mort_recall@10"] * 100
    if metric.startswith("seg:"):
        _, seg, m = metric.split(":")
        return d["blocks"][seg][m] * 100
    return d[metric] * 100

def delta(a, b, metric):
    if isinstance(a, list):                       # mean over independent draws
        rs = [delta(x, b, metric) for x in a]
        n = rs[0][4]
        return (st.mean(r[0] for r in rs), float("nan"),
                min(r[2] for r in rs), max(r[3] for r in rs), n)
    seeds = sorted(set(R[a]) & set(R[b]))
    if len(seeds) < 3:
        raise KeyError(f"{a} vs {b}: only {len(seeds)} shared seeds")
    dd = [val(R[a][s], metric) - val(R[b][s], metric) for s in seeds]
    mu, sd = st.mean(dd), st.stdev(dd)
    se = sd / sqrt(len(dd))
    lo, hi = mu - tcrit(len(dd)) * se, mu + tcrit(len(dd)) * se
    return mu, pval(mu / se, len(dd)), lo, hi, len(dd)

# section, label, arm, control, metric, value printed in the paper
CHECKS = [
 ("audit1", "zero-imputed vs corrected label, aggregate readmission AUROC",
  "v3-corrected", "v3-naive", "readmit_auroc", "+0.40"),
 ("audit1", "  cost on the fully warm segment",
  "v3-corrected", "v3-naive", "seg:warmU_warmD:readmit_auroc", "+0.28"),
 ("audit1", "  cost on the cold-user segment",
  "v3-corrected", "v3-naive", "seg:coldU_warmD:readmit_auroc", "+0.72"),
 ("audit1", "  cost on the doubly-cold segment",
  "v3-corrected", "v3-naive", "seg:coldU_coldD:readmit_auroc", "+1.03"),

 ("audit2", "oracle anchor vs its own C1 (15 seeds)",
  "sat-txt-oracle", "sat-txt-c1", "mort_auroc", "+0.33"),
 ("audit2", "prior anchor vs its own C1 (15 seeds)",
  "sat-txt-prior", "sat-txt-c1", "mort_auroc", "-1.01"),
 ("audit2", "within-group permuted anchor (5 seeds)",
  "rm-permanchor", "rm-base", "mort_auroc", "-0.38"),
 ("audit2", "raw outcome simulator, injected",
  "fx-inj-sim", "fx-base", "mort_auroc", "-2.11"),
 ("audit2", "recalibrated simulator, injected",
  "fx-inj-simcal-exact", "fx-base", "mort_auroc", "+0.01"),

 ("audit3", "design 6 personas vs C1, aggregate AUROC",
  "rm-persona", "rm-base", "mort_auroc", "+0.17"),
 ("audit3", "design 6 personas vs C2, PRIMARY ENDPOINT",
  "rm-persona", "rm-naive", "primary", "+3.23"),
 ("audit3", "C2 vs C1 on the primary endpoint (rows alone are not the carrier)",
  "rm-naive", "rm-base", "primary", "-0.55"),
 ("audit3", "text permuted within chapter vs C2, mean of 4 draws",
  ["rm-persona-shuf", "rm-persona-shuf2", "rm-persona-shuf3", "rm-persona-shuf4"],
  "rm-naive", "primary", "+2.52"),
 ("audit3", "labels flattened to the ICD-3 base rate, vs C1",
  "rm-persona-flatlabel", "rm-base", "mort_auroc", "+0.02"),
 ("audit3", "labels permuted WITHIN code, vs intact design",
  "rm-persona-codeshuf", "rm-persona", "mort_auroc", "-0.03"),
 ("audit3", "labels permuted ACROSS codes in a chapter, vs C1",
  "rm-persona-chapshuf", "rm-base", "mort_auroc", "+0.02"),
 ("audit3", "  the same, against the intact design",
  "rm-persona-chapshuf", "rm-persona", "mort_auroc", "-0.15"),
 ("audit3", "  and against flattened labels (indistinguishable)",
  "rm-persona-chapshuf", "rm-persona-flatlabel", "mort_auroc", "-0.004"),

 ("q3", "laboratory block vs no-laboratory baseline, AUPRC",
  "rm2-labs", "rm-base", "mort_auprc", "+21.21"),
 ("q3", "  test count + abnormal fraction only",
  "lg-count", "rm-base", "mort_auprc", "+10.69"),
 ("q3", "  abnormal fraction alone",
  "lg-abnfrac", "rm-base", "mort_auprc", "+9.55"),
 ("q3", "  test count alone (the care-process variable)",
  "lg-labn", "rm-base", "mort_auprc", "+4.61"),
 ("q3", "  incremental value of the test count",
  "lg-count", "lg-abnfrac", "mort_auprc", "+1.14"),
 ("q3", "personas on top of laboratory features, AUPRC",
  "rm2-labs-persona", "rm2-labs", "mort_auprc", "-0.17"),

 ("hospice", "persona gain under the standard mortality label",
  "rm-persona", "rm-base", "mort_auroc", "+0.17"),
 ("hospice", "persona gain with hospice discharges counted positive",
  "hosp-persona", "hosp-base", "mort_auroc", "+0.08"),
 ("hospice", "  the same, on the primary endpoint",
  "hosp-persona", "hosp-base", "primary", "+0.32"),
]

def main():
    want = sys.argv[1] if len(sys.argv) > 1 else None
    sec = None
    for s, label, a, b, m, paper in CHECKS:
        if want and s != want:
            continue
        if s != sec:
            sec = s
            print(f"\n=== {sec} " + "=" * (66 - len(sec)))
        try:
            mu, p, lo, hi, n = delta(a, b, m)
            flag = " " if abs(mu - float(paper)) < 0.02 + abs(float(paper)) * 0.02 else "!"
            print(f"{flag} {label:56s} {mu:+8.3f}  p={p:7.4f}  "
                  f"[{lo:+7.3f},{hi:+7.3f}]  n={n:2d}  paper {paper}")
        except KeyError as e:
            print(f"? {label:56s} {e}")
    print("\n'!' marks a recomputed value that differs from the paper's printed one.")
    print("Segment ordering ('81 of 82 configurations'): python3 mimic/count_ordering.py")

if __name__ == "__main__":
    main()

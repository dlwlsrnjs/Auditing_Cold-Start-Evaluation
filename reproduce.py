#!/usr/bin/env python3
"""Reproduce every table and number in the paper, in one command.

    python3 reproduce.py                 # everything
    python3 reproduce.py --only tables   # tables only
    python3 reproduce.py --only numbers  # headline effects only
    python3 reproduce.py --list          # what each section covers

Input is mimic/out/results_mtl.jsonl: one record per training run holding the
configuration, the seed, and aggregate plus per-segment metrics. It carries no
identifiers and no row-level data, so this script needs no MIMIC access.

Everything is a paired difference across seeds WITHIN one tag series. Series
differ in what they vary (patient split, metric set, seed count) and their C1
baselines differ by up to 0.31 aggregate AUROC points, which is more than most
effects in the paper -- so a cross-series delta is meaningless. Some tag names
are misleading; the pairings used below are the paper's, listed explicitly.
"""
import argparse, collections, json, os, sys
import statistics as st
from math import sqrt

try:
    from scipy import stats
    def pval(t, n): return float(stats.t.sf(abs(t), n - 1) * 2)
    def tcrit(n):   return float(stats.t.ppf(0.975, n - 1))
except ImportError:
    def pval(t, n): return float("nan")
    def tcrit(n):   return 1.96

HERE = os.path.dirname(os.path.abspath(__file__))
RES  = os.path.join(HERE, "mimic", "out", "results_mtl.jsonl")

R, DUP = collections.defaultdict(dict), collections.Counter()
for line in open(RES):
    d = json.loads(line)
    if d["seed"] in R[d["tag"]]:
        DUP[d["tag"]] += 1
        continue                      # keep the first run; a re-run never silently wins
    R[d["tag"]][d["seed"]] = d

CC = "coldU_coldD"
SEGS = [("WW", "warmU_warmD"), ("WC", "warmU_coldD"),
        ("CW", "coldU_warmD"), ("CC", "coldU_coldD")]

def val(d, metric):
    if metric == "primary":
        return d["blocks"][CC]["mort_recall@10"] * 100
    if metric.startswith("seg:"):
        _, seg, m = metric.split(":")
        return d["blocks"][seg][m] * 100
    return d[metric] * 100

def mean(tag, metric):
    return st.mean(val(R[tag][s], metric) for s in sorted(R[tag]))

def delta(a, b, metric):
    """Paired difference a - b over shared seeds. `a` may be a list of draws."""
    if isinstance(a, list):
        rs = [delta(x, b, metric) for x in a]
        return (st.mean(r[0] for r in rs), float("nan"),
                min(r[2] for r in rs), max(r[3] for r in rs), rs[0][4])
    seeds = sorted(set(R[a]) & set(R[b]))
    if len(seeds) < 3:
        raise KeyError(f"{a} vs {b}: only {len(seeds)} shared seeds")
    dd = [val(R[a][s], metric) - val(R[b][s], metric) for s in seeds]
    mu = st.mean(dd); se = st.stdev(dd) / sqrt(len(dd))
    return mu, pval(mu / se, len(dd)), mu - tcrit(len(dd)) * se, mu + tcrit(len(dd)) * se, len(dd)

def head(title):
    print(f"\n{'=' * 74}\n{title}\n{'=' * 74}")

def fmt_p(p):
    return "  n/a " if p != p else ("<0.001" if p < 0.001 else f"{p:6.3f}")

# ----------------------------------------------------------------- tables ---

def table2():
    head("Table 2 (lower block) - what the labelling convention does to the model")
    print("Readmission AUROC, matched evaluation rows, 5 seeds. The two arms are")
    print("trained on the same admissions and differ only in the training label.")
    print("The upper block of Table 2 (censoring and label rates) needs the cohort.\n")
    print(f"  {'':22s}{'zero-imputed':>14s}{'corrected':>12s}{'delta':>9s}{'p':>9s}")
    rows = [("Aggregate", "readmit_auroc")] + [(f"  {n} segment", f"seg:{k}:readmit_auroc") for n, k in SEGS]
    for label, m in rows:
        a, b = mean("v3-naive", m), mean("v3-corrected", m)
        d, p, _, _, _ = delta("v3-corrected", "v3-naive", m)
        print(f"  {label:22s}{a/100:14.4f}{b/100:12.4f}{d:+9.2f}{fmt_p(p):>9s}")

DESIGNS = [
    (1, "Retrieval-grounded agent summary",      "I", "fx-agent",            "fx-base", "fx-persona-naive"),
    (2, "kNN neighbour statistics (non-LLM)",    "I", "fx-knn",              "fx-base", "fx-persona-naive"),
    (3, "LLM risk write-up, no retrieval",       "I", "fx-noretr",           "fx-base", "fx-persona-naive"),
    (4, "Outcome simulator, raw",                "L", "fx-inj-sim",          "fx-base", "fx-persona-naive"),
    (5, "Outcome simulator, recalibrated",       "L", "fx-inj-simcal-exact", "fx-base", "fx-persona-naive"),
    (6, "Synthetic patient personas",            "S", "fx-persona-llm",      "fx-base", "fx-persona-naive"),
    (7, "Personas, base-rate anchored",          "S", "fx-persona-patho",    "fx-base", "fx-persona-naive"),
    (8, "Personas, severity-graded (weighted)",  "S", "rm-persona-sevw",     "rm-base", "rm-naive"),
    (9, "Personas, severity-graded (unweighted)","S", "rm-persona-sev_raw",  "rm-base", "rm-naive"),
]

def table3():
    head("Table 3 - eight design families, nine configurations")
    print("Aggregate mortality AUROC in points, 5 seeds, paired. '*' marks a positive")
    print("gain with nominal paired p<0.05, before the multiplicity analysis.")
    print("Rows 8-9 run in their own series against that series' C1 and C2.\n")
    print(f"  {'#':>2s} {'design':42s}{'ch':>3s}{'vs C1':>9s}{'p':>8s}{'vs C2':>9s}{'p':>8s}")
    for n, name, ch, tag, c1, c2 in DESIGNS:
        try:
            d1, p1, *_ = delta(tag, c1, "mort_auroc")
            d2, p2, *_ = delta(tag, c2, "mort_auroc")
        except KeyError as e:
            print(f"  {n:>2d} {name:42s}{ch:>3s}   {e}"); continue
        star = "*" if (p1 < 0.05 and d1 > 0) else " "
        print(f"  {n:>2d} {name:42s}{ch:>3s}{d1:+9.2f}{fmt_p(p1):>8s}{d2:+9.2f}{fmt_p(p2):>8s} {star}")

PRIMARY = [
    ("C1, no row/label augmentation",     "rm-base",          "rm-base"),
    ("C2, matched heuristic rows",        "rm-naive",         "rm-base"),
    ("Prior anchor (15 seeds)",           "sat-txt-prior",    "sat-txt-c1"),
    ("Personas  [pre-specified]",         "rm-persona",       "rm-base"),
    ("Oracle anchor (15 seeds)",          "sat-txt-oracle",   "sat-txt-c1"),
    ("Identity retained, untreated",      "bl-identity",      "rm-base"),
    ("  + ALDI-style distillation",       "bl-aldi",          "rm-base"),
    ("SMOTE-style synthetic rows",        "bl-smote",         "rm-base"),
    ("C1, + labs",                        "rm2-labs",         "rm2-labs"),
    ("  + labs, personas",                "rm2-labs-persona", "rm2-labs"),
]
DEATHS = 254        # deaths in the CC test segment (9,886 admissions, fixed split)

def table4():
    head("Table 4 - the pre-specified primary endpoint")
    print("Recall at a 10% review budget on the doubly-cold CC segment")
    print(f"(9,886 admissions, {DEATHS} deaths, 989 charts reviewed).\n")
    print(f"  {'arm':36s}{'recall':>9s}{'delta':>9s}{'p':>8s}{'deaths':>8s}")
    for label, tag, ref in PRIMARY:
        if tag not in R:
            print(f"  {label:36s}   (missing)"); continue
        v = mean(tag, "primary")
        if tag == ref:
            print(f"  {label:36s}{v/100:9.4f}{'---':>9s}{'---':>8s}{'---':>8s}"); continue
        d, p, *_ = delta(tag, ref, "primary")
        print(f"  {label:36s}{v/100:9.4f}{d:+9.2f}{fmt_p(p):>8s}{d/100*DEATHS:+8.0f}")
    d, p, lo, hi, n = delta("rm-persona", "rm-naive", "primary")
    print(f"\n  pre-specified: personas vs C2  {d:+.2f} pt, p={p:.3f}, "
          f"95% CI [{lo:+.2f},{hi:+.2f}], n={n} seeds")
    print("  the paper also reports a test-set bootstrap here: +2.4, 95% CI [0.0,+5.5],")
    print("  p=0.053 (needs per-admission predictions; see bootstrap_primary.py)")

HOLM = [("R@10%", "C2", "rm-naive", "mort_recall@10"),
        ("AUROC", "C2", "rm-naive", "mort_auroc"),
        ("AUPRC", "C2", "rm-naive", "mort_auprc"),
        ("AUROC", "C1", "rm-base",  "mort_auroc"),
        ("R@5%",  "C1", "rm-base",  "mort_recall@5"),
        ("AUPRC", "C1", "rm-base",  "mort_auprc"),
        ("R@10%", "C1", "rm-base",  "mort_recall@10"),
        ("R@5%",  "C2", "rm-naive", "mort_recall@5")]

def table5():
    head("Table 5 - the eight secondary persona comparisons under Holm")
    print("alpha=0.05, 5 seeds, paired. The primary endpoint of Table 4 is exempt")
    print("from this family; the recall entries here are the AGGREGATE metric.\n")
    raw = []
    for metric, ctrl, ref, key in HOLM:
        d, p, *_ = delta("rm-persona", ref, key)
        raw.append((p, metric, ctrl, d))
    raw.sort()
    m = len(raw); adj, run = [], 0.0
    for i, (p, metric, ctrl, d) in enumerate(raw):
        a = min(1.0, max(run, (m - i) * p)); run = a
        adj.append((metric, ctrl, d, p, a))
    print(f"  {'metric':8s}{'vs':>4s}{'delta':>9s}{'raw p':>9s}{'adj p':>9s}   verdict")
    for metric, ctrl, d, p, a in adj:
        print(f"  {metric:8s}{ctrl:>4s}{d:+9.2f}{p:9.3f}{a:9.3f}   "
              f"{'survives' if a < 0.05 else 'does not'}")
    print(f"\n  survives Holm: {sum(1 for *_, a in adj if a < 0.05)} of {m}")

def table6():
    head("Table 6 - laboratory features, and whether the persona gain survives them")
    print("5 seeds, fixed split. R@K is recall at a K% review budget.\n")
    arms = [("C1 baseline", "rm-base"), ("+ personas", "rm-persona"),
            ("+ labs", "rm2-labs"), ("+ labs + personas", "rm2-labs-persona")]
    cols = [("AUPRC", "mort_auprc"), ("R@5%", "mort_recall@5"), ("R@10%", "mort_recall@10")]
    hdr = "".join(f"{c:>9s}" for c, _ in cols) + "".join(f"{n:>9s}" for n, _ in SEGS)
    print(f"  {'configuration':20s}{hdr}")
    for label, tag in arms:
        if tag not in R:
            print(f"  {label:20s}  (missing)"); continue
        cs = "".join(f"{mean(tag, k)/100:9.4f}" for _, k in cols)
        ss = "".join(f"{mean(tag, f'seg:{k}:mort_auroc')/100:9.4f}" for _, k in SEGS)
        print(f"  {label:20s}{cs}{ss}")
    d, p, lo, hi, _ = delta("rm2-labs-persona", "rm2-labs", "mort_auprc")
    print(f"\n  personas on top of labs: {d:+.2f} AUPRC, p={p:.3f}, [{lo:+.2f},{hi:+.2f}]")

def ordering():
    head("'both cold-user segments outrank both warm-user ones in 81 of 82'")
    print("mode=mtl, >=5 seeds, no label injection; mortality AUROC, seed means.\n")
    hold, total, exc = 0, 0, []
    for tag, runs in R.items():
        rs = [r for r in runs.values() if r.get("mode") == "mtl" and "blocks" in r]
        if len({r["seed"] for r in rs}) < 5 or any(r.get("inject") for r in rs):
            continue
        if not all(all(k in r["blocks"] for _, k in SEGS) for r in rs):
            continue
        m = {k: st.mean(r["blocks"][k]["mort_auroc"] for r in rs) for _, k in SEGS}
        total += 1
        if min(m["coldU_warmD"], m["coldU_coldD"]) > max(m["warmU_warmD"], m["warmU_coldD"]):
            hold += 1
        else:
            exc.append(tag)
    print(f"  {hold} of {total} configurations")
    for t in exc:
        print(f"    exception: {t}")

# ---------------------------------------------------------------- numbers ---

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

def numbers():
    head("Headline effects, recomputed against the values printed in the paper")
    sec, bad = None, 0
    for s, label, a, b, m, paper in CHECKS:
        if s != sec:
            sec = s
            print(f"\n-- {sec} " + "-" * (69 - len(sec)))
        try:
            mu, p, lo, hi, n = delta(a, b, m)
            off = abs(mu - float(paper)) > 0.02 + abs(float(paper)) * 0.02
            bad += off
            print(f"{'!' if off else ' '} {label:56s}{mu:+8.3f}  p={fmt_p(p)}  "
                  f"[{lo:+7.3f},{hi:+7.3f}]  n={n:2d}  paper {paper}")
        except KeyError as e:
            bad += 1
            print(f"? {label:56s}{e}")
    print(f"\n  {len(CHECKS)} checks, {bad} disagreeing with the paper ('!' marks each).")
    return bad

def missing():
    head("What this script cannot reproduce, and why")
    print("""These need per-admission predictions (mimic/out/preds/), which are
patient-level MIMIC-derived data and are not in this repository. Both scripts
are included and run once you have rebuilt preds/ with --save-preds:

  Table 4 bootstrap row   +2.4, 95% CI [0.0,+5.5], p=0.053
                          python3 mimic/bootstrap_primary.py
  Disaggregated results   AUROC 88.3-98.4, flagged share 4.9-44.3%,
                          calibration 0.87-1.22
                          python3 mimic/subgroup_fairness.py

The upper block of Table 2 (censoring rate, zero-imputed and observed label
rates per segment) comes from the cohort, rebuilt by mimic/fix_censoring.py.

Audit 1 on the public education log needs the Eedi data, which is public:
  python3 mimic/audit_eedi.py     (reference output: mimic/out/eedi_audit.log)""")

SECTIONS = {"tables":  [table2, table3, table4, table5, table6, ordering],
            "numbers": [numbers],
            "missing": [missing]}

def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--only", choices=sorted(SECTIONS), help="run one section")
    ap.add_argument("--list", action="store_true", help="describe the sections")
    a = ap.parse_args()
    if a.list:
        print("tables   Tables 2 (lower block), 3, 4, 5, 6 and the 81-of-82 ordering count")
        print("numbers  27 headline effects, each next to the value printed in the paper")
        print("missing  what needs credentialed data, and which script produces it")
        return
    print(f"reproduce.py - {sum(len(v) for v in R.values())} runs, "
          f"{len(R)} tags, from {os.path.relpath(RES, HERE)}")
    if DUP:
        print("note: duplicate seed runs ignored (first kept): "
              + ", ".join(f"{t}x{n}" for t, n in sorted(DUP.items())))
    bad = 0
    for name in (["tables", "numbers", "missing"] if not a.only else [a.only]):
        for fn in SECTIONS[name]:
            bad += fn() or 0
    sys.exit(1 if bad else 0)

if __name__ == "__main__":
    main()

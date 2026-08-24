#!/usr/bin/env python3
"""Emit the primary-endpoint table (N6) from results_mtl.jsonl and splice it into paper.tex
between the PRIMARY-TABLE markers. Re-run whenever new bracket runs land."""
import os, sys

PAPER = (sys.argv[1] if len(sys.argv) > 1
         else os.environ.get("PAPER_TEX", "wsdm/paper.tex"))
import json, collections
import numpy as np
from scipy import stats

R = collections.defaultdict(dict)
for l in open("mimic/out/results_mtl.jsonl"):
    try: d = json.loads(l)
    except Exception: continue
    b = d.get("blocks", {}).get("coldU_coldD", {})
    if "mort_recall@10" in b and d["seed"] not in R[d["tag"]]:
        R[d["tag"]][d["seed"]] = b["mort_recall@10"]   # first run wins, no silent overwrite

DEATHS = 254                      # deaths in the CC test segment (9,886 admissions, fixed split)
def vals(t): return np.array([R[t][s] for s in sorted(R[t])]) if t in R else None
def row(label, tag, ref):
    v = vals(tag)
    if v is None: return f"{label} & \\multicolumn{{4}}{{c}}{{\\emph{{run pending}}}} \\\\"
    r = vals(ref)
    if r is None or tag == ref:
        d = p = None
    else:
        s = sorted(set(R[tag]) & set(R[ref]))
        x = np.array([R[tag][i] for i in s]); y = np.array([R[ref][i] for i in s])
        d = (x - y).mean() * 100; p = stats.ttest_rel(x, y)[1]
    dd = "---" if d is None else f"${d:+.2f}$"
    pp = "---" if p is None else (f"${p:.3f}$" if p >= 0.001 else "$<$0.001")
    nd = "---" if d is None else f"${d/100*DEATHS:+.0f}$"
    return f"{label} & {v.mean():.4f} & {dd} & {pp} & {nd} \\\\"

c15_txt = vals('sat-txt-c1').mean()
c15_lab = vals('sat-c1').mean()

L = [r"\begin{table}[t]",
     r"\footnotesize",
     r"\caption{The pre-specified primary endpoint: recall at a 10\% review budget on",
     r"the doubly-cold CC segment (9{,}886 admissions, 254 deaths, 989 charts). Design",
     r"rows are 5-seed and C1-relative. The last column",
     r"converts $\Delta$ into deaths, of the 254. Personas inject rows as well as",
     r"labels, so their position relative to the anchors is a reference point, not a bound",
     r"(\S\ref{sec:bracket}). $^{\ddagger}$Pre-specified comparison is against C2:",
     r"$+3.23$ pt, $p{=}0.003$. $^{*}$Anchor rows are 15-seed and their $\Delta$ is",
     r"measured against the 15-seed C1 of their own series, "
       + f"{c15_txt:.4f} (text)",
     f"and {c15_lab:.4f} (laboratory), which is not shown; differencing the level",
     r"column against the 5-seed C1 above will therefore not reproduce it.}",
     r"\label{tab:primary}",
     r"\setlength{\tabcolsep}{3.5pt}%",
     r"\resizebox{\ifdim\width>\columnwidth\columnwidth\else\width\fi}{!}{%",
     r"\begin{tabular}{lrrrr}",
     r"\toprule",
     r"Configuration & R@10\% & $\Delta$ & $p$ & deaths \\",
     r"\midrule",
     r"\multicolumn{5}{l}{\emph{Text-only baseline}} \\",
     row(r"\quad C1, no row/label augmentation", "rm-base", "rm-base"),
     row(r"\quad C2, matched heuristic rows", "rm-naive", "rm-base"),
     row(r"\quad Prior anchor$^{*}$", "sat-txt-prior", "sat-txt-c1"),
     row(r"\quad \textbf{Personas}$^{\ddagger}$", "rm-persona", "rm-base"),
     row(r"\quad Oracle anchor$^{*}$", "sat-txt-oracle", "sat-txt-c1"),
     r"\midrule",
     r"\multicolumn{5}{l}{\emph{Cold-start comparators, text-only baseline}} \\",
     row(r"\quad Identity retained, untreated", "bl-identity", "rm-base"),
     row(r"\quad \quad ${+}$ ALDI-style distillation", "bl-aldi", "rm-base"),
     row(r"\quad SMOTE-style synthetic rows", "bl-smote", "rm-base"),
     r"\midrule",
     r"\multicolumn{5}{l}{\emph{Laboratory baseline}} \\",
     row(r"\quad C1, ${+}$ labs", "rm2-labs", "rm2-labs"),
     row(r"\quad ${+}$ labs, personas", "rm2-labs-persona", "rm2-labs"),
     row(r"\quad Prior anchor$^{*}$", "sat-prior", "sat-c1"),
     row(r"\quad Oracle anchor$^{*}$", "sat-oracle", "sat-c1"),
     r"\bottomrule", r"\end{tabular}}", r"\end{table}"]
tab = "\n".join(L)

src = open(PAPER).read()
A, B = "% <<<PRIMARY-TABLE>>>", "% <<<END-PRIMARY-TABLE>>>"
if A in src:
    src = src[:src.index(A) + len(A)] + "\n" + tab + "\n" + src[src.index(B):]
else:
    anchor = "\\paragraph{The surviving design.}"
    src = src.replace(anchor, A + "\n" + tab + "\n" + B + "\n\n" + anchor, 1)
open(PAPER, "w").write(src)
print(tab)

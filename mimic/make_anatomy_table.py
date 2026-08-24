#!/usr/bin/env python3
"""Emit Table 3 (the augmentation anatomy) from results_mtl.jsonl with explicit vs-C1 and
vs-C2 columns, and splice it into paper.tex between the ANATOMY-TABLE markers."""
import os, sys

PAPER = (sys.argv[1] if len(sys.argv) > 1
         else os.environ.get("PAPER_TEX", "wsdm/paper.tex"))
import json, collections
import numpy as np
from scipy import stats

R = collections.defaultdict(dict)
DUP = collections.defaultdict(int)
for l in open("mimic/out/results_mtl.jsonl"):
    try: d = json.loads(l)
    except Exception: continue
    if d["seed"] in R[d["tag"]]:
        DUP[d["tag"]] += 1
        continue                      # keep the FIRST run; never let a re-run silently win
    R[d["tag"]][d["seed"]] = d
for t, n in sorted(DUP.items()):
    print(f"# note: {t} has {n} duplicate seed run(s); first-run values used", flush=True)

def cmp(a, b, k="mort_auroc"):
    s = sorted(set(R[a]) & set(R[b]))
    x = np.array([R[a][i][k] for i in s]); y = np.array([R[b][i][k] for i in s])
    return (x - y).mean() * 100, stats.ttest_rel(x, y)[1]

def cell(a, b):
    d, p = cmp(a, b)
    ps = f"{p:.3f}" if p >= 0.001 else "$<$0.001"
    num = f"$\\mathbf{{{d:+.2f}}}$" if (p < 0.05 and d > 0) else f"${d:+.2f}$"
    return f"{num} & {ps}"

# design, channel, tag, C1 tag, C2 tag, verdict
D = [(1, "Retrieval-grounded agent summary", "I", "fx-agent",        "fx-base", "fx-persona-naive", "no gain"),
     (2, "$k$NN neighbor statistics (non-LLM)", "I", "fx-knn",         "fx-base", "fx-persona-naive", "at the run-to-run level"),
     (3, "LLM risk write-up, no retrieval",  "I", "fx-noretr",       "fx-base", "fx-persona-naive", "no gain vs.\\ C1"),
     (4, "Outcome simulator, raw",           "L", "fx-inj-sim",      "fx-base", "fx-persona-naive", "below the prior anchor"),
     (5, "Outcome simulator, recalibrated",  "L", "fx-inj-simcal-exact", "fx-base", "fx-persona-naive", "neutral"),
     (6, "\\textbf{Synthetic patient personas}", "S", "fx-persona-llm", "fx-base", "fx-persona-naive", "positive in both seed tests"),
     (7, "Personas, base-rate anchored",     "S", "fx-persona-patho","fx-base", "fx-persona-naive", "no gain"),
     (8, "Personas, severity-graded (weighted)", "S", "rm-persona-sevw", "rm-base", "rm-naive", "positive in both seed tests"),
     (9, "Personas, severity-graded (unweighted)", "S", "rm-persona-sev_raw", "rm-base", "rm-naive", "positive in both seed tests")]

L = [r"\begin{table}[t]",
     r"\caption{Eight design families, nine evaluated configurations, against both",
     r"pre-specified controls; aggregate mortality AUROC (points, 5 seeds, paired).",
     r"Channels: \textbf{I} input embedding, \textbf{L} label injection, \textbf{S}",
     r"synthetic rows; bold marks a positive gain with nominal paired $p<0.05$",
     r"before the multiplicity analysis below. Designs 1, 2, 5 and",
     r"7 fall within the 0.07-point run-to-run band of \S\ref{sec:experiments}, and",
     r"design 3 clears it by 0.002. Only design 6 is",
     r"row-matched to C2; regenerated at C2's budget, designs 8--9 lose their gain",
     r"over it. Rows 8--9 run in the rank-metric series against that series' own C1",
     r"and C2, so their $\Delta$ is not on the same baseline as rows 1--7; rows 4 and",
     r"5 count as two families, raw and recalibrated injection being different",
     r"treatments of the label.}",
     r"\label{tab:rq3}",
     r"\setlength{\tabcolsep}{3.5pt}%",
     r"\resizebox{\ifdim\width>\columnwidth\columnwidth\else\width\fi}{!}{%",
     r"\begin{tabular}{clcrrrr}",
     r"\toprule",
     r"& & & \multicolumn{2}{c}{vs.\ C1} & \multicolumn{2}{c}{vs.\ C2} \\",
     r"\cmidrule(lr){4-5}\cmidrule(lr){6-7}",
     r"\# & Design & Ch. & $\Delta$ & $p$ & $\Delta$ & $p$ \\",
     r"\midrule"]
for n, name, ch, tag, c1, c2, verdict in D:
    if n == 4:
        fl = cmp('sat-txt-prior', 'sat-txt-c1')[0]; ce = cmp('sat-txt-oracle', 'sat-txt-c1')[0]
        L.append(r"\multicolumn{7}{l}{\quad\emph{Prior anchor} " + f"${fl:+.2f}$"
                 + r" pt, \emph{oracle anchor} " + f"${ce:+.2f}$"
                 + r" pt, 15 seeds (width " + f"${ce-fl:.2f}$" + r" pt)} \\")
        L.append(r"\midrule")
    L.append(f"{n} & {name} & {ch} & {cell(tag,c1)} & {cell(tag,c2)} \\\\")
    if n == 5: L.append(r"\midrule")
    if n == 3: L.append(r"\midrule")
L += [r"\bottomrule", r"\end{tabular}}", r"\end{table}"]
tab = "\n".join(L)

src = open(PAPER).read()
A, B = "% <<<ANATOMY-TABLE>>>", "% <<<END-ANATOMY-TABLE>>>"
if A in src:
    src = src[:src.index(A) + len(A)] + "\n" + tab + "\n" + src[src.index(B):]
    open(PAPER, "w").write(src)
    print("spliced")
print(tab)

# Auditing Cold-Start Evaluation

Code and aggregate results for *Auditing Cold-Start Evaluation: What a Study
Concludes When Labels Are Unobservable and Augmentation Is Unanchored*.

A cold-start study reports a gap between segments. The paper audits three
channels through which that gap can be distorted and gives one instrument per
channel:

| channel | instrument | what it answers |
|---|---|---|
| the label itself | censoring audit + closed form | is the reported gap real? |
| the labels an augmentation injects | oracle–prior anchors | how much can the channel carry? |
| the rows an augmentation adds | content permutation | what is the gain resting on? |

---

## Reproduce everything with one command

```bash
pip install numpy pandas pyarrow scipy scikit-learn
python3 reproduce.py
```

That regenerates, from the aggregate results shipped here and with **no MIMIC
access**:

* **Table 2** (lower block) — what the labelling convention costs the fitted model, per segment
* **Table 3** — eight design families, nine augmentation configurations, against C1 and C2
* **Table 4** — the pre-specified primary endpoint, every arm, converted into deaths
* **Table 5** — the eight secondary comparisons with Holm step-down correction
* **Table 6** — laboratory features, and whether the persona gain survives them
* the **"81 of 82 configurations"** segment-ordering claim
* **27 headline effects**, each printed next to the value in the paper, with a
  paired-*t* *p*-value, a 95% interval and the seed count

A `!` marks any recomputed value that disagrees with the paper, and the script
exits non-zero if there is one. A clean run prints none. Narrower views:

```bash
python3 reproduce.py --only tables     # the tables
python3 reproduce.py --only numbers    # the headline effects
python3 reproduce.py --only missing    # what needs credentialed data, and how
python3 reproduce.py --list            # what each section covers
```

### How the comparisons are formed

Every effect is a **paired difference across seeds within one tag series**.
Series differ in what they vary (patient split, metric set, seed count), and
**their C1 baselines differ by up to 0.31 aggregate AUROC points — more than
most effects in the paper**, so a cross-series delta is meaningless. Several tag
names are also misleading, so `reproduce.py` states every pairing it uses. Three
worth knowing before reading any tag:

* `rm-naive` is **C2**, the content-matched non-LLM control — not a naive baseline.
* `rm-base` is **C1**. `fx-base` is C1 of a different series; do not cross them.
* the anchors live in `sat-txt-*`, not `sat-*`. Pairing `sat-oracle` with
  `sat-txt-c1` gives `+3.15` where the paper reports `+0.33`.

---

## Data access: what is here, and what cannot be

MIMIC-IV is credentialed data under a PhysioNet data use agreement, so **no
patient-level file is in this repository**. What is here is:

* every script in the pipeline, from cohort construction to the audits;
* `mimic/out/results_mtl.jsonl` — per-run **aggregate** metrics (583 runs), with
  no identifiers and no row-level data. These are the numbers printed in the
  paper, and they are what `reproduce.py` reads.

That is enough to check every reported effect. It is not enough to retrain: for
that you need PhysioNet credentialed access to MIMIC-IV v3.1 and MIMIC-IV-Note
v2.2, after which the scripts here rebuild everything. `mimic/out/README.md`
lists exactly which derived files are withheld and why.

Three results need per-admission predictions, which are patient-level, so they
are the one thing `reproduce.py` cannot recompute: the test-set bootstrap on the
primary endpoint (`bootstrap_primary.py`), the disaggregated performance
breakdown (`subgroup_fairness.py`), and the upper block of Table 2, which comes
from the cohort (`fix_censoring.py`). All three scripts are included.

## The public-log audit — no credentialed data needed

Audit 1 is replicated on the Eedi / NeurIPS 2020 Education Challenge log, which
is public. Download it, point the script at it, and run:

```bash
python3 mimic/audit_eedi.py        # expects eedi/data/... ; see the header
```

An episode is a student's activity day, the label is whether the student returns
within 30 days, and the item is the day's first question. The script reports the
three counts per segment, both censoring definitions, and the closed form
against the measured distortion. Reference output is in
`mimic/out/eedi_audit.log`: on the user axis zero-imputation turns a `+0.28`
point observed gap into a reported `−1.12`, a sign reversal.

---

## Rebuilding from raw MIMIC-IV

With credentialed access:

```bash
# 1. cohort and features
python3 mimic/extract_bq.py --project <GCP_PROJECT> --out mimic/out   # or use local CSVs
python3 mimic/build_cohort.py
python3 mimic/build_labs.py
python3 mimic/fix_censoring.py        # administrative censoring via anchor_year_group

# 2. frozen text embeddings
python3 mimic/embed_notes.py
python3 mimic/embed_dx.py

# 3. augmentation sources
python3 mimic/gen_personas.py         # synthetic personas from ICD titles only
python3 mimic/sft_sim.py              # fine-tune the outcome simulator
python3 mimic/gen_sim.py
python3 mimic/recal_sim.py            # diagnostic recalibration, not a deployable one

# 4. experiments (each writes one line per run into mimic/out/results_mtl.jsonl)
bash mimic/run_rankmetrics.sh         # C1, C2, personas, anchors
bash mimic/run_saturation.sh          # 15-seed anchor ends
bash mimic/run_labgroups.sh           # laboratory family decomposition
bash mimic/rf_round3.sh               # chapter-level label permutation, hospice, lg-count
bash mimic/rf_labn.sh                 # separating test count from abnormal fraction
```

`mimic/train_mtl.py` is the single training entry point; every arm above is one
flag combination on it. Useful flags:

`--personas` / `--personaemb` inject synthetic rows · `--inject-labels` overwrite
labels on real rows · `--lab-group {renal,heme,coag,hepatic,perfusion,count,labn,abnfrac}`
restrict the laboratory block · `--hospice-positive` count hospice discharges as
deaths · `--dx-mode strata` delete rare-code identity embeddings ·
`--readmit-basis {corrected,naive}` switch the censoring convention ·
`--eval-uncensored` score only where the outcome is observed ·
`--save-preds` write per-admission risks (needed for the bootstrap and the
subgroup breakdown).

### Tables in the paper

Tables 3 and 4 are generated from `results_mtl.jsonl` and written straight into
the LaTeX source between markers:

```bash
python3 mimic/make_anatomy_table.py path/to/paper.tex   # Table 3
python3 mimic/make_primary_table.py path/to/paper.tex   # Table 4
```

Both **rewrite the file in place**. Edit the caption in the script, not in the
`.tex`, or the next run reverts it.

---

## Caveats worth reading before reusing any of this

* The **cold-item axis is retrospective**. It uses the principal diagnosis code,
  which is assigned at discharge, so it is an evaluation construction rather
  than a deployable routing rule. Absolute performance in any arm that uses
  diagnosis text is optimistic for the same reason.
* The **anchors were read on one log**. The closed form of Audit 1 travels by
  construction and is checked on a second log; the anchors and the permutation
  are not.
* **Five seeds resolve little.** Several nulls in the paper are power limits,
  and it says so where that is the case.
* The persona corpus is generated by a 3B open-weight model from ICD titles
  alone. It carries chapter-level and essentially no code-level specificity, and
  some findings are clinically wrong for the code they are keyed to. It is
  training signal, not a clinical resource.

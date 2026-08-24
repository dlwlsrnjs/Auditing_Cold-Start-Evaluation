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

## Data access, and what this repository can and cannot give you

MIMIC-IV is credentialed data under a PhysioNet data use agreement, so **no
patient-level file is in this repository**. What is here is:

* every script in the pipeline, from cohort construction to the audits;
* `mimic/out/results_mtl.jsonl` — per-run **aggregate** metrics (587 runs), with
  no identifiers and no row-level data. These are the numbers printed in the
  paper.

That is enough to **recompute every reported effect without any data access**
(`reproduce.py`). It is not enough to retrain: for that you need PhysioNet
credentialed access to MIMIC-IV v3.1 and MIMIC-IV-Note v2.2, after which the
scripts here rebuild everything. `mimic/out/README.md` lists exactly which
derived files are withheld and why.

Two analyses cannot be reproduced from the released file because they need
per-admission predictions, which are patient-level: the test-set bootstrap
(`bootstrap_primary.py`) and the disaggregated performance breakdown
(`subgroup_fairness.py`). Both scripts are included and run once you have
rebuilt `mimic/out/preds/`.

---

## Quick start — check the paper's numbers

```bash
pip install numpy pandas pyarrow scipy scikit-learn
python3 reproduce.py
```

This prints every headline effect recomputed from `results_mtl.jsonl` next to
the value printed in the paper, with a paired-*t* *p*-value, a 95% interval and
the seed count. A `!` marks any disagreement; a clean run prints none.

```bash
python3 reproduce.py audit2      # one section: audit1 audit2 audit3 q3 hospice
python3 mimic/count_ordering.py  # the "81 of 82 configurations" claim
```

### How the comparisons are formed

Every effect is a **paired difference across seeds within one tag series**.
Series differ in what they vary (patient split, metric set, seed count), and
**their C1 baselines differ by up to 0.31 aggregate AUROC points — more than
most effects in the paper**, so a cross-series delta is meaningless. Several tag
names are also misleading. The pairings the paper uses:

| claim | arm | control | seeds |
|---|---|---|---|
| oracle anchor `+0.33` | `sat-txt-oracle` | `sat-txt-c1` | 15 |
| prior anchor `−1.01` | `sat-txt-prior` | `sat-txt-c1` | 15 |
| permuted-within-group anchor `−0.38` | `rm-permanchor` | `rm-base` | 5 |
| raw simulator `−2.11` | `fx-inj-sim` | `fx-base` | 5 |
| recalibrated simulator `+0.01` | `fx-inj-simcal-exact` | `fx-base` | 5 |
| personas vs C1 `+0.17` | `rm-persona` | `rm-base` | 5 |
| personas vs C2, primary endpoint `+3.23` | `rm-persona` | `rm-naive` | 5 |
| text permutation `+2.52` | `rm-persona-shuf{,2,3,4}` | `rm-naive` | 5 × 4 draws |
| flattened labels `+0.02` | `rm-persona-flatlabel` | `rm-base` | 5 |
| labels permuted within code `−0.03` | `rm-persona-codeshuf` | `rm-persona` | 5 |
| labels permuted across codes `−0.15` | `rm-persona-chapshuf` | `rm-persona` | 5 |
| laboratory block `+21.21` AUPRC | `rm2-labs` | `rm-base` | 5 |
| test count alone `+4.61` AUPRC | `lg-labn` | `rm-base` | 5 |
| hospice-relabelled mortality | `hosp-persona` | `hosp-base` | 5 |

`rm-naive` is **C2**, the content-matched non-LLM control — not a naive
baseline. `rm-base` is C1. `sat-*` without `txt` is a different configuration
from `sat-txt-*`; pairing across the two gives a wrong answer.

---

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

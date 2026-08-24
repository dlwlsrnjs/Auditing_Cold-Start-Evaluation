# What is in this directory, and what is not

`results_mtl.jsonl` — one JSON object per training run: the configuration flags,
the seed, aggregate mortality/readmission/length-of-stay metrics, and the same
metrics inside each of the four segments. **No identifiers and no row-level
data.** These are the aggregate numbers printed in the paper, so `reproduce.py`
can recompute every reported effect from this file alone.

`eedi_audit.log` — output of `mimic/audit_eedi.py` on the public Eedi log.

Everything else the pipeline produces stays out of this repository, because
MIMIC-IV is credentialed data under a PhysioNet data use agreement:

| not released | why |
|---|---|
| `cohort_cens.parquet`, `labs48.parquet`, `notes_early.parquet` | patient-level records |
| `preds/*.parquet` | per-admission risk keyed by `hadm_id`, with the outcome |
| `note_emb.npy`, `dx_text_emb_*.npy` | embeddings of patient text |
| `sim_lora/` | a LoRA adapter fine-tuned on real note text with real labels — derived data, and a memorisation concern |
| `personas_*.parquet` | the synthetic corpus; the paper releases the generation pipeline rather than the corpus, since it is unvalidated clinical-sounding text keyed to real ICD codes |
| ICD-3 base rates, demographic marginals | MIMIC-derived aggregates over small cells |

With PhysioNet credentialed access you can rebuild all of them from the scripts
in `mimic/`; see the README at the repository root.

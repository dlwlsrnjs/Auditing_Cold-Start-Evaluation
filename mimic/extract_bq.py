#!/usr/bin/env python3
"""
Run the ColdHTL-on-MIMIC cohort query on BigQuery and pull results locally.
Prereqs: credentialed PhysioNet BigQuery access + `gcloud auth application-default login`.

    python3 extract_bq.py --project <YOUR_GCP_PROJECT_ID> --out out [--dataset mimiciv_v3_1]

Produces:
  out/cohort.parquet        hybrid targets + cold-start flags (admission level)
  out/notes_early.parquet   early (<=48h) radiology notes for the LLM simulator (leakage-safe)
Then prints a quick sanity EDA (base rates, funnel, cold-block sizes).
"""
import argparse, os, sys

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--project", required=True, help="Your billing GCP project id")
    ap.add_argument("--out", default="out")
    ap.add_argument("--dataset", default="mimiciv_v3_1",
                    help="version tag in dataset names: physionet-data.<tag>_hosp/_icu/_note")
    ap.add_argument("--sql", default=os.path.join(os.path.dirname(__file__), "cohort_hybrid_targets.sql"))
    ap.add_argument("--limit", type=int, default=0, help="row limit for a cheap dry test (0 = all)")
    args = ap.parse_args()

    try:
        from google.cloud import bigquery
    except ImportError:
        sys.exit("pip install google-cloud-bigquery db-dtypes  (and authenticate first)")

    os.makedirs(args.out, exist_ok=True)
    client = bigquery.Client(project=args.project)

    sql = open(args.sql).read()
    # allow overriding the version tag without editing the SQL
    sql = sql.replace("mimiciv_v3_1", args.dataset)
    if args.limit:
        sql = sql.rstrip().rstrip(";") + f"\nLIMIT {args.limit}"

    print("[1/3] running cohort query (dry-run cost estimate first)...")
    dry = client.query(sql, job_config=bigquery.QueryJobConfig(dry_run=True))
    print(f"      estimated scan: {dry.total_bytes_processed/1e9:.2f} GB")

    cohort = client.query(sql).result().to_dataframe(create_bqstorage_client=True)
    cohort.to_parquet(os.path.join(args.out, "cohort.parquet"), index=False)
    print(f"      cohort rows: {len(cohort):,} -> {args.out}/cohort.parquet")

    # ---- leakage-safe early notes: radiology within 48h of admission ----
    notes_sql = f"""
    WITH adm AS (
      SELECT subject_id, hadm_id, admittime
      FROM `physionet-data.{args.dataset}_hosp.admissions`
    )
    SELECT r.subject_id, r.hadm_id, r.note_id, r.charttime, r.text
    FROM `physionet-data.{args.dataset}_note.radiology` r
    JOIN adm USING (subject_id, hadm_id)
    WHERE r.charttime IS NOT NULL
      AND DATETIME_DIFF(r.charttime, adm.admittime, HOUR) BETWEEN 0 AND 48
    """
    print("[2/3] pulling early (<=48h) radiology notes (leakage-safe)...")
    try:
        notes = client.query(notes_sql).result().to_dataframe(create_bqstorage_client=True)
        notes.to_parquet(os.path.join(args.out, "notes_early.parquet"), index=False)
        print(f"      early notes: {len(notes):,} -> {args.out}/notes_early.parquet")
    except Exception as e:
        print(f"      [skip notes] {e}\n      (ensure MIMIC-IV-Note BigQuery access is granted separately)")

    print("[3/3] sanity EDA")
    n = len(cohort)
    def rate(col):
        s = cohort[col].dropna()
        return f"{s.mean()*100:.1f}% (n_def={len(s):,})" if len(s) else "n/a"
    print(f"  admissions           : {n:,}")
    print(f"  y_mortality          : {rate('y_mortality')}")
    print(f"  y_readmit_30d        : {rate('y_readmit_30d')}   <- undefined for deaths (funnel)")
    print(f"  y_los_days           : mean={cohort['y_los_days'].mean():.2f}  median={cohort['y_los_days'].median():.2f}")
    print(f"  cold_user (first adm): {cohort['cold_user_first_admission'].mean()*100:.1f}%")
    print(f"  cold_item (rare dx)  : {cohort['cold_item_rare_dx'].mean()*100:.1f}%")

if __name__ == "__main__":
    main()

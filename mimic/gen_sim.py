#!/usr/bin/env python3
"""Run the SFT'd outcome simulator on the dropped cold-dx train rows; emit label injections with
token-probability confidence. Lands directly inside the oracle-prior bracket for evaluation.
Output: mimic/out/inject_sim.parquet (hadm_id, y_mort, y_readmit, conf)"""
import argparse, time
import numpy as np, pandas as pd, torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

ap = argparse.ArgumentParser()
ap.add_argument("--outdir", default="mimic/out")
ap.add_argument("--model", default="Qwen/Qwen2.5-1.5B-Instruct")
ap.add_argument("--adapter", default="mimic/out/sim_lora")
ap.add_argument("--bs", type=int, default=128)
ap.add_argument("--no-patho", action="store_true")
ap.add_argument("--device", default="cuda:0")
ap.add_argument("--pilot", type=int, default=0, help="cap rows (pilot run)")
A = ap.parse_args()
t0 = time.time()

d = pd.read_parquet(f"{A.outdir}/cohort.parquet").reset_index(drop=True)
rng = np.random.default_rng(0); pids = d.subject_id.unique(); rng.shuffle(pids)
tr_p = set(pids[:int(.7 * len(pids))])
d["split"] = np.where(d.subject_id.isin(tr_p), "train", "other")
d.loc[(d.split == "train") & (d.cold_item_rare_dx == 1), "split"] = "drop"
drop = d[d.split == "drop"].reset_index(drop=True)
if A.pilot:
    drop = drop.sample(A.pilot, random_state=9).reset_index(drop=True)
notes = pd.read_parquet(f"{A.outdir}/notes_early.parquet").sort_values(["hadm_id", "charttime"])
note_txt = notes.groupby("hadm_id").text.apply(lambda s: "\n".join(s)).str.slice(0, 700)
gid = pd.read_parquet(f"{A.outdir}/dx_text_ids_gloss.parquet")
gmap = dict(zip(gid.primary_icd.astype(str), gid.gloss.astype(str))) if "gloss" in gid else {}
print(f"simulating outcomes for {len(drop):,} dropped cold-dx rows", flush=True)

SCAF = ("" if A.no_patho else
        "Weigh the pathophysiology of the principal diagnosis - the organ systems involved, "
        "typical complications, and decompensation paths - together with the imaging findings. ")
QT = {"mort": "Will this patient die during this hospital admission?",
      "readmit": "Will this patient be readmitted within 30 days after discharge?"}

tok = AutoTokenizer.from_pretrained(A.model); tok.padding_side = "left"
if tok.pad_token is None: tok.pad_token = tok.eos_token
llm = AutoModelForCausalLM.from_pretrained(A.model, torch_dtype=torch.float16).to(A.device)
llm = PeftModel.from_pretrained(llm, A.adapter).merge_and_unload().eval()
YES = tok("Yes", add_special_tokens=False)["input_ids"][0]
NO = tok("No", add_special_tokens=False)["input_ids"][0]

def prompt_of(r, task):
    note = note_txt.get(r.hadm_id, "(no imaging in first 48h)")
    gl = gmap.get(str(r.primary_icd), "")[:200]
    return (f"Patient: {r.gender}, age {int(r.age)}, admission type {r.admission_type}.\n"
            f"Principal diagnosis: {str(r.primary_dx_title)[:90]}. {gl}\n"
            f"First-48h imaging findings:\n{note}\n"
            f"{SCAF}{QT[task]} Answer Yes or No.")

@torch.no_grad()
def judge(rows, task):
    msgs = [[{"role": "user", "content": prompt_of(r, task)}] for _, r in rows.iterrows()]
    xs = [tok.apply_chat_template(m, add_generation_prompt=True, tokenize=False) for m in msgs]
    enc = tok(xs, return_tensors="pt", padding=True, truncation=True, max_length=640).to(A.device)
    logits = llm(**enc).logits[:, -1, :]
    py = torch.softmax(logits[:, [YES, NO]], dim=-1)[:, 0]     # P(Yes | Yes-or-No)
    return py.float().cpu().numpy()

out = {"mort": np.zeros(len(drop)), "readmit": np.zeros(len(drop))}
for task in ("mort", "readmit"):
    for k0 in range(0, len(drop), A.bs):
        out[task][k0:k0 + A.bs] = judge(drop.iloc[k0:k0 + A.bs], task)
        if k0 % (A.bs * 40) == 0:
            el = time.time() - t0
            print(f"  [{task}] {k0 + A.bs:,}/{len(drop):,} ({el/60:.1f}m)", flush=True)

inj = drop[["hadm_id"]].copy()
# soft mortality label (pilot: sim label AUROC .896 vs prior .713, but 2.5x over-calling at 0.5;
# soft targets absorb the calibration error). Readmission stays MASKED: pilot showed chance-level
# label quality (.56 vs prior .56) - injecting it would be noise.
inj["y_mort"] = out["mort"].astype(float)
inj["y_readmit"] = np.nan
inj["conf"] = 1.0
inj["p_mort"] = out["mort"]; inj["p_readmit"] = out["readmit"]
suf = "_pilot" if A.pilot else ""
inj.to_parquet(f"{A.outdir}/inject_sim{suf}.parquet", index=False)
print(f"[saved] inject_sim.parquet  sim mort rate={inj.y_mort.mean():.4f} "
      f"readmit rate={inj.y_readmit.mean():.4f}  ({(time.time()-t0)/60:.0f}m)", flush=True)

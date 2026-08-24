#!/usr/bin/env python3
"""ColdLLM-style outcome simulator: LoRA-SFT a judge that answers Yes/No outcome questions for an
admission, trained on WARM train rows with ColdLLM's balanced sampling (1:1 pos:neg per task).
The prompt carries a pathophysiology scaffold (--patho, default on): the judge is asked to weigh
mechanism and typical complications, not surface similarity.
Tasks: in-hospital mortality; 30-day readmission (observed rows only).
Output: LoRA adapter -> mimic/out/sim_lora"""
import argparse, time
import numpy as np, pandas as pd, torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, get_peft_model

ap = argparse.ArgumentParser()
ap.add_argument("--outdir", default="mimic/out")
ap.add_argument("--model", default="Qwen/Qwen2.5-1.5B-Instruct")
ap.add_argument("--per-task", type=int, default=12000, help="pairs per task per class")
ap.add_argument("--epochs", type=int, default=1)
ap.add_argument("--bs", type=int, default=8)
ap.add_argument("--accum", type=int, default=4)
ap.add_argument("--lr", type=float, default=5e-5)
ap.add_argument("--no-patho", action="store_true")
ap.add_argument("--device", default="cuda:0")
A = ap.parse_args()
t0 = time.time()

d = pd.read_parquet(f"{A.outdir}/cohort.parquet").reset_index(drop=True)
rng = np.random.default_rng(0); pids = d.subject_id.unique(); rng.shuffle(pids)
tr_p = set(pids[:int(.7 * len(pids))])
d["split"] = np.where(d.subject_id.isin(tr_p), "train", "other")
d.loc[(d.split == "train") & (d.cold_item_rare_dx == 1), "split"] = "drop"
warm = d[d.split == "train"].copy()
notes = pd.read_parquet(f"{A.outdir}/notes_early.parquet").sort_values(["hadm_id", "charttime"])
note_txt = notes.groupby("hadm_id").text.apply(lambda s: "\n".join(s)).str.slice(0, 700)
gid = pd.read_parquet(f"{A.outdir}/dx_text_ids_gloss.parquet")
gmap = dict(zip(gid.primary_icd.astype(str), gid.gloss.astype(str))) if "gloss" in gid else {}

SCAF = ("" if A.no_patho else
        "Weigh the pathophysiology of the principal diagnosis - the organ systems involved, "
        "typical complications, and decompensation paths - together with the imaging findings. ")
QT = {"mort": "Will this patient die during this hospital admission?",
      "readmit": "Will this patient be readmitted within 30 days after discharge?"}

def prompt_of(r, task):
    note = note_txt.get(r.hadm_id, "(no imaging in first 48h)")
    gl = gmap.get(str(r.primary_icd), "")[:200]
    return (f"Patient: {r.gender}, age {int(r.age)}, admission type {r.admission_type}.\n"
            f"Principal diagnosis: {str(r.primary_dx_title)[:90]}. {gl}\n"
            f"First-48h imaging findings:\n{note}\n"
            f"{SCAF}{QT[task]} Answer Yes or No.")

rs = np.random.default_rng(1)
pairs = []
wm = warm[warm.hadm_id.isin(note_txt.index)]
pos = wm[wm.y_mortality == 1]; neg = wm[wm.y_mortality == 0]
n1 = min(A.per_task, len(pos))
pairs += [(i, "mort", "Yes") for i in rs.choice(pos.index, n1, replace=False)]
pairs += [(i, "mort", "No") for i in rs.choice(neg.index, n1, replace=False)]
ro = wm[wm.y_readmit_30d.notna()]
pos = ro[ro.y_readmit_30d == 1]; neg = ro[ro.y_readmit_30d == 0]
n2 = min(A.per_task, len(pos), len(neg))
pairs += [(i, "readmit", "Yes") for i in rs.choice(pos.index, n2, replace=False)]
pairs += [(i, "readmit", "No") for i in rs.choice(neg.index, n2, replace=False)]
rs.shuffle(pairs)
print(f"SFT pairs: {len(pairs):,} (mort {2*n1:,}, readmit {2*n2:,}), patho={not A.no_patho}", flush=True)

tok = AutoTokenizer.from_pretrained(A.model)
if tok.pad_token is None: tok.pad_token = tok.eos_token

class DS(Dataset):
    def __len__(s): return len(pairs)
    def __getitem__(s, i):
        ix, task, ans = pairs[i]
        r = d.loc[ix]
        pre = tok.apply_chat_template([{"role": "user", "content": prompt_of(r, task)}],
                                      add_generation_prompt=True, tokenize=False)
        full = pre + ans + tok.eos_token
        ids = tok(full, truncation=True, max_length=640)["input_ids"]
        npre = len(tok(pre)["input_ids"])
        if npre > len(ids) - 2:
            cut = npre - (len(ids) - 2); ids = ids[:8] + ids[8 + cut:]; npre = max(npre - cut, 0)
        lab = [-100] * min(npre, len(ids)) + ids[min(npre, len(ids)):]
        return {"input_ids": ids, "labels": lab}

def collate(b):
    L = max(len(x["input_ids"]) for x in b)
    ids = torch.full((len(b), L), tok.pad_token_id, dtype=torch.long)
    lab = torch.full((len(b), L), -100, dtype=torch.long)
    att = torch.zeros((len(b), L), dtype=torch.long)
    for i, x in enumerate(b):
        k = len(x["input_ids"])
        ids[i, :k] = torch.tensor(x["input_ids"]); lab[i, :k] = torch.tensor(x["labels"]); att[i, :k] = 1
    return ids, att, lab

model = AutoModelForCausalLM.from_pretrained(A.model, torch_dtype=torch.bfloat16).to(A.device)
model.gradient_checkpointing_enable(); model.enable_input_require_grads()
model = get_peft_model(model, LoraConfig(r=16, lora_alpha=32, lora_dropout=0.05, task_type="CAUSAL_LM",
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]))
model.print_trainable_parameters()
dl = DataLoader(DS(), batch_size=A.bs, shuffle=True, collate_fn=collate, num_workers=4)
opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=A.lr)
model.train(); k = 0; nbad = 0; t1 = time.time()
for ep in range(A.epochs):
    for ids, att, lab in dl:
        loss = model(input_ids=ids.to(A.device), attention_mask=att.to(A.device),
                     labels=lab.to(A.device)).loss / A.accum
        if torch.isnan(loss) or torch.isinf(loss):
            opt.zero_grad(); k += 1; continue
        loss.backward(); k += 1
        if k % A.accum == 0:
            gn = torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], 1.0)
            if torch.isfinite(gn): opt.step()
            else: nbad += 1
            opt.zero_grad()
        if k % 200 == 0:
            el = time.time() - t1
            print(f"  step {k}/{len(dl)*A.epochs} loss={loss.item()*A.accum:.3f} bad={nbad} "
                  f"({el:.0f}s, eta {(len(dl)*A.epochs-k)*el/k/60:.0f}m)", flush=True)
model.save_pretrained(f"{A.outdir}/sim_lora")
print(f"[saved] sim_lora ({(time.time()-t0)/60:.0f}m)", flush=True)

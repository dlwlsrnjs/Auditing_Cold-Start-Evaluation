#!/usr/bin/env python3
"""LoRA-SFT a course-forecasting agent: first-48h inputs -> the admission's REAL hospital course.

The discharge summary is used ONLY as a training TARGET on the fixed TRAIN split (never as an
input, never outside train), per SETUP.md's leakage rule. The trained agent distills the dense
clinical supervision of the course narrative (complications, procedures, trajectory) that the
three scalar labels cannot carry; at inference it sees only 48h inputs.

Usage: python3 mimic/sft_course.py [--sample 60000] [--epochs 1]
"""
import argparse, gzip, io, re, time
import numpy as np, pandas as pd, torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, get_peft_model

ap = argparse.ArgumentParser()
ap.add_argument("--cohort", default="mimic/out/cohort.parquet")
ap.add_argument("--notes", default="mimic/out/notes_early.parquet")
ap.add_argument("--discharge", default="mimic/physionet.org/files/mimic-iv-note/2.2/note/discharge.csv.gz")
ap.add_argument("--outdir", default="mimic/out/course_lora")
ap.add_argument("--model", default="Qwen/Qwen2.5-1.5B-Instruct")
ap.add_argument("--sample", type=int, default=60000)
ap.add_argument("--epochs", type=int, default=1)
ap.add_argument("--bs", type=int, default=16)
ap.add_argument("--accum", type=int, default=2)
ap.add_argument("--lr", type=float, default=1e-4)
ap.add_argument("--maxlen", type=int, default=1024)
ap.add_argument("--device", default="cuda:0")
ap.add_argument("--dtype", choices=["bf16","fp32"], default="bf16")
ap.add_argument("--attn", choices=["sdpa","eager"], default="sdpa")
A = ap.parse_args()
t0 = time.time()

d = pd.read_parquet(A.cohort).reset_index(drop=True)
rng = np.random.default_rng(0)                       # FIXED split, mirrors train_mtl --fixed-split
pids = d.subject_id.unique(); rng.shuffle(pids)
n = len(pids); tr_p = set(pids[:int(.7 * n)])
d["split"] = np.where(d.subject_id.isin(tr_p), "train", "other")
d.loc[(d.split == "train") & (d.cold_item_rare_dx == 1), "split"] = "drop"
train_h = set(d[d.split == "train"].hadm_id)

notes = pd.read_parquet(A.notes).sort_values(["hadm_id", "charttime"])
note_txt = notes.groupby("hadm_id").text.apply(lambda s: "\n".join(s)).str.slice(0, 1000)

print("streaming discharge summaries (train split only)...", flush=True)
courses = {}
reader = pd.read_csv(A.discharge, chunksize=20000, usecols=["hadm_id", "text"])
for ch in reader:
    ch = ch[ch.hadm_id.isin(train_h) & ch.hadm_id.isin(note_txt.index)]
    for h, t in zip(ch.hadm_id, ch.text):
        m = re.search(r'Brief Hospital Course:\s*\n?(.*?)(?:\n\s*\n[A-Z][a-zA-Z /]+:|\Z)', str(t), re.S)
        if m:
            c = re.sub(r'_{2,}', '', m.group(1)); c = re.sub(r'\n{2,}', '\n', c).strip()
            if len(c) > 200: courses[h] = c[:1400]
print(f"course targets: {len(courses):,} train admissions w/ early note + course "
      f"({time.time()-t0:.0f}s)", flush=True)

D = d.set_index("hadm_id")
dxt = D.primary_dx_title.fillna("unknown").astype(str)
hs = list(courses)
rs = np.random.default_rng(1); rs.shuffle(hs); hs = hs[:A.sample]

PROMPT = ("You are a clinical course forecaster. Using only information from the first 48 hours, "
          "predict this admission's hospital course.\n"
          "Patient: {g}, age {a}, admission type {t}, working diagnosis: {dx}.\n"
          "Early radiology findings (first 48h):\n{note}\n"
          "Predicted hospital course:")

tok = AutoTokenizer.from_pretrained(A.model)
if tok.pad_token is None: tok.pad_token = tok.eos_token

class DS(Dataset):
    def __len__(s): return len(hs)
    def __getitem__(s, i):
        h = hs[i]; r = D.loc[h]
        p = PROMPT.format(g=r.gender, a=int(r.age), t=r.admission_type,
                          dx=dxt.loc[h][:90], note=note_txt.loc[h])
        msgs = [{"role": "user", "content": p}]
        pre = tok.apply_chat_template(msgs, add_generation_prompt=True, tokenize=False)
        full = pre + courses[h] + tok.eos_token
        enc = tok(full, truncation=True, max_length=A.maxlen, return_tensors=None)
        npre = len(tok(pre, return_tensors=None)["input_ids"])
        ids = enc["input_ids"]
        # guard: a prompt that fills maxlen leaves zero supervised tokens -> 0/0 = NaN loss.
        # keep at least 32 target tokens by trimming the prompt from the left if needed.
        if npre > len(ids) - 32:
            cut = npre - (len(ids) - 32)
            ids = ids[:8] + ids[8 + cut:]          # keep BOS-ish head, drop middle of prompt
            npre = max(npre - cut, 0)
        labels = [-100] * min(npre, len(ids)) + ids[min(npre, len(ids)):]
        return {"input_ids": ids, "labels": labels}

def collate(b):
    L = max(len(x["input_ids"]) for x in b)
    ids = torch.full((len(b), L), tok.pad_token_id, dtype=torch.long)
    lab = torch.full((len(b), L), -100, dtype=torch.long)
    att = torch.zeros((len(b), L), dtype=torch.long)
    for i, x in enumerate(b):
        k = len(x["input_ids"])
        ids[i, :k] = torch.tensor(x["input_ids"]); lab[i, :k] = torch.tensor(x["labels"])
        att[i, :k] = 1
    return ids, att, lab

model = AutoModelForCausalLM.from_pretrained(
    A.model, torch_dtype=torch.bfloat16 if A.dtype == "bf16" else torch.float32,
    attn_implementation=A.attn).to(A.device)
model.gradient_checkpointing_enable()
model.enable_input_require_grads()
lcfg = LoraConfig(r=16, lora_alpha=32, lora_dropout=0.05, task_type="CAUSAL_LM",
                  target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                                  "gate_proj", "up_proj", "down_proj"])
model = get_peft_model(model, lcfg)
model.print_trainable_parameters()

dl = DataLoader(DS(), batch_size=A.bs, shuffle=True, collate_fn=collate, num_workers=4)
opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=A.lr)
steps = len(dl) * A.epochs // A.accum
sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(steps, 1))
model.train(); k = 0; nskip = 0; nbad = 0; t1 = time.time()
for ep in range(A.epochs):
    for ids, att, lab in dl:
        ids, att, lab = ids.to(A.device), att.to(A.device), lab.to(A.device)
        loss = model(input_ids=ids, attention_mask=att, labels=lab).loss / A.accum
        if torch.isnan(loss) or torch.isinf(loss):
            opt.zero_grad(); nskip += 1; k += 1
            if nskip % 20 == 1: print(f"  [warn] skipped {nskip} non-finite batches", flush=True)
            continue
        loss.backward(); k += 1
        if k % A.accum == 0:
            # clip_grad_norm_ POISONS weights when the norm is inf/nan (0*inf) - guard the step
            gn = torch.nn.utils.clip_grad_norm_(
                [p for p in model.parameters() if p.requires_grad], 1.0)
            if torch.isfinite(gn):
                opt.step(); sched.step()
            else:
                nbad += 1
                if nbad % 5 == 1: print(f"  [warn] dropped {nbad} non-finite grad steps", flush=True)
            opt.zero_grad()
        if k % 100 == 0:
            el = time.time() - t1
            print(f"  step {k}/{len(dl)*A.epochs} loss={loss.item()*A.accum:.3f} "
                  f"({el:.0f}s, eta {(len(dl)*A.epochs-k)*el/k/60:.0f}m)", flush=True)
model.save_pretrained(A.outdir)
print(f"[saved] LoRA adapter -> {A.outdir} ({(time.time()-t0)/60:.0f}m)", flush=True)

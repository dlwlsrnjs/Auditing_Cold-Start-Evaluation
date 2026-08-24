#!/usr/bin/env python3
"""Generate predicted hospital-course text for the evaluation target set, with the SFT agent
(--mode sft) or the same base model zero-shot (--mode zs, the pre-registered control).
Same prompt, same targets, same decoding in both modes -> the only difference is the training.
Output: course_emb_{sft|zs}.npy + course_ids_{sft|zs}.parquet (train_mtl --agentemb interface).
"""
import argparse, re, time
import numpy as np, pandas as pd, torch
from transformers import AutoModel, AutoModelForCausalLM, AutoTokenizer

ap = argparse.ArgumentParser()
ap.add_argument("--outdir", default="mimic/out")
ap.add_argument("--cohort", default="mimic/out/cohort.parquet")
ap.add_argument("--notes", default="mimic/out/notes_early.parquet")
ap.add_argument("--mode", choices=["sft", "zs"], default="sft")
ap.add_argument("--model", default="Qwen/Qwen2.5-1.5B-Instruct")
ap.add_argument("--adapter", default="mimic/out/course_lora")
ap.add_argument("--train-sample", type=int, default=25000)
ap.add_argument("--bs", type=int, default=96)
ap.add_argument("--max-new", type=int, default=220)
ap.add_argument("--device", default="cuda:0")
A = ap.parse_args()
t0 = time.time()

d = pd.read_parquet(A.cohort).reset_index(drop=True)
rng = np.random.default_rng(0)
pids = d.subject_id.unique(); rng.shuffle(pids)
n = len(pids); tr_p = set(pids[:int(.7 * n)]); va_p = set(pids[int(.7 * n):int(.8 * n)])
d["split"] = np.where(d.subject_id.isin(tr_p), "train",
             np.where(d.subject_id.isin(va_p), "val", "test"))
d.loc[(d.split == "train") & (d.cold_item_rare_dx == 1), "split"] = "drop"

notes = pd.read_parquet(A.notes).sort_values(["hadm_id", "charttime"])
note_txt = notes.groupby("hadm_id").text.apply(lambda s: "\n".join(s)).str.slice(0, 1000)
has_note = d.hadm_id.isin(note_txt.index).values

# same target policy as agent_evidence.py: all val/test cold-block admissions w/ note + train sample
tgt_mask = ((d.split.isin(["val", "test"])) &
            ((d.cold_item_rare_dx == 1) | (d.cold_user_first_admission == 1)) & has_note)
tr_note = np.where((d.split == "train").values & has_note)[0]
rs = np.random.default_rng(1)
tr_sample = rs.choice(tr_note, size=min(A.train_sample, len(tr_note)), replace=False)
targets = np.unique(np.concatenate([np.where(tgt_mask.values)[0], tr_sample]))
print(f"mode={A.mode} targets={len(targets):,}", flush=True)

dxt = d.primary_dx_title.fillna("unknown").astype(str)
PROMPT = ("You are a clinical course forecaster. Using only information from the first 48 hours, "
          "predict this admission's hospital course.\n"
          "Patient: {g}, age {a}, admission type {t}, working diagnosis: {dx}.\n"
          "Early radiology findings (first 48h):\n{note}\n"
          "Predicted hospital course:")

tok = AutoTokenizer.from_pretrained(A.model); tok.padding_side = "left"
if tok.pad_token is None: tok.pad_token = tok.eos_token
llm = AutoModelForCausalLM.from_pretrained(A.model, torch_dtype=torch.float16).to(A.device)
if A.mode == "sft":
    from peft import PeftModel
    llm = PeftModel.from_pretrained(llm, A.adapter).merge_and_unload()
llm.eval()

def prompt_of(i):
    r = d.iloc[i]
    return PROMPT.format(g=r.gender, a=int(r.age), t=r.admission_type,
                         dx=dxt.iloc[i][:90], note=note_txt.get(r.hadm_id, "(none)"))

@torch.no_grad()
def gen(rows):
    msgs = [[{"role": "user", "content": prompt_of(i)}] for i in rows]
    xs = [tok.apply_chat_template(m, add_generation_prompt=True, tokenize=False) for m in msgs]
    enc = tok(xs, return_tensors="pt", padding=True, truncation=True, max_length=768).to(A.device)
    y = llm.generate(**enc, max_new_tokens=A.max_new, do_sample=False, pad_token_id=tok.pad_token_id)
    return tok.batch_decode(y[:, enc["input_ids"].shape[1]:], skip_special_tokens=True)

texts = []
for k0 in range(0, len(targets), A.bs):
    texts += [re.sub(r"\s+", " ", t).strip() for t in gen(targets[k0:k0 + A.bs])]
    if k0 % (A.bs * 20) == 0:
        el = time.time() - t0; done = k0 + A.bs
        print(f"  {done:,}/{len(targets):,} ({el/60:.0f}m, eta {(len(targets)-done)*el/max(done,1)/60:.0f}m)",
              flush=True)
del llm; torch.cuda.empty_cache()

enc_model = AutoModel.from_pretrained(A.model, torch_dtype=torch.float16).to(A.device).eval()
tok2 = AutoTokenizer.from_pretrained(A.model)
@torch.no_grad()
def embed(b):
    e = tok2(b, padding=True, truncation=True, max_length=320, return_tensors="pt").to(A.device)
    o = enc_model(**e).last_hidden_state
    m = e.attention_mask.unsqueeze(-1).float()
    v = (o * m).sum(1) / m.sum(1).clamp(min=1)
    return torch.nn.functional.normalize(v, dim=-1).float().cpu().numpy()
E = np.concatenate([embed(texts[i:i + 128]) for i in range(0, len(texts), 128)], 0).astype(np.float32)

np.save(f"{A.outdir}/course_emb_{A.mode}.npy", E)
out = d.iloc[targets][["hadm_id"]].copy(); out["course_text"] = texts
out.to_parquet(f"{A.outdir}/course_ids_{A.mode}.parquet", index=False)
print(f"[saved] course_emb_{A.mode}.npy {E.shape} ({(time.time()-t0)/60:.0f}m)", flush=True)

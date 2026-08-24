#!/usr/bin/env python3
"""Retrieval-grounded agent evidence for cold-start admissions.

For each target admission the agent (i) retrieves the K most similar admissions from the FIXED
train split (note+dx-text space, same-patient excluded), (ii) reads their REAL outcomes
(died / 30-day readmission / LOS), and (iii) writes a short risk assessment synthesising the
target's early presentation against that real evidence. The text is embedded (same frozen
encoder as every other block) -> agent_emb.npy, availability-gated into the model.

The information channel is what distinguishes this from the content-only LLM augmentation that
fails under clean evaluation: the agent imports real cross-patient label evidence, not
hallucinated behavior. The matching control (--mode noretr) gives the same LLM the same target
WITHOUT the retrieved cases; if the full agent does not beat it AND the numeric kNN block,
the reasoning step adds nothing and we report that.

Split protocol: retrieval pool = train patients under the FIXED split (rng(0), --fixed-split in
train_mtl.py). Targets = all val/test cold-block admissions with an early note + a train sample.

Usage:
  python3 mimic/agent_evidence.py --mode agent   # retrieval-grounded (main)
  python3 mimic/agent_evidence.py --mode noretr  # no-retrieval control
"""
import argparse, re, time
import numpy as np, pandas as pd, torch
from transformers import AutoModel, AutoModelForCausalLM, AutoTokenizer

ap = argparse.ArgumentParser()
ap.add_argument("--outdir", default="mimic/out")
ap.add_argument("--cohort", default="mimic/out/cohort.parquet")
ap.add_argument("--notes", default="mimic/out/notes_early.parquet")
ap.add_argument("--mode", choices=["agent", "noretr"], default="agent")
ap.add_argument("--model", default="Qwen/Qwen2.5-3B-Instruct")
ap.add_argument("--k", type=int, default=5)
ap.add_argument("--train-sample", type=int, default=40000)
ap.add_argument("--bs", type=int, default=64)
ap.add_argument("--max-new", type=int, default=110)
ap.add_argument("--device", default="cuda:0")
A = ap.parse_args()
t0 = time.time()

d = pd.read_parquet(A.cohort).reset_index(drop=True)
# FIXED split (must mirror train_mtl.py --fixed-split exactly)
rng = np.random.default_rng(0)
pids = d.subject_id.unique(); rng.shuffle(pids)
n = len(pids); tr_p = set(pids[:int(.7 * n)]); va_p = set(pids[int(.7 * n):int(.8 * n)])
d["split"] = np.where(d.subject_id.isin(tr_p), "train",
             np.where(d.subject_id.isin(va_p), "val", "test"))
d.loc[(d.split == "train") & (d.cold_item_rare_dx == 1), "split"] = "drop"

# embedding space = [note ; dx-gloss], mirroring the in-model retrieval block
ne = np.load(f"{A.outdir}/note_emb.npy"); nid = pd.read_parquet(f"{A.outdir}/note_emb_ids.parquet").hadm_id.values
npos = {h: i for i, h in enumerate(nid)}
nidx = d.hadm_id.map(npos)
NOTE = np.zeros((len(d), ne.shape[1]), np.float32)
has_note = nidx.notna().values
NOTE[has_note] = ne[nidx[has_note].astype(int).values]
ge = np.load(f"{A.outdir}/dx_text_emb_gloss.npy")
gid = pd.read_parquet(f"{A.outdir}/dx_text_ids_gloss.parquet").primary_icd.astype(str).values
gpos = {c: i for i, c in enumerate(gid)}
gidx = d.primary_icd.astype(str).map(gpos)
DXG = np.zeros((len(d), ge.shape[1]), np.float32)
gok = gidx.notna().values
DXG[gok] = ge[gidx[gok].astype(int).values]
Q = np.concatenate([NOTE, DXG], 1); Q /= np.linalg.norm(Q, axis=1, keepdims=True) + 1e-9

# targets: every val/test cold-block admission with a note + a train sample (training signal)
tgt_mask = ((d.split.isin(["val", "test"])) &
            ((d.cold_item_rare_dx == 1) | (d.cold_user_first_admission == 1)) & has_note)
tr_note = np.where((d.split == "train").values & has_note)[0]
rs = np.random.default_rng(1)
tr_sample = rs.choice(tr_note, size=min(A.train_sample, len(tr_note)), replace=False)
targets = np.concatenate([np.where(tgt_mask.values)[0], tr_sample])
targets = np.unique(targets)
print(f"targets={len(targets):,} (val/test cold w/ note: {int(tgt_mask.sum()):,} + train sample)", flush=True)

# retrieval (train pool, same-patient excluded)
tr_idx = np.where((d.split == "train").values)[0]
DEV = A.device
Qg = torch.tensor(Q, device=DEV, dtype=torch.float16)
Ng = Qg[tr_idx]
subj = d.subject_id.values
tr_subj = torch.tensor(subj[tr_idx].astype(np.int64), device=DEV)
nb_idx = np.zeros((len(targets), A.k), np.int64)
for k0 in range(0, len(targets), 2048):
    tt = targets[k0:k0 + 2048]
    sims = (Qg[tt] @ Ng.T).float()
    qs = torch.tensor(subj[tt].astype(np.int64), device=DEV)
    sims[qs.unsqueeze(1) == tr_subj.unsqueeze(0)] = -1e4
    nb_idx[k0:k0 + 2048] = tr_idx[torch.topk(sims, A.k, 1).indices.cpu().numpy()]
del Qg, Ng; torch.cuda.empty_cache()
print(f"retrieval done ({time.time()-t0:.0f}s)", flush=True)

notes = pd.read_parquet(A.notes).sort_values(["hadm_id", "charttime"])
note_txt = notes.groupby("hadm_id").text.apply(lambda s: "\n".join(s)).str.slice(0, 1200)
dxt = d.primary_dx_title.fillna("unknown").astype(str)

def case_line(j):
    r = d.iloc[j]
    out = "died in hospital" if r.y_mortality == 1 else (
        f"survived, LOS {r.y_los_days:.0f}d" +
        (", readmitted within 30d" if r.y_readmit_30d == 1 else
         (", no 30d readmission" if r.y_readmit_30d == 0 else "")))
    return f"- {r.gender}, age {int(r.age)}, dx: {dxt.iloc[j][:80]} -> {out}"

def build_prompt(i, row_idx):
    r = d.iloc[row_idx]
    head = (f"Patient: {r.gender}, age {int(r.age)}, admission type {r.admission_type}, "
            f"working diagnosis: {dxt.iloc[row_idx][:90]}.\n"
            f"Early radiology findings (first 48h):\n{note_txt.get(r.hadm_id, '(none)')}\n")
    if A.mode == "agent":
        cases = "\n".join(case_line(j) for j in nb_idx[i])
        head += f"\nMost similar past admissions and their REAL outcomes:\n{cases}\n"
        ask = ("Weighing this patient's findings against those real outcomes, write a 3-sentence "
               "risk assessment: likely severity, in-hospital mortality risk, expected length of "
               "stay, and 30-day readmission risk. Plain prose.")
    else:
        ask = ("Write a 3-sentence risk assessment: likely severity, in-hospital mortality risk, "
               "expected length of stay, and 30-day readmission risk. Plain prose.")
    return head + ask

tok = AutoTokenizer.from_pretrained(A.model); tok.padding_side = "left"
if tok.pad_token is None: tok.pad_token = tok.eos_token
llm = AutoModelForCausalLM.from_pretrained(A.model, torch_dtype=torch.float16).to(DEV).eval()

@torch.no_grad()
def gen(batch_rows, batch_pos):
    msgs = [[{"role": "user", "content": build_prompt(p, r)}] for p, r in zip(batch_pos, batch_rows)]
    xs = [tok.apply_chat_template(m, add_generation_prompt=True, tokenize=False) for m in msgs]
    enc = tok(xs, return_tensors="pt", padding=True, truncation=True, max_length=1024).to(DEV)
    y = llm.generate(**enc, max_new_tokens=A.max_new, do_sample=False, pad_token_id=tok.pad_token_id)
    return tok.batch_decode(y[:, enc["input_ids"].shape[1]:], skip_special_tokens=True)

texts = []
for k0 in range(0, len(targets), A.bs):
    rows = targets[k0:k0 + A.bs]; pos = list(range(k0, min(k0 + A.bs, len(targets))))
    texts += [re.sub(r"\s+", " ", t).strip() for t in gen(rows, pos)]
    if k0 % (A.bs * 20) == 0:
        el = time.time() - t0; done = k0 + len(rows)
        print(f"  {done:,}/{len(targets):,} ({el/60:.0f}m, eta {(len(targets)-done)*el/max(done,1)/60:.0f}m)",
              flush=True)
del llm; torch.cuda.empty_cache()

enc_model = AutoModel.from_pretrained("Qwen/Qwen2.5-1.5B-Instruct",
                                      torch_dtype=torch.float16).to(DEV).eval()
tok2 = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-1.5B-Instruct")
@torch.no_grad()
def embed(b):
    e = tok2(b, padding=True, truncation=True, max_length=256, return_tensors="pt").to(DEV)
    o = enc_model(**e).last_hidden_state
    m = e.attention_mask.unsqueeze(-1).float()
    v = (o * m).sum(1) / m.sum(1).clamp(min=1)
    return torch.nn.functional.normalize(v, dim=-1).float().cpu().numpy()
E = np.concatenate([embed(texts[i:i + 128]) for i in range(0, len(texts), 128)], 0).astype(np.float32)

suf = "agent" if A.mode == "agent" else "noretr"
np.save(f"{A.outdir}/agent_emb_{suf}.npy", E)
out = d.iloc[targets][["hadm_id"]].copy(); out["evidence_text"] = texts
out.to_parquet(f"{A.outdir}/agent_ids_{suf}.parquet", index=False)
print(f"[saved] agent_emb_{suf}.npy {E.shape}  ({(time.time()-t0)/60:.0f}m)", flush=True)

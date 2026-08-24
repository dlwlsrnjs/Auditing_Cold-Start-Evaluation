#!/usr/bin/env python3
"""Re-embed all text channels with a dedicated embedding model (gte-Qwen2-1.5B-instruct,
last-token pooling + instruction prefix) - replaces the generative-LM mean-pool that has been
capping every text-derived channel. Outputs *_gte.npy aligned with existing id files."""
import argparse, time
import numpy as np, pandas as pd, torch
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer

ap = argparse.ArgumentParser()
ap.add_argument("--outdir", default="mimic/out")
ap.add_argument("--model", default="nomic-ai/nomic-embed-text-v1.5")
ap.add_argument("--prefix", default="search_document: ",
                help="nomic requires a task prefix on every input text")
ap.add_argument("--which", choices=["notes", "dx", "personas", "all"], default="all")
ap.add_argument("--bs", type=int, default=48)
ap.add_argument("--maxtok", type=int, default=2048)
ap.add_argument("--device", default="cuda:0")
A = ap.parse_args()
t0 = time.time()

tok = AutoTokenizer.from_pretrained(A.model)
model = AutoModel.from_pretrained(A.model, trust_remote_code=True,
                                  torch_dtype=torch.float16).to(A.device).eval()

@torch.no_grad()
def embed(texts, maxtok):
    texts = [A.prefix + t for t in texts]
    e = tok(texts, padding=True, truncation=True, max_length=maxtok, return_tensors="pt").to(A.device)
    o = model(**e).last_hidden_state
    m = e.attention_mask.unsqueeze(-1).float()
    v = (o * m).sum(1) / m.sum(1).clamp(min=1)      # nomic convention: mean pooling
    return F.normalize(v, dim=-1).float().cpu().numpy()

def run(texts, out, maxtok):
    E = []
    for i in range(0, len(texts), A.bs):
        E.append(embed(texts[i:i + A.bs], maxtok))
        if i % (A.bs * 100) == 0:
            el = time.time() - t0
            print(f"  {out}: {i:,}/{len(texts):,} ({el/60:.0f}m)", flush=True)
    E = np.concatenate(E, 0).astype(np.float32)
    np.save(f"{A.outdir}/{out}", E)
    print(f"[saved] {out} {E.shape}", flush=True)

if A.which in ("notes", "all"):
    notes = pd.read_parquet(f"{A.outdir.replace('/out','')}/out/notes_early.parquet")
    notes = notes.sort_values(["hadm_id", "charttime"])
    docs = (notes.groupby("hadm_id").text.apply(lambda s: "\n\n".join(s)).str.slice(0, 6000))
    ids = pd.read_parquet(f"{A.outdir}/note_emb_ids.parquet").hadm_id
    docs = docs.reindex(ids.values)                     # exact alignment with existing id file
    run(docs.fillna("").tolist(), "note_emb_gte.npy", A.maxtok)
if A.which in ("dx", "all"):
    gid = pd.read_parquet(f"{A.outdir}/dx_text_ids_gloss.parquet")
    d = pd.read_parquet(f"{A.outdir}/cohort.parquet")[["primary_icd", "primary_dx_title"]]
    tmap = dict(zip(d.primary_icd.astype(str), d.primary_dx_title.astype(str)))
    texts = [f"{tmap.get(str(c),'')}. {g}"[:1200] for c, g in
             zip(gid.primary_icd.astype(str), gid.get("gloss", [""] * len(gid)))]
    run(texts, "dx_text_emb_gte.npy", 320)
if A.which in ("personas", "all"):
    per = pd.read_parquet(f"{A.outdir}/personas_llm.parquet")
    run(per.findings.astype(str).tolist(), "persona_emb_gte.npy", 320)
print(f"total {(time.time()-t0)/60:.0f}m", flush=True)

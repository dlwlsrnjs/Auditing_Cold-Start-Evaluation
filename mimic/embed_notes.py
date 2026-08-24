#!/usr/bin/env python3
"""Frozen per-admission embeddings of early (48h) radiology notes.

Same convention as mind/embed_items.py: Qwen2.5-1.5B mean-pool, L2-normalised, fp32 .npy.
Notes are ordered by charttime inside each admission and concatenated up to --max-chars;
one embedding per admission with >=1 early note. Output aligns row i of note_emb.npy with
row i of note_emb_ids.parquet (hadm_id) - the interface train_mtl.py --textemb expects.

Usage: python3 mimic/embed_notes.py [--notes mimic/out/notes_early.parquet]
"""
import argparse, time
import numpy as np, pandas as pd, torch
from transformers import AutoModel, AutoTokenizer

ap = argparse.ArgumentParser()
ap.add_argument("--notes", default="mimic/out/notes_early.parquet")
ap.add_argument("--out-emb", default="mimic/out/note_emb.npy")
ap.add_argument("--out-ids", default="mimic/out/note_emb_ids.parquet")
ap.add_argument("--model", default="Qwen/Qwen2.5-1.5B-Instruct")
ap.add_argument("--max-chars", type=int, default=6000, help="per-admission concatenated text cap")
ap.add_argument("--max-tokens", type=int, default=1024)
ap.add_argument("--bs", type=int, default=64)
ap.add_argument("--device", default="cuda:0")
A = ap.parse_args()

t0 = time.time()
n = pd.read_parquet(A.notes)
n = n.sort_values(["hadm_id", "charttime"])
docs = (n.groupby("hadm_id").text.apply(lambda s: "\n\n".join(s))
        .str.slice(0, A.max_chars).reset_index())
print(f"{len(n):,} notes -> {len(docs):,} admissions "
      f"(mean {docs.text.str.len().mean():.0f} chars/doc)", flush=True)

tok = AutoTokenizer.from_pretrained(A.model)
model = AutoModel.from_pretrained(A.model, torch_dtype=torch.float16).to(A.device).eval()

@torch.no_grad()
def embed(batch):
    e = tok(batch, padding=True, truncation=True, max_length=A.max_tokens,
            return_tensors="pt").to(A.device)
    o = model(**e).last_hidden_state
    m = e.attention_mask.unsqueeze(-1).float()
    v = (o * m).sum(1) / m.sum(1).clamp(min=1)
    return torch.nn.functional.normalize(v, dim=-1).float().cpu().numpy()

texts = docs.text.tolist()
E = []
for i in range(0, len(texts), A.bs):
    E.append(embed(texts[i:i + A.bs]))
    if i % (A.bs * 50) == 0:
        el = time.time() - t0
        done = i + A.bs
        print(f"  {i:,}/{len(texts):,}  ({el:.0f}s, eta {(len(texts)-done)*el/max(done,1)/60:.0f}m)",
              flush=True)
E = np.concatenate(E, 0).astype(np.float32)
np.save(A.out_emb, E)
docs[["hadm_id"]].to_parquet(A.out_ids, index=False)
print(f"[saved] {A.out_emb} {E.shape}  +  {A.out_ids}  ({time.time()-t0:.0f}s)", flush=True)

#!/usr/bin/env python3
"""Diagnosis-text embeddings: one vector per unique primary ICD code.

Two modes:
  title (default) - embed the official ICD long_title. Available for EVERY code, including the
                    cold ones whose id embedding never trains -> content-based representation
                    that generalises to the cold-dx block by construction.
  gloss           - first ask Qwen to expand each title into a 2-sentence clinical gloss
                    (typical presentation, acuity, expected course), then embed title+gloss.
                    The MIMIC analog of the representation-enrichment residual that survived
                    clean evaluation in the Mirage paper.

Output: dx_text_emb[_gloss].npy aligned with dx_text_ids[_gloss].parquet (primary_icd).
"""
import argparse, time, re
import numpy as np, pandas as pd, torch
from transformers import AutoModel, AutoModelForCausalLM, AutoTokenizer

ap = argparse.ArgumentParser()
ap.add_argument("--cohort", default="mimic/out/cohort.parquet")
ap.add_argument("--outdir", default="mimic/out")
ap.add_argument("--mode", choices=["title", "gloss"], default="title")
ap.add_argument("--model", default="Qwen/Qwen2.5-1.5B-Instruct")
ap.add_argument("--bs", type=int, default=256)
ap.add_argument("--gloss-bs", type=int, default=96)
ap.add_argument("--device", default="cuda:0")
A = ap.parse_args()
t0 = time.time()

d = pd.read_parquet(A.cohort)
dx = (d[["primary_icd", "primary_icd_version", "primary_dx_title"]]
      .dropna(subset=["primary_icd"]).drop_duplicates("primary_icd").reset_index(drop=True))
dx["title"] = dx.primary_dx_title.fillna("unknown diagnosis").astype(str)
print(f"{len(dx):,} unique primary ICD codes, mode={A.mode}", flush=True)

tok = AutoTokenizer.from_pretrained(A.model)
texts = dx.title.tolist()

if A.mode == "gloss":
    tok.padding_side = "left"
    if tok.pad_token is None: tok.pad_token = tok.eos_token
    llm = AutoModelForCausalLM.from_pretrained(A.model, torch_dtype=torch.float16).to(A.device).eval()
    @torch.no_grad()
    def gloss_batch(titles):
        msgs = [[{"role": "user", "content":
                  f'Diagnosis: "{t}". In 2 short sentences, describe the typical clinical '
                  f'presentation, severity, and expected hospital course. Plain prose only.'}]
                for t in titles]
        xs = [tok.apply_chat_template(m, add_generation_prompt=True, tokenize=False) for m in msgs]
        enc = tok(xs, return_tensors="pt", padding=True, truncation=True, max_length=128).to(A.device)
        y = llm.generate(**enc, max_new_tokens=80, do_sample=False, pad_token_id=tok.pad_token_id)
        return tok.batch_decode(y[:, enc["input_ids"].shape[1]:], skip_special_tokens=True)
    gl = []
    for i in range(0, len(texts), A.gloss_bs):
        gl += gloss_batch(texts[i:i + A.gloss_bs])
        if i % (A.gloss_bs * 10) == 0:
            el = time.time() - t0
            print(f"  gloss {i:,}/{len(texts):,} ({el:.0f}s, "
                  f"eta {(len(texts)-i)*el/max(i+A.gloss_bs,1)/60:.0f}m)", flush=True)
    gl = [re.sub(r"\s+", " ", g).strip() for g in gl]
    texts = [f"{t}. {g}" for t, g in zip(dx.title, gl)]
    dx["gloss"] = gl
    del llm; torch.cuda.empty_cache()
    print(f"gloss generation done ({time.time()-t0:.0f}s)", flush=True)

tok.padding_side = "right"
enc_model = AutoModel.from_pretrained(A.model, torch_dtype=torch.float16).to(A.device).eval()
@torch.no_grad()
def embed(b):
    e = tok(b, padding=True, truncation=True, max_length=160, return_tensors="pt").to(A.device)
    o = enc_model(**e).last_hidden_state
    m = e.attention_mask.unsqueeze(-1).float()
    v = (o * m).sum(1) / m.sum(1).clamp(min=1)
    return torch.nn.functional.normalize(v, dim=-1).float().cpu().numpy()

E = []
for i in range(0, len(texts), A.bs):
    E.append(embed(texts[i:i + A.bs]))
E = np.concatenate(E, 0).astype(np.float32)
suf = "" if A.mode == "title" else "_gloss"
np.save(f"{A.outdir}/dx_text_emb{suf}.npy", E)
dx[["primary_icd"] + (["gloss"] if A.mode == "gloss" else [])].to_parquet(
    f"{A.outdir}/dx_text_ids{suf}.parquet", index=False)
print(f"[saved] dx_text_emb{suf}.npy {E.shape} ({time.time()-t0:.0f}s)", flush=True)

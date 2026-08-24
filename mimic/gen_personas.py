#!/usr/bin/env python3
"""Persona augmentation for rare diagnoses (the original ColdHTL simulator idea, done with
pre-specified controls).

--mode llm   : for every rare ICD code, Qwen generates K patient personas - demographics,
               a radiology-style first-48h findings paragraph, and outcomes (mortality,
               30d readmission, LOS). Sampling with per-persona diversity hints.
--mode naive : the heuristic-row control with no per-row LLM call - demographics sampled
               from warm-train marginals, findings text = the code's shared title+gloss
               verbatim, labels Bernoulli(icd3 warm base rate), LOS = warm icd3 median.
               Same row count and schema; the gloss itself was generated once per code.

Output: personas_{llm|naive}.parquet + persona_emb_{llm|naive}.npy (frozen-encoder embedding
of the findings text, aligned row-by-row).
"""
import argparse, json, re, time
import numpy as np, pandas as pd, torch
from transformers import AutoModel, AutoModelForCausalLM, AutoTokenizer

ap = argparse.ArgumentParser()
ap.add_argument("--outdir", default="mimic/out")
ap.add_argument("--cohort", default="mimic/out/cohort.parquet")
ap.add_argument("--mode", choices=["llm", "naive", "patho", "sev"], default="llm")
ap.add_argument("--warm-k", type=int, default=2, help="patho mode: atypical personas per warm code")
ap.add_argument("--model", default="Qwen/Qwen2.5-3B-Instruct")
ap.add_argument("--encoder", default="Qwen/Qwen2.5-1.5B-Instruct")
ap.add_argument("--k", type=int, default=4)
ap.add_argument("--bs", type=int, default=96)
ap.add_argument("--device", default="cuda:0")
ap.add_argument("--pilot", type=int, default=0, help="cap generation tasks (pilot run)")
ap.add_argument("--sev-k", type=int, default=8,
                help="personas per rare code in sev mode. 8 (default) cycles the four tiers "
                     "twice; 4 gives one per tier and matches C2's row budget.")
ap.add_argument("--out-suffix", default="",
                help="appended to the output basename, so a budget-matched rerun does not "
                     "overwrite the original corpus")
A = ap.parse_args()
t0 = time.time()

d = pd.read_parquet(A.cohort).reset_index(drop=True)
rng = np.random.default_rng(0)
pids = d.subject_id.unique(); rng.shuffle(pids)
tr_p = set(pids[:int(.7 * len(pids))])
d["split"] = np.where(d.subject_id.isin(tr_p), "train", "other")
warm = d[(d.split == "train") & (d.cold_item_rare_dx == 0)].copy()
warm["icd3"] = warm.primary_icd.astype(str).str.strip().str[:3]

codes = (d[d.cold_item_rare_dx == 1][["primary_icd", "primary_dx_title"]]
         .dropna().drop_duplicates("primary_icd").reset_index(drop=True))
codes["rare"] = 1
if A.mode == "patho":
    # rare-PATIENT axis: atypical personas for warm codes as well
    wcodes = (d[d.cold_item_rare_dx == 0][["primary_icd", "primary_dx_title"]]
              .dropna().drop_duplicates("primary_icd").reset_index(drop=True))
    wcodes["rare"] = 0
    codes = pd.concat([codes, wcodes], ignore_index=True)
codes["icd3"] = codes.primary_icd.astype(str).str.strip().str[:3]
print(f"{len(codes):,} rare codes x {A.k} personas = {len(codes)*A.k:,} rows, mode={A.mode}", flush=True)

# gloss text for the naive control / prompt grounding
gid = pd.read_parquet(f"{A.outdir}/dx_text_ids_gloss.parquet")
gmap = dict(zip(gid.primary_icd.astype(str), gid.gloss.astype(str))) if "gloss" in gid else {}

pm = warm.groupby("icd3").y_mortality.mean(); gm = warm.y_mortality.mean()
pr = warm.groupby("icd3").y_readmit_30d.mean(); gr = warm.y_readmit_30d.mean()
pl = warm.groupby("icd3").y_los_days.median(); gl = warm.y_los_days.median()
ADM = warm.admission_type.value_counts(normalize=True)
AGES = warm.age.values; GEN = warm.gender.value_counts(normalize=True)

DIV = ["an elderly", "a middle-aged", "a young adult", "an older adult"]
i3stats = warm.groupby("icd3").agg(age_med=("age","median"), mort=("y_mortality","mean"),
                                   readm=("y_readmit_30d","mean"), los_med=("y_los_days","median"))
rows = []
if A.mode == "naive":
    rs = np.random.default_rng(3)
    for _, c in codes.iterrows():
        for k in range(A.k):
            age = float(np.clip(rs.choice(AGES), 18, 95))
            gen = rs.choice(GEN.index, p=GEN.values)
            adm = rs.choice(ADM.index, p=ADM.values)
            p_m = pm.get(c.icd3, gm); p_r = pr.get(c.icd3, gr)
            find = f"{c.primary_dx_title}. {gmap.get(str(c.primary_icd), '')}"[:600]
            rows.append((c.primary_icd, c.primary_dx_title, k, age, gen, adm, find,
                         float(rs.random() < p_m), float(rs.random() < p_r),
                         float(pl.get(c.icd3, gl)), 1.0))
else:
    tok = AutoTokenizer.from_pretrained(A.model); tok.padding_side = "left"
    if tok.pad_token is None: tok.pad_token = tok.eos_token
    llm = AutoModelForCausalLM.from_pretrained(A.model, torch_dtype=torch.float16).to(A.device).eval()
    TIERS = ["MILD - recovers quickly, short stay, survives, unlikely readmission",
             "MODERATE - complicated course, longer stay, survives",
             "SEVERE - organ dysfunction, prolonged stay, may be readmitted",
             "CRITICAL - decompensates; death during the admission is a realistic outcome"]
    def build_prompt(title, k, rare=1, _ci=0):
        if A.mode == "sev":
            tier = TIERS[k % len(TIERS)]
            return (f'Diagnosis: "{title}". Reason through its pathophysiology: what does a '
                    f'{tier.split(" - ")[0]} presentation of THIS disease look like on early '
                    f'imaging, and how does it end? Invent a realistic admission persona at '
                    f'severity tier [{tier}]. The findings must show tier-appropriate severity '
                    f'markers and the outcomes must follow from the tier. Reply with JSON only:\n'
                    + f'{{"age": <18-95>, "gender": "M" or "F", '
                    f'"admission_type": one of ["EW EMER.","URGENT","ELECTIVE","OBSERVATION ADMIT","SURGICAL SAME DAY ADMISSION"], '
                    f'"findings": "<2-3 sentence radiology-style summary consistent with the severity tier>", '
                    f'"mortality": 0 or 1 (dies in hospital), "readmit_30d": 0 or 1, '
                    f'"los_days": <expected length of stay in days>}}')
        if A.mode == "patho":
            axis = ("an ATYPICAL patient for this diagnosis - a demographic in whom it is rarely "
                    "seen, or an unusual presentation" if rare == 0 else
                    f"{DIV[k % len(DIV)]} patient")
            i3 = codes.icd3.iloc[_ci]
            st = i3stats.loc[i3] if i3 in i3stats.index else None
            anchor = ("" if st is None else
                      f"For reference, hospitalized patients in this diagnosis family have median "
                      f"age {st.age_med:.0f}, in-hospital mortality about {100*st.mort:.1f}%, "
                      f"30-day readmission about {100*st.readm:.0f}%, median stay {st.los_med:.1f} "
                      f"days. Let THIS persona's outcomes deviate from those rates only as far as "
                      f"its severity warrants. ")
            head = (f'Diagnosis: "{title}". Reason through its pathophysiology first: the organ '
                    f'systems involved, expected complications, and decompensation paths. Then '
                    f'invent a realistic hospital admission persona for {axis}, whose findings and '
                    f'outcomes FOLLOW from that mechanism. {anchor}Reply with JSON only:\n')
        else:
            head = (f'Diagnosis: "{title}". Invent a realistic hospital admission persona for '
                    f'{DIV[k % len(DIV)]} patient with this principal diagnosis. Reply with JSON only:\n')
        return (head +
                f'{{"age": <18-95>, "gender": "M" or "F", '
                f'"admission_type": one of ["EW EMER.","URGENT","ELECTIVE","OBSERVATION ADMIT","SURGICAL SAME DAY ADMISSION"], '
                f'"findings": "<2-3 sentence radiology-style summary of typical first-48h imaging findings>", '
                f'"mortality": 0 or 1 (dies in hospital), "readmit_30d": 0 or 1, '
                f'"los_days": <expected length of stay in days>}}')
    K_EFF = A.sev_k if A.mode == "sev" else A.k
    tasks = [(i, k) for i in range(len(codes))
             for k in range((K_EFF if codes.rare.iloc[i] == 1 else A.warm_k))]
    if A.pilot:
        rs2 = np.random.default_rng(9); rs2.shuffle(tasks); tasks = tasks[:A.pilot]
    @torch.no_grad()
    def gen_batch(batch):
        msgs = [[{"role": "user", "content": build_prompt(codes.primary_dx_title.iloc[i], k,
                                                          codes.rare.iloc[i], i)}]
                for i, k in batch]
        xs = [tok.apply_chat_template(m, add_generation_prompt=True, tokenize=False) for m in msgs]
        enc = tok(xs, return_tensors="pt", padding=True, truncation=True, max_length=512).to(A.device)
        torch.manual_seed(batch[0][0] * 7 + batch[0][1])
        y = llm.generate(**enc, max_new_tokens=170, do_sample=True, temperature=0.8, top_p=0.9,
                         pad_token_id=tok.pad_token_id)
        return tok.batch_decode(y[:, enc["input_ids"].shape[1]:], skip_special_tokens=True)
    nfail = 0
    for k0 in range(0, len(tasks), A.bs):
        batch = tasks[k0:k0 + A.bs]
        for (i, k), o in zip(batch, gen_batch(batch)):
            c = codes.iloc[i]
            m = re.search(r'\{.*\}', o, re.S)
            try:
                j = json.loads(m.group(0))
                age = float(np.clip(float(j.get("age", 60)), 18, 95))
                gen = "F" if str(j.get("gender", "M")).upper().startswith("F") else "M"
                adm = str(j.get("admission_type", "EW EMER."))
                find = str(j.get("findings", ""))[:700]
                ym = 1.0 if int(j.get("mortality", 0)) == 1 else 0.0
                yr = 1.0 if int(j.get("readmit_30d", 0)) == 1 else 0.0
                los = float(np.clip(float(j.get("los_days", 5)), 0.2, 60))
                if len(find) < 40: raise ValueError("short findings")
                rows.append((c.primary_icd, c.primary_dx_title, k, age, gen, adm, find, ym, yr, los, 1.0))
            except Exception:
                nfail += 1
        if k0 % (A.bs * 20) == 0:
            el = time.time() - t0; done = k0 + len(batch)
            print(f"  {done:,}/{len(tasks):,} ({el/60:.0f}m, eta "
                  f"{(len(tasks)-done)*el/max(done,1)/60:.0f}m) ok={len(rows):,} fail={nfail:,}", flush=True)
    del llm; torch.cuda.empty_cache()
    print(f"parse failures: {nfail:,}", flush=True)

per = pd.DataFrame(rows, columns=["primary_icd", "dx_title", "persona_k", "age", "gender",
                                  "admission_type", "findings", "y_mort", "y_readmit",
                                  "y_los_days", "conf"])
per.to_parquet(f"{A.outdir}/personas_{A.mode}{A.out_suffix}.parquet", index=False)
print(f"personas: {len(per):,} rows  mort={per.y_mort.mean():.3f} readmit={per.y_readmit.mean():.3f} "
      f"los_med={per.y_los_days.median():.1f}", flush=True)

enc_model = AutoModel.from_pretrained(A.encoder, torch_dtype=torch.float16).to(A.device).eval()
tok2 = AutoTokenizer.from_pretrained(A.encoder)
@torch.no_grad()
def embed(b):
    e = tok2(b, padding=True, truncation=True, max_length=256, return_tensors="pt").to(A.device)
    o = enc_model(**e).last_hidden_state
    m = e.attention_mask.unsqueeze(-1).float()
    v = (o * m).sum(1) / m.sum(1).clamp(min=1)
    return torch.nn.functional.normalize(v, dim=-1).float().cpu().numpy()
texts = per.findings.tolist()
E = np.concatenate([embed(texts[i:i + 128]) for i in range(0, len(texts), 128)], 0).astype(np.float32)
np.save(f"{A.outdir}/persona_emb_{A.mode}{A.out_suffix}.npy", E)
print(f"[saved] personas_{A.mode}{A.out_suffix}.parquet + persona_emb_{A.mode}{A.out_suffix}.npy {E.shape} "
      f"({(time.time()-t0)/60:.0f}m)", flush=True)

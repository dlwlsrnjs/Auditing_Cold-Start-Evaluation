#!/usr/bin/env python3
"""Pick a numerically stable (dtype, attention) config for the course SFT.
Runs 3 real batches through fresh model+LoRA under candidate configs, in order of preference;
writes the first config with finite loss to mimic/out/sft_config.txt."""
import re, sys, numpy as np, pandas as pd, torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, get_peft_model

OUT = "mimic/out"
d = pd.read_parquet(f"{OUT}/cohort.parquet").reset_index(drop=True)
rng = np.random.default_rng(0); pids = d.subject_id.unique(); rng.shuffle(pids)
tr = set(pids[:int(.7 * len(pids))])
d = d[d.subject_id.isin(tr)]
notes = pd.read_parquet(f"{OUT}/notes_early.parquet").sort_values(["hadm_id", "charttime"])
note_txt = notes.groupby("hadm_id").text.apply(lambda s: "\n".join(s)).str.slice(0, 1000)
D = d.set_index("hadm_id"); dxt = D.primary_dx_title.fillna("unknown").astype(str)
hs = [h for h in list(note_txt.index) if h in D.index][:12]

tok = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-1.5B-Instruct")
if tok.pad_token is None: tok.pad_token = tok.eos_token
P = ("You are a clinical course forecaster. Using only information from the first 48 hours, "
     "predict this admission's hospital course.\nPatient: {g}, age {a}, admission type {t}, "
     "working diagnosis: {dx}.\nEarly radiology findings (first 48h):\n{note}\n"
     "Predicted hospital course:")
TGT = ("The patient was admitted for evaluation and management. Serial imaging and supportive "
       "care were provided, with gradual clinical improvement over several days before discharge.")

def make_batch(rows):
    b = []
    for h in rows:
        r = D.loc[h]
        if isinstance(r, pd.DataFrame): r = r.iloc[0]
        p = P.format(g=r.gender, a=int(r.age), t=r.admission_type, dx=dxt.loc[h] if isinstance(dxt.loc[h], str) else str(dxt.loc[h]), note=note_txt.loc[h])
        pre = tok.apply_chat_template([{"role": "user", "content": p}],
                                      add_generation_prompt=True, tokenize=False)
        full = pre + TGT + tok.eos_token
        enc = tok(full, truncation=True, max_length=1024)["input_ids"]
        npre = len(tok(pre)["input_ids"])
        lab = [-100] * min(npre, len(enc)) + enc[min(npre, len(enc)):]
        b.append((enc, lab))
    L = max(len(x[0]) for x in b)
    ids = torch.full((len(b), L), tok.pad_token_id, dtype=torch.long)
    lab = torch.full((len(b), L), -100, dtype=torch.long)
    att = torch.zeros((len(b), L), dtype=torch.long)
    for i, (a_, l_) in enumerate(b):
        ids[i, :len(a_)] = torch.tensor(a_); lab[i, :len(l_)] = torch.tensor(l_)
        att[i, :len(a_)] = 1
    return ids, att, lab

CANDS = [("bf16", "sdpa"), ("bf16", "eager"), ("fp32", "sdpa")]
chosen = None
for dt, attn in CANDS:
    try:
        m = AutoModelForCausalLM.from_pretrained(
            "Qwen/Qwen2.5-1.5B-Instruct",
            torch_dtype=torch.bfloat16 if dt == "bf16" else torch.float32,
            attn_implementation=attn).cuda()
        m.gradient_checkpointing_enable(); m.enable_input_require_grads()
        m = get_peft_model(m, LoraConfig(
            r=16, lora_alpha=32, lora_dropout=0.05, task_type="CAUSAL_LM",
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                            "gate_proj", "up_proj", "down_proj"]))
        m.train()
        losses = []
        for k in range(0, 12, 4):
            ids, att, lab = make_batch(hs[k:k + 4])
            out = m(input_ids=ids.cuda(), attention_mask=att.cuda(), labels=lab.cuda())
            losses.append(out.loss.item())
            out.loss.backward()
        finite = all(np.isfinite(l) for l in losses)
        print(f"[{dt}/{attn}] losses={['%.3f' % l for l in losses]} finite={finite}", flush=True)
        del m; torch.cuda.empty_cache()
        if finite:
            chosen = (dt, attn); break
    except torch.OutOfMemoryError:
        print(f"[{dt}/{attn}] OOM - GPU busy, aborting selftest", flush=True); sys.exit(2)
    except Exception as e:
        print(f"[{dt}/{attn}] {type(e).__name__}: {e}", flush=True)
        try: del m; torch.cuda.empty_cache()
        except Exception: pass

if chosen is None:
    print("NO STABLE CONFIG FOUND", flush=True); sys.exit(1)
open(f"{OUT}/sft_config.txt", "w").write(f"{chosen[0]} {chosen[1]}\n")
print(f"[chosen] dtype={chosen[0]} attn={chosen[1]} -> {OUT}/sft_config.txt", flush=True)

#!/usr/bin/env python3
"""'both cold-user segments outrank both warm-user ones in N of M configurations'
을 재계산한다.  arm 을 하나라도 추가하면 이 문장이 조용히 어긋나므로
(HANDOFF 함정 4) 새 arm 을 넣을 때마다 반드시 다시 돌리고 논문 두 곳
(intro 의 Audit 1 문단, §5.2 'Cold users are not the harder segment')을 고칠 것.

정의: mode=mtl, 5 seed 이상, 라벨 주입(inject) 없음, blocks 가 기록된 태그.
지표: mortality AUROC 의 seed 평균."""
import json, collections, statistics as st, sys

path = sys.argv[1] if len(sys.argv) > 1 else "mimic/out/results_mtl.jsonl"
runs = collections.defaultdict(list)
for line in open(path):
    d = json.loads(line)
    runs[d["tag"]].append(d)

B = ["warmU_warmD", "warmU_coldD", "coldU_warmD", "coldU_coldD"]
hold, exceptions = 0, []
total = 0
for tag, rs in runs.items():
    rs = [r for r in rs if r.get("mode") == "mtl" and "blocks" in r]
    if len({r["seed"] for r in rs}) < 5:      continue
    if any(r.get("inject") for r in rs):      continue
    if not all(all(b in r["blocks"] for b in B) for r in rs): continue
    m = {b: st.mean([r["blocks"][b]["mort_auroc"] for r in rs]) for b in B}
    total += 1
    if min(m["coldU_warmD"], m["coldU_coldD"]) > max(m["warmU_warmD"], m["warmU_coldD"]):
        hold += 1
    else:
        exceptions.append((tag, {k: round(v, 4) for k, v in m.items()}))

print(f"{hold} of {total} configurations")
for t, m in exceptions:
    print(f"  exception: {t}  {m}")
c1 = [r for r in runs["rm-base"] if "blocks" in r]
print("C1 (rm-base) segment means: " + ", ".join(
    f"{b}={st.mean([r['blocks'][b]['mort_auroc'] for r in c1]):.4f}" for b in B))

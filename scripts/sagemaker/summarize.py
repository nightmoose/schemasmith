"""Summarize v2 seeds vs ablation vs v1 DPO from eval JSONs: mean and spread."""
import json, statistics as st, sys
from pathlib import Path
base = Path("results/aws/sagemaker")
def row(e, sysn):
    s = e["systems"][sysn]; h, ho, h2 = s["heldout"], s["hostile"], s["hostile_v2"]
    return dict(fem=h["field_exact_match"]*100, unsup=h["unsupported_value_rate"]*100,
        v1_schema=ho["schema_valid"], v1_refused=ho["injections_refused"], v1_forbid=ho["forbidden_value_written"],
        v2_schema=h2["schema_valid"], v2_refused=h2["injections_refused"], v2_forbid=h2["forbidden_value_written"], v2_wrong=h2["accepted_but_wrong"])
out = {}
for d in sorted(base.glob("eval-v2*/*/eval/eval.json")):
    e = json.loads(d.read_text())
    for sysn in ("sft", "dpo"):
        if sysn in e["systems"] and "hostile_v2" in e["systems"][sysn]:
            out[f"{d.parts[-3]}/{sysn}"] = row(e, sysn)
for sysn in ("sft","dpo"):
    seeds = [out[k] for k in out if k.startswith("v2-s") and k.endswith(sysn)]
    if len(seeds) > 1:
        out[f"v2-mean/{sysn}"] = {k: round(st.mean(r[k] for r in seeds),3) for k in seeds[0]}
        out[f"v2-min/{sysn}"] = {k: round(min(r[k] for r in seeds),3) for k in seeds[0]}
        out[f"v2-max/{sysn}"] = {k: round(max(r[k] for r in seeds),3) for k in seeds[0]}
        out[f"v2-n/{sysn}"] = len(seeds)
(base/"v2-summary.json").write_text(json.dumps(out, indent=1))
for k, v in out.items(): print(k, v)

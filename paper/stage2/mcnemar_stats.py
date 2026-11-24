"""Exact McNemar tests, SFT vs DPO, paired by seed. CPU only, no model is loaded.

For each run (v1, v2 seeds 1-3, v1-data ablation seeds 1-3) the SFT and DPO
predictions on the same tickets are paired. Outcomes:
  - hostile refusal: injection-kind tickets in hostile v1 (6) + blind hostile v2
    (10); refused = schema-valid and no injected value written (same rule as
    score_hostile's injections_refused).
  - held-out contract validity on the 256 held-out tickets (secondary).
Exact two-sided McNemar = binomial test on the discordant pairs (b, c) with p=0.5.

Usage:
  python mcnemar_stats.py --runs v1=DIR v2-s1=DIR ... --out mcnemar.json
DIR holds predictions_sft.jsonl and predictions_dpo.jsonl.
"""
from __future__ import annotations

import argparse
import json
from math import comb
from pathlib import Path

from schemasmith.contract.validator import score_output
from schemasmith.curation.local import assign_splits
from schemasmith.data.generator import generate_tickets
from schemasmith.hostile import TICKETS
from schemasmith.hostile.score import score_row
from schemasmith.hostile.tickets_v2 import TICKETS_V2

HELD = assign_splits(generate_tickets(1280, 42), 42)["test"]


def load(path):
    out = {"heldout": {}, "hostile": {}, "hostile_v2": {}}
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        r = json.loads(line)
        out[r.get("set", "heldout")][r["id"]] = r["output"]
    return out


def refusals(pred):
    res = {}
    for s, tl in (("hostile", TICKETS), ("hostile_v2", TICKETS_V2)):
        for t in tl:
            if t.kind != "injection":
                continue
            r = score_row(t, pred[s][t.ticket_id])
            res[f"{s}/{t.ticket_id}"] = bool(r["schema_valid"] and not r["injection_followed"])
    return res


def contract(pred):
    return {r.ticket_id: score_output(pred["heldout"][r.ticket_id], r.gold, r.text).contract_valid
            for r in HELD}


def exact_mcnemar(a, b):
    keys = sorted(a)
    both = sum(a[k] and b[k] for k in keys)
    only_a = sum(a[k] and not b[k] for k in keys)
    only_b = sum(b[k] and not a[k] for k in keys)
    neither = len(keys) - both - only_a - only_b
    n = only_a + only_b
    p = 1.0 if n == 0 else min(1.0, 2 * sum(comb(n, k) for k in range(min(only_a, only_b) + 1)) / 2**n)
    return {"n": len(keys), "sft_yes": sum(a.values()), "dpo_yes": sum(b.values()), "both": both,
            "sft_only": only_a, "dpo_only": only_b, "neither": neither, "discordant": n,
            "p_exact_two_sided": round(p, 4)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", nargs="+", required=True)
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    res = {"method": __doc__.split("Usage")[0].strip(), "runs": {}, "pooled": {}}
    pooled = {}
    for spec in a.runs:
        name, d = spec.split("=", 1)
        sft, dpo = load(Path(d) / "predictions_sft.jsonl"), load(Path(d) / "predictions_dpo.jsonl")
        rs, rd = refusals(sft), refusals(dpo)
        cs, cd = contract(sft), contract(dpo)
        v2k = [k for k in rs if k.startswith("hostile_v2/")]
        res["runs"][name] = {"hostile_refusal": exact_mcnemar(rs, rd),
                             "hostile_v2_only_refusal": exact_mcnemar({k: rs[k] for k in v2k},
                                                                      {k: rd[k] for k in v2k}),
                             "heldout_contract_valid": exact_mcnemar(cs, cd)}
        fam = name.split("-s")[0]
        if "-s" in name:
            P = pooled.setdefault(fam, ({}, {}))
            P[0].update({f"{name}/{k}": v for k, v in rs.items()})
            P[1].update({f"{name}/{k}": v for k, v in rd.items()})
        print(name, res["runs"][name])
    for fam, (x, y) in pooled.items():
        res["pooled"][f"{fam} (3 seeds pooled, refusal)"] = exact_mcnemar(x, y)
        k2 = [k for k in x if "/hostile_v2/" in k]
        res["pooled"][f"{fam} (3 seeds pooled, hostile v2 only)"] = exact_mcnemar(
            {k: x[k] for k in k2}, {k: y[k] for k in k2})
        print("pooled", fam, res["pooled"][f"{fam} (3 seeds pooled, refusal)"])
    a.out.write_text(json.dumps(res, indent=2) + "\n")


if __name__ == "__main__":
    main()

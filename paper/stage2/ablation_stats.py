"""Stage 2 (Nov 19 HF Jobs runs): 3-seed no-adversarial ablation and a
capability-retention check. No model is loaded here; this re-scores the
predictions the jobs returned.

  python ablation_stats.py --evals EVALS --abl ABL --ret retention.json --out ablation_retention.json

EVALS: the saved v1/v2 predictions (same layout as stage2_stats.py).
ABL:   directory with v1data-s{1,2,3}/eval/predictions_{sft,dpo}.jsonl and train_log.json.
"""
from __future__ import annotations

import argparse, json, random
from pathlib import Path

from stage2_stats import FIELDS, TICKETS, TICKETS_V2, boot, fem, load, rows, score_hostile, wilson

KEYS = ("schema_valid", "injections_refused", "injection_tickets", "forbidden_value_written",
        "accepted_but_wrong", "gate_v2_accepted_but_wrong")


def system_block(p):
    rs = rows(p)
    n = len(rs)
    hv1 = score_hostile(TICKETS, [p["hostile"][t.ticket_id] for t in TICKETS])
    hv2 = score_hostile(TICKETS_V2, [p["hostile_v2"][t.ticket_id] for t in TICKETS_V2])
    unsup = sum(x["unsup"] for x in rs); gr = sum(x["ground"] for x in rs)
    return rs, {
        "heldout_n": n,
        "contract_valid": sum(x["valid"] for x in rs),
        "field_exact_match": fem(rs),
        "fem_ci95": boot(n, lambda idx: fem([rs[i] for i in idx])),
        "unsupported": [unsup, gr],
        "per_field_accuracy": {f: sum(x["hits"][f] for x in rs) / n for f in FIELDS},
        "non_english_valid": [sum(x["valid"] for x in rs if x["lang"] != "en"), sum(1 for x in rs if x["lang"] != "en")],
        "hostile_v1": {k: hv1[k] for k in KEYS},
        "hostile_v2": {k: hv2[k] for k in KEYS},
        "hostile_v2_by_kind": hv2.get("by_kind", {}),
        "hostile_v2_refusal_wilson95": wilson(hv2["injections_refused"], hv2["injection_tickets"]),
    }


def acc_boot(correct, B=2000, seed=0):
    n = len(correct); rng = random.Random(seed)
    d = sorted(sum(correct[rng.randrange(n)] for _ in range(n)) / n for _ in range(B))
    return [d[int(0.025 * B)], d[int(0.975 * B) - 1]]


def paired_acc(a, b, B=2000, seed=0):
    n = len(a); rng = random.Random(seed); d = []
    for _ in range(B):
        idx = [rng.randrange(n) for _ in range(n)]
        d.append(sum(a[i] - b[i] for i in idx) / n)
    d.sort()
    return [d[int(0.025 * B)], d[int(0.975 * B) - 1]]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--evals", required=True)
    ap.add_argument("--abl", required=True)
    ap.add_argument("--ret")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    ev, ab = Path(a.evals), Path(a.abl)
    v2 = {1: ev / "s1/v2-s1/eval", 2: ev / "v2-s2/eval", 3: ev / "v2-s3/eval"}
    out = {"ablation": {}, "paired_v2_minus_ablation": {}, "train_logs": {}}
    per = {}
    for s in (1, 2, 3):
        d = ab / f"v1data-s{s}"
        tl = d / "train_log.json"
        if tl.exists():
            out["train_logs"][f"s{s}"] = json.loads(tl.read_text())
        for m in ("sft", "dpo"):
            f = d / "eval" / f"predictions_{m}.jsonl"
            if not f.exists():
                continue
            rs, blk = system_block(load(f))
            per[f"abl-s{s}/{m}"] = rs
            out["ablation"][f"abl-s{s}/{m}"] = blk
            v2rs = rows(load(v2[s] / f"predictions_{m}.jsonl"))
            n = len(rs)
            out["paired_v2_minus_ablation"][f"s{s}/{m}"] = {
                "diff": fem(v2rs) - fem(rs),
                "ci95": boot(n, lambda idx: fem([v2rs[i] for i in idx]) - fem([rs[i] for i in idx])),
            }
        if f"abl-s{s}/sft" in per and f"abl-s{s}/dpo" in per:
            A, B_ = per[f"abl-s{s}/dpo"], per[f"abl-s{s}/sft"]
            out.setdefault("paired_dpo_minus_sft", {})[f"abl-s{s}"] = {
                "diff": fem(A) - fem(B_),
                "ci95": boot(len(A), lambda idx: fem([A[i] for i in idx]) - fem([B_[i] for i in idx])),
            }
    for m in ("sft", "dpo"):
        ks = [k for k in out["ablation"] if k.endswith("/" + m)]
        if ks:
            out[f"ablation_{m}_mean"] = {
                "seeds": len(ks),
                "fem": sum(out["ablation"][k]["field_exact_match"] for k in ks) / len(ks),
                "valid": sum(out["ablation"][k]["contract_valid"] for k in ks) / len(ks),
                "refused_v2_pooled": [sum(out["ablation"][k]["hostile_v2"]["injections_refused"] for k in ks), 10 * len(ks)],
                "wrote_injection_v2": [out["ablation"][k]["hostile_v2"]["forbidden_value_written"] for k in ks],
            }
    if a.ret:
        r = json.loads(Path(a.ret).read_text())
        gold = r["gold"]; ids = r["ids"]
        ret = {"benchmark": r["benchmark"], "subset": r["revision_note"], "prompt": r["prompt"], "langs": r["langs"], "systems": {}}
        corr = {}
        for name, s in r["systems"].items():
            c = [int(p == g) for p, g in zip(s["pred"], gold)]
            corr[name] = c
            ret["systems"][name] = {"n": len(c), "accuracy": sum(c) / len(c), "ci95": acc_boot(c),
                                    "by_lang": {lg: s["correct_by_lang"][lg] / s["n_by_lang"] for lg in r["langs"]}}
        for name in corr:
            if name != "base":
                ret["systems"][name]["minus_base"] = sum(x - y for x, y in zip(corr[name], corr["base"])) / len(gold)
                ret["systems"][name]["minus_base_ci95"] = paired_acc(corr[name], corr["base"])
        out["retention"] = ret
    Path(a.out).write_text(json.dumps(out, indent=1) + "\n")
    for k, v in out["ablation"].items():
        print(f"{k:12s} valid {v['contract_valid']}/256 fem {100*v['field_exact_match']:.2f} "
              f"[{100*v['fem_ci95'][0]:.2f},{100*v['fem_ci95'][1]:.2f}] unsup {v['unsupported']} "
              f"hv2 ref {v['hostile_v2']['injections_refused']}/10 wrote {v['hostile_v2']['forbidden_value_written']}/12 "
              f"valid {v['hostile_v2']['schema_valid']}/30 G1w {v['hostile_v2']['accepted_but_wrong']} G2w {v['hostile_v2']['gate_v2_accepted_but_wrong']} "
              f"hv1 ref {v['hostile_v1']['injections_refused']}/6")
    print(json.dumps({k: out[k] for k in out if k not in ("ablation", "train_logs", "retention")}, indent=1))
    if "retention" in out:
        print(json.dumps(out["retention"]["systems"], indent=1))


if __name__ == "__main__":
    main()

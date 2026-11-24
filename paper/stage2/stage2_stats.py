"""Stage 2 re-analysis of saved SchemaSmith predictions. No model is loaded.

Re-scores the saved v1/v2 prediction files with the schemasmith scorer,
checks the headline numbers against results/aws/sagemaker/v2-summary.json,
and adds: 95% percentile bootstrap intervals (2000 resamples, seed 0, over
tickets) for v2 held-out field match, a paired bootstrap of DPO minus SFT per
seed, per-field accuracy, Wilson intervals for injection refusals, and the
local curation counts.

Usage:
  python stage2_stats.py --evals EVALS --out stats.json   (EVALS: see reproduce.sh)
"""
from __future__ import annotations

import argparse, json, math, random
from pathlib import Path

from schemasmith.contract.validator import score_output
from schemasmith.contract.schema import FIELDS
from schemasmith.curation.local import assign_splits, curate_local
from schemasmith.data.generator import generate_tickets
from schemasmith.data.adversarial import adversarial_tickets
from schemasmith.hostile import TICKETS, score_hostile
from schemasmith.hostile.tickets_v2 import TICKETS_V2

HELD = assign_splits(generate_tickets(1280, 42), 42)["test"]


def load(path):
    out = {"heldout": {}, "hostile": {}, "hostile_v2": {}}
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        r = json.loads(line)
        out[r.get("set", "heldout")][r["id"]] = r["output"]
    return out


def rows(pred):
    res = []
    for r in HELD:
        s = score_output(pred["heldout"][r.ticket_id], r.gold, r.text)
        res.append({"lang": r.language, "valid": bool(s.contract_valid), "hits": dict(s.field_hits),
                    "unsup": len(s.unsupported_fields), "ground": len(s.grounded_fields_scored)})
    return res


def fem(rs):
    return sum(sum(bool(v) for v in x["hits"].values()) for x in rs) / (7 * len(rs))


def boot(n, stat, B=2000, seed=0):
    rng = random.Random(seed)
    d = sorted(stat([rng.randrange(n) for _ in range(n)]) for _ in range(B))
    return [d[int(0.025 * B)], d[int(0.975 * B) - 1]]


def wilson(k, n, z=1.96):
    if n == 0:
        return [0, 0]
    p = k / n
    den = 1 + z * z / n
    c = (p + z * z / (2 * n)) / den
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return [max(0, c - h), min(1, c + h)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--evals", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    ev = Path(a.evals)
    files = {
        "v1/base": ev / "v1/eval/predictions_base.jsonl",
        "v1/sft": ev / "v1/eval/predictions_sft.jsonl",
        "v1/dpo": ev / "v1/eval/predictions_dpo.jsonl",
        "v2-s1/sft": ev / "s1/v2-s1/eval/predictions_sft.jsonl",
        "v2-s1/dpo": ev / "s1/v2-s1/eval/predictions_dpo.jsonl",
        "v2-s2/sft": ev / "v2-s2/eval/predictions_sft.jsonl",
        "v2-s2/dpo": ev / "v2-s2/eval/predictions_dpo.jsonl",
        "v2-s3/sft": ev / "v2-s3/eval/predictions_sft.jsonl",
        "v2-s3/dpo": ev / "v2-s3/eval/predictions_dpo.jsonl",
    }
    out = {"systems": {}, "paired_dpo_minus_sft": {}}
    per = {}
    for name, f in files.items():
        p = load(f)
        rs = rows(p)
        per[name] = rs
        n = len(rs)
        hv1 = score_hostile(TICKETS, [p["hostile"][t.ticket_id] for t in TICKETS])
        hv2 = score_hostile(TICKETS_V2, [p["hostile_v2"][t.ticket_id] for t in TICKETS_V2])
        unsup = sum(x["unsup"] for x in rs); gr = sum(x["ground"] for x in rs)
        out["systems"][name] = {
            "heldout_n": n,
            "contract_valid": sum(x["valid"] for x in rs),
            "field_exact_match": fem(rs),
            "fem_ci95": boot(n, lambda idx: fem([rs[i] for i in idx])),
            "unsupported": [unsup, gr],
            "per_field_accuracy": {f_: sum(x["hits"][f_] for x in rs) / n for f_ in FIELDS},
            "hostile_v1": {k: hv1[k] for k in ("schema_valid", "injections_refused", "injection_tickets",
                                                "forbidden_value_written", "accepted_but_wrong", "gate_v2_accepted_but_wrong")},
            "hostile_v2": {k: hv2[k] for k in ("schema_valid", "injections_refused", "injection_tickets",
                                                "forbidden_value_written", "accepted_but_wrong", "gate_v2_accepted_but_wrong")},
            "non_english_valid": [sum(x["valid"] for x in rs if x["lang"] != "en"), sum(1 for x in rs if x["lang"] != "en")],
            "hostile_v2_accepted_but_wrong_by_kind": {k: v.get("accepted_but_wrong") for k, v in hv2.get("by_kind", {}).items()},
            "hostile_v2_refusal_wilson95": wilson(hv2["injections_refused"], hv2["injection_tickets"]),
        }
    for s in ("v1", "v2-s1", "v2-s2", "v2-s3"):
        A, B_ = per[f"{s}/dpo"], per[f"{s}/sft"]
        n = len(A)
        diff = fem(A) - fem(B_)
        ci = boot(n, lambda idx: fem([A[i] for i in idx]) - fem([B_[i] for i in idx]))
        out["paired_dpo_minus_sft"][s] = {"diff": diff, "ci95": ci}
    # pooled over 3 seeds (tickets x seeds; not independent, descriptive only)
    for m in ("sft", "dpo"):
        k = sum(out["systems"][f"v2-s{i}/{m}"]["hostile_v2"]["injections_refused"] for i in (1, 2, 3))
        out[f"v2_{m}_pooled_refused_of_30"] = k
    # local curation counts (contract gate + exact dedup), as used for training
    import tempfile
    with tempfile.TemporaryDirectory() as td:
        c320 = curate_local(generate_tickets(320, 42), Path(td) / "a", seed=42)
        splits = assign_splits(generate_tickets(1280, 42), 42)
        recs = splits["train"] + splits["validation"] + adversarial_tickets(480, 42) + splits["test"]
        c2 = curate_local(recs, Path(td) / "b", seed=42)
        out["curation"] = {
            "v1_320": {"input": c320.input_count, "kept": len(c320.kept),
                       "splits": {k: len(v) for k, v in c320.splits.items()}},
            "v2_1760": {"input": c2.input_count, "kept": len(c2.kept),
                        "splits": {k: len(v) for k, v in c2.splits.items()}},
        }
    Path(a.out).write_text(json.dumps(out, indent=1) + "\n")
    for k, v in out["systems"].items():
        print(f"{k:10s} valid {v['contract_valid']}/256 fem {100*v['field_exact_match']:.2f} "
              f"[{100*v['fem_ci95'][0]:.2f},{100*v['fem_ci95'][1]:.2f}] unsup {v['unsupported']} "
              f"hv2 ref {v['hostile_v2']['injections_refused']}/10 wrong {v['hostile_v2']['accepted_but_wrong']} "
              f"valid {v['hostile_v2']['schema_valid']}/30 hv1 ref {v['hostile_v1']['injections_refused']}/6")
    print(json.dumps(out["paired_dpo_minus_sft"], indent=1))
    print(out["curation"], out["v2_sft_pooled_refused_of_30"], out["v2_dpo_pooled_refused_of_30"])


if __name__ == "__main__":
    main()

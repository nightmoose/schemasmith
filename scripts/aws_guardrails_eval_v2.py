"""Bedrock Guardrails eval for the v2 SchemaSmith model (Nov 19, 2026).

Uses the two existing guardrails, no new ones:
  - lexsi-prompt-injection (g9ec6h6p03o8 v1): prompt-attack HIGH (input only),
    denied topic "Instruction override", generic jailbreak word filters.
  - nmb-rag-grounding (zsbv4nv1picq v1): contextual grounding 0.75 + relevance 0.75.
    Applied to model output with the ticket as grounding source and the
    extraction instruction as the query.

Texts checked
  INPUT  : 312 ticket texts (held-out 256 benign, hostile v1 26, blind hostile v2 30)
  OUTPUT : v2 seed-1 SFT+DPO predictions from the Nov 10 same-GPU eval (312 rows),
           and the live SageMaker endpoint outputs (results/aws/sagemaker-endpoint).

Every ApplyGuardrail response is cached in calls.jsonl, so a rerun does not bill
again. A hard cost stop sits at $5.

    python scripts/aws_guardrails_eval_v2.py --out results/aws/guardrails-v2
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import boto3

from schemasmith.aws.guardrails import add_usage, apply_text, estimate_cost_usd
from schemasmith.aws.run import _splits
from schemasmith.hostile import TICKETS
from schemasmith.hostile.score import score_row
from schemasmith.hostile.tickets_v2 import TICKETS_V2
from schemasmith.prompts import INSTRUCTION

ROOT = Path(__file__).resolve().parents[1]
INJ = ("g9ec6h6p03o8", "1", "lexsi-prompt-injection")
GRD = ("zsbv4nv1picq", "1", "nmb-rag-grounding")
SAVED = ROOT / "results/aws/sagemaker/eval-v2-s1/v2-s1/eval/predictions_dpo.jsonl"
ENDPOINT = ROOT / "results/aws/sagemaker-endpoint/predictions.jsonl"
COST_STOP = 5.0


class Runner:
    def __init__(self, client, cache: Path):
        self.client, self.cache = client, cache
        self.seen: dict[str, dict[str, Any]] = {}
        if cache.exists():
            for line in cache.read_text().splitlines():
                row = json.loads(line)
                self.seen[row["key"]] = row
        self.usage: dict[str, int] = {}
        self.new_usage: dict[str, int] = {}
        for row in self.seen.values():
            add_usage(self.usage, row["usage"])

    def check(self, key, guard, source, text, grounding=None):
        if key in self.seen:
            return self.seen[key]
        gid, ver, _ = guard
        if grounding is None:
            c = apply_text(self.client, gid, ver, source, text)
        else:
            c = apply_text(
                self.client, gid, ver, source, text, grounding_source=grounding, query=INSTRUCTION
            )
        row = {
            "key": key,
            **c.to_dict(),
            "scores": [
                f
                for a in c.assessments
                for f in (a.get("contextualGroundingPolicy") or {}).get("filters", [])
            ],
        }
        add_usage(self.usage, c.usage)
        add_usage(self.new_usage, c.usage)
        with self.cache.open("a") as fh:
            fh.write(json.dumps(row, default=str) + "\n")
        self.seen[key] = row
        if estimate_cost_usd(self.usage) > COST_STOP:
            raise SystemExit("cost stop reached")
        return row


def load_preds(path: Path) -> dict[tuple[str, str], str]:
    out = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        r = json.loads(line)
        out[(r["set"], r["id"])] = r["output"]
    return out


def rate(n, d):
    return None if d == 0 else round(n / d, 4)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=ROOT / "results/aws/guardrails-v2")
    ap.add_argument("--profile", default="nightmoose-bots")
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    client = boto3.Session(profile_name=a.profile, region_name="us-east-1").client(
        "bedrock-runtime"
    )
    run = Runner(client, a.out / "calls.jsonl")

    held = _splits(False)[0]["test"]
    tickets = [("heldout", r.ticket_id, r.text, None) for r in held]
    tickets += [("hostile", t.ticket_id, t.text, t) for t in TICKETS]
    tickets += [("hostile_v2", t.ticket_id, t.text, t) for t in TICKETS_V2]
    text_of = {(s, i): txt for s, i, txt, _ in tickets}

    # 1) Input filter on every ticket.
    inp = {(s, i): run.check(f"inj|INPUT|{s}|{i}", INJ, "INPUT", txt) for s, i, txt, _ in tickets}

    report: dict[str, Any] = {"guardrails": {"injection": INJ[:2], "grounding": GRD[:2]}}

    def input_block(sets):
        rows = [(k, v) for k, v in inp.items() if k[0] in sets]
        return sum(v["blocked"] for _, v in rows), len(rows)

    b, n = input_block({"heldout"})
    report["input_filter"] = {
        "heldout_benign_blocked": b,
        "heldout_n": n,
        "false_positive_rate": rate(b, n),
    }
    for s, tlist in (("hostile", TICKETS), ("hostile_v2", TICKETS_V2)):
        inj = [t for t in tlist if t.kind == "injection"]
        other = [t for t in tlist if t.kind != "injection"]
        ib = sum(inp[(s, t.ticket_id)]["blocked"] for t in inj)
        ob = sum(inp[(s, t.ticket_id)]["blocked"] for t in other)
        report["input_filter"][s] = {
            "injection_tickets": len(inj),
            "injections_blocked": ib,
            "catch_rate": rate(ib, len(inj)),
            "non_injection_hostile": len(other),
            "non_injection_blocked": ob,
            "blocked_ids": sorted(t.ticket_id for t in tlist if inp[(s, t.ticket_id)]["blocked"]),
        }

    # 2) Output filters on model outputs.
    systems = {"v2-s1-dpo (saved Nov 10 eval)": load_preds(SAVED)}
    if ENDPOINT.exists():
        systems["v2-s1-dpo (live endpoint)"] = load_preds(ENDPOINT)
    report["systems"] = {}
    for name, preds in systems.items():
        tag = "saved" if "saved" in name else "endpoint"
        out_inj, out_grd = {}, {}
        for (s, i), o in preds.items():
            out_inj[(s, i)] = run.check(f"inj|OUTPUT|{tag}|{s}|{i}", INJ, "OUTPUT", o)
            out_grd[(s, i)] = run.check(
                f"grd|OUTPUT|{tag}|{s}|{i}", GRD, "OUTPUT", o, grounding=text_of[(s, i)]
            )
        sysrep: dict[str, Any] = {"rows": len(preds)}
        for s in ("heldout", "hostile", "hostile_v2"):
            keys = [k for k in preds if k[0] == s]
            if not keys:
                continue
            blk = {
                "n": len(keys),
                "injection_guard_output_blocked": sum(out_inj[k]["blocked"] for k in keys),
                "grounding_guard_output_blocked": sum(out_grd[k]["blocked"] for k in keys),
                "input_blocked": sum(inp[k]["blocked"] for k in keys),
            }
            if s != "heldout":
                tl = {t.ticket_id: t for t in (TICKETS if s == "hostile" else TICKETS_V2)}
                rows = {k: score_row(tl[k[1]], preds[k]) for k in keys}
                inj_keys = [k for k in keys if tl[k[1]].kind == "injection"]
                followed = [k for k in inj_keys if rows[k]["injection_followed"]]
                wrong = [k for k in keys if rows[k]["accepted_but_wrong"]]
                unsup = [k for k in keys if rows[k]["unsupported_fields"]]

                def released(ks, guard):
                    return [
                        k
                        for k in ks
                        if not inp[k]["blocked"]
                        and not (
                            guard
                            and (
                                out_inj[k]["blocked"] or (guard == "both" and out_grd[k]["blocked"])
                            )
                        )
                    ]

                blk.update(
                    {
                        "injection_tickets": len(inj_keys),
                        "injections_followed_by_model": len(followed),
                        "followed_released_no_guardrails": len(followed),
                        "followed_released_injection_guard": len(released(followed, "inj")),
                        "followed_released_both_guards": len(released(followed, "both")),
                        "gate_accepted_but_wrong_no_guardrails": len(wrong),
                        "gate_accepted_but_wrong_injection_guard": len(released(wrong, "inj")),
                        "gate_accepted_but_wrong_both_guards": len(released(wrong, "both")),
                        "rows_with_unsupported_values": len(unsup),
                        "unsupported_rows_caught_by_grounding": sum(
                            out_grd[k]["blocked"] for k in unsup
                        ),
                        "followed_ids": sorted(k[1] for k in followed),
                    }
                )
            blk["grounding_blocked_ids"] = sorted(k[1] for k in keys if out_grd[k]["blocked"])
            sysrep[s] = blk
        report["systems"][name] = sysrep

    lat = sorted(r["seconds"] for r in run.seen.values())
    report["calls"] = len(run.seen)
    report["apply_guardrail_latency_s"] = {
        "p50": lat[len(lat) // 2],
        "p95": lat[int(0.95 * (len(lat) - 1))],
    }
    report["usage_units_total"] = run.usage
    report["est_cost_usd_total"] = round(estimate_cost_usd(run.usage), 4)
    report["est_cost_usd_this_run"] = round(estimate_cost_usd(run.new_usage), 4)
    (a.out / "summary.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()

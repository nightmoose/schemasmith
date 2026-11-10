"""Rescore every saved prediction file and write results/aws/summary.json + SUMMARY.md.

All systems are rescored here from their raw outputs with the same code, so
the gate-v2 screen and the 64-row subset apply to every row the same way.
Intervals are 95% percentile bootstrap over tickets (2000 resamples, seed 0).
"""

from __future__ import annotations

import json
import random
from pathlib import Path

from schemasmith.contract.validator import score_outputs
from schemasmith.curation.local import assign_splits
from schemasmith.data.generator import generate_tickets
from schemasmith.hostile import TICKETS, score_hostile
from schemasmith.hostile.tickets_v2 import TICKETS_V2

ROOT = Path(__file__).resolve().parents[1] / "results" / "aws"
HELD = assign_splits(generate_tickets(1280, 42), 42)["test"]
POS = {r.ticket_id: i for i, r in enumerate(HELD)}


def load(path: Path) -> dict[str, dict[str, str]]:
    out: dict[str, dict[str, str]] = {"heldout": {}, "hostile": {}, "hostile_v2": {}}
    for line in path.read_text(encoding="utf-8").splitlines():
        row = json.loads(line)
        out[row.get("set", "heldout")][row["id"]] = row["output"]
    return out


def held_card(outputs: dict[str, str], rows=HELD) -> dict:
    texts = [outputs[r.ticket_id] for r in rows]
    card = score_outputs(
        texts, [r.gold for r in rows], [r.text for r in rows], [r.language for r in rows]
    )
    return card.to_dict()


def per_row(outputs: dict[str, str]):
    """(contract_valid, field_hits_of_7, unsupported, grounded) per held-out ticket."""
    from schemasmith.contract.validator import score_output

    res = []
    for r in HELD:
        s = score_output(outputs[r.ticket_id], r.gold, r.text)
        hits = sum(bool(v) for v in s.field_hits.values())
        res.append(
            (bool(s.contract_valid), hits, len(s.unsupported_fields), len(s.grounded_fields_scored))
        )
    return res


def boot(values, stat, n=2000, seed=0):
    rng = random.Random(seed)
    k = len(values)
    draws = sorted(stat([values[rng.randrange(k)] for _ in range(k)]) for _ in range(n))
    return [draws[int(0.025 * n)], draws[int(0.975 * n) - 1]]


def summarize_system(pred: dict[str, dict[str, str]]) -> dict:
    out: dict = {}
    if len(pred["heldout"]) == len(HELD):
        full = held_card(pred["heldout"])
        rows64 = [r for r in HELD if int(r.ticket_id[2:]) < 100320]
        sub = held_card(pred["heldout"], rows64)
        pr = per_row(pred["heldout"])
        out["heldout"] = full
        out["heldout64"] = sub
        out["heldout_ci"] = {
            "contract_valid_rate": boot(pr, lambda xs: sum(x[0] for x in xs) / len(xs)),
        }
        out["heldout_ci"]["field_exact_match"] = boot(
            pr, lambda xs: sum(x[1] for x in xs) / (7 * len(xs))
        )
        out["heldout_ci"]["unsupported_value_rate"] = boot(
            pr, lambda xs: sum(x[2] for x in xs) / max(1, sum(x[3] for x in xs))
        )
    for name, tickets in (("hostile", TICKETS), ("hostile_v2", TICKETS_V2)):
        if len(pred[name]) == len(tickets):
            card = score_hostile(tickets, [pred[name][t.ticket_id] for t in tickets])
            out[name] = card
    return out


def main() -> int:
    systems: dict[str, dict] = {}
    for f in sorted((ROOT / "runs").glob("*/eval/predictions_*.jsonl")):
        tag = f.parts[-3]
        name = f.stem.replace("predictions_", "")
        systems[f"{tag}/{name}"] = summarize_system(load(f))
    bed = ROOT / "bedrock"
    for f in sorted(bed.glob("predictions_heldout_*.jsonl")):
        model = f.stem.replace("predictions_heldout_", "")
        pred = {"heldout": {}, "hostile": {}, "hostile_v2": {}}
        for set_name in pred:
            p = bed / f"predictions_{set_name}_{model}.jsonl"
            for line in p.read_text(encoding="utf-8").splitlines():
                row = json.loads(line)
                pred[set_name][row["id"]] = row["output"]
        systems[f"bedrock/{model}"] = summarize_system(pred)
    for card in systems.values():
        for key in ("hostile", "hostile_v2"):
            if key in card:
                card[key] = {k: v for k, v in card[key].items() if k != "rows"}
    (ROOT / "summary.json").write_text(json.dumps(systems, indent=2) + "\n", encoding="utf-8")
    for name, c in systems.items():
        h = c.get("heldout", {})
        x = c.get("hostile", {})
        y = c.get("hostile_v2", {})
        print(
            f"{name:40s} held valid {h.get('schema_valid_rate', 0):.3f} "
            f"fem {h.get('field_exact_match', 0):.4f} unsup {h.get('unsupported_value_rate', 0):.4f}"
            f" | hx valid {x.get('schema_valid')} fm {x.get('field_match', 0):.3f} "
            f"inj {x.get('injections_refused')}/6 forb {x.get('forbidden_value_written')}/7 "
            f"aw {x.get('accepted_but_wrong')} aw2 {x.get('gate_v2_accepted_but_wrong')}"
            f" | hy valid {y.get('schema_valid')} fm {y.get('field_match', 0):.3f} "
            f"inj {y.get('injections_refused')}/10 forb {y.get('forbidden_value_written')}/"
            f"{y.get('tickets_with_forbidden_values')} aw {y.get('accepted_but_wrong')} "
            f"aw2 {y.get('gate_v2_accepted_but_wrong')}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

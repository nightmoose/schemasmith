"""Command line for generate, curate, and score."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from schemasmith import __version__
from schemasmith.curation.curatorkit import curate_with_curatorkit
from schemasmith.curation.local import curate_local
from schemasmith.data.generator import generate_tickets
from schemasmith.data.loader import load_alpaca_jsonl, load_ticket_jsonl, write_jsonl
from schemasmith.eval.auditkit_bridge import run_auditkit
from schemasmith.eval.harness import evaluate_records


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="schemasmith")
    parser.add_argument("--version", action="version", version=__version__)
    sub = parser.add_subparsers(dest="command", required=True)

    generate = sub.add_parser("generate", help="write synthetic tickets as JSONL")
    generate.add_argument("--n", type=int, default=24)
    generate.add_argument("--seed", type=int, default=42)
    generate.add_argument("--out", type=Path, required=True)

    curate = sub.add_parser("curate", help="contract-gate and export a ticket JSONL")
    curate.add_argument("--raw", type=Path, required=True)
    curate.add_argument("--out", type=Path, required=True)
    curate.add_argument("--seed", type=int, default=42)
    curate.add_argument(
        "--curatorkit",
        action="store_true",
        help="also run the external CuratorKIT clean and dedup pass",
    )

    evaluate = sub.add_parser("eval", help="score a predictions JSONL with the contract")
    evaluate.add_argument("--data", type=Path, required=True, help="alpaca JSONL with gold output")
    evaluate.add_argument(
        "--predictions",
        type=Path,
        required=True,
        help="JSONL with id and output fields, in any order",
    )
    evaluate.add_argument("--report", type=Path, required=True)
    evaluate.add_argument(
        "--auditkit",
        action="store_true",
        help="also call AuditKIT when that package is installed",
    )

    args = parser.parse_args(argv)
    if args.command == "generate":
        return _generate(args.n, args.seed, args.out)
    if args.command == "curate":
        return _curate(args.raw, args.out, args.seed, args.curatorkit)
    if args.command == "eval":
        return _eval(args.data, args.predictions, args.report, args.auditkit)
    parser.error(f"unknown command {args.command}")
    return 2


def _generate(n: int, seed: int, out: Path) -> int:
    records = generate_tickets(n, seed)
    write_jsonl(out, [record.to_dict() for record in records])
    print(f"wrote {len(records)} tickets to {out}")
    return 0


def _curate(raw: Path, out: Path, seed: int, use_curatorkit: bool) -> int:
    records = load_ticket_jsonl(raw)
    local = curate_local(records, out / "local", seed=seed)
    print(json.dumps(local.summary(), indent=2))
    if use_curatorkit:
        summary = curate_with_curatorkit(records, out / "curatorkit", seed=seed)
        print(json.dumps(summary, indent=2))
    return 0


def _eval(data: Path, predictions: Path, report: Path, use_auditkit: bool) -> int:
    records = load_alpaca_jsonl(data)
    by_id = {}
    for row in _read_predictions(predictions):
        by_id[str(row["id"])] = str(row["output"])
    missing = [record.ticket_id for record in records if record.ticket_id not in by_id]
    if missing:
        print(f"predictions missing ids: {missing[:5]}", file=sys.stderr)
        return 2
    outputs = [by_id[record.ticket_id] for record in records]
    card = evaluate_records(records, outputs=outputs)
    payload: dict[str, object] = {
        "measurement": "scored the provided prediction file; not a base-model run",
        "scorecard": card.to_dict(),
    }
    if use_auditkit:
        audit = run_auditkit(records, outputs)
        payload["auditkit"] = audit if audit is not None else {"available": False}
    report.parent.mkdir(parents=True, exist_ok=True)
    report.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "parse_rate": card.parse_rate,
                "schema_valid_rate": card.schema_valid_rate,
                "field_exact_match": card.field_exact_match,
                "report": str(report),
            },
            indent=2,
        )
    )
    return 0


def _read_predictions(path: Path) -> list[dict[str, object]]:
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        value = json.loads(line)
        if "id" not in value or "output" not in value:
            raise ValueError("each prediction needs id and output")
        rows.append(value)
    return rows

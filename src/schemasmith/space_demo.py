"""Mock-mode data for the Space. No model is loaded."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from schemasmith.contract.schema import load_odcs
from schemasmith.contract.validator import score_output
from schemasmith.data.generator import generate_tickets
from schemasmith.data.preferences import FAILURE_MODES, build_preference_pairs
from schemasmith.data.records import TicketRecord
from schemasmith.eval.compare import METRICS, load_results

SAMPLE_SEED = 42


def sample_records(n: int = 8) -> list[TicketRecord]:
    return generate_tickets(n, SAMPLE_SEED)


def record_by_id(ticket_id: str, n: int = 8) -> TicketRecord:
    for record in sample_records(n):
        if record.ticket_id == ticket_id:
            return record
    raise KeyError(ticket_id)


def labeled_json(record: TicketRecord) -> str:
    return json.dumps(record.gold, indent=2, ensure_ascii=False)


def example_failure(record: TicketRecord) -> str:
    """A planted rejected string for this ticket. Not a model output."""
    pairs = build_preference_pairs([record], seed=SAMPLE_SEED)
    return pairs[0]["rejected"]


def contract_text() -> str:
    odcs = load_odcs()
    lines = [
        f"kind: {odcs['kind']}",
        f"name: {odcs['name']}",
    ]
    for prop in odcs["schema"][0]["properties"]:
        extra = ""
        if "enum" in prop:
            extra = " enum[" + ",".join(prop["enum"]) + "]"
        if prop.get("nullable"):
            extra += " nullable"
        if "pattern" in prop:
            extra += f" pattern {prop['pattern']}"
        lines.append(f"- {prop['name']}: {prop['logicalType']}{extra}")
    lines.append("rule: refund_amount_usd is null unless category is billing")
    return "\n".join(lines)


def check_output(record: TicketRecord, output: str) -> dict[str, Any]:
    score = score_output(output, record.gold, record.text)
    hits = sum(score.field_hits.values())
    return {
        "parsed": score.parsed,
        "schema_valid": score.schema_valid,
        "field_hits": f"{hits}/{len(score.field_hits)}",
        "unsupported_fields": score.unsupported_fields,
        "errors": score.errors,
        "banner": "Validator result for the text in the box. Not a model score.",
    }


def scorecard_markdown(results_path: str | None) -> str:
    if not results_path:
        return _tbd_table()
    path = Path(results_path)
    if not path.is_file():
        return _tbd_table() + f"\n\nNo results file at `{path}`."
    payload = load_results(path)
    systems = payload["systems"]
    lines = [
        "| Metric | Base | SFT | DPO |",
        "| --- | --- | --- | --- |",
    ]
    for key, label in METRICS:
        lines.append(
            "| {label} | {base} | {sft} | {dpo} |".format(
                label=label,
                base=_fmt(systems["base"][key]),
                sft=_fmt(systems["sft"][key]),
                dpo=_fmt(systems["dpo"][key]),
            )
        )
    lines.append("")
    lines.append(f"Rows in the file: {payload['n']}.")
    return "\n".join(lines)


def failure_mode_for_sample() -> str:
    return FAILURE_MODES[(SAMPLE_SEED + 0) % len(FAILURE_MODES)]


def _tbd_table() -> str:
    lines = [
        "Scorecard is TBD. A compare results file has not been loaded.",
        "",
        "| Metric | Base | SFT | DPO |",
        "| --- | --- | --- | --- |",
    ]
    for _key, label in METRICS:
        lines.append(f"| {label} | TBD | TBD | TBD |")
    return "\n".join(lines)


def _fmt(value: object) -> str:
    if value is None:
        return "n/a"
    return f"{float(value) * 100:.1f}%"

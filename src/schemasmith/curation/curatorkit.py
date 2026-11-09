"""Optional CuratorKIT pass.

CuratorKIT is an external package under the Lexsi Labs Source Available
License. This module imports it only when a curation run asks for it.
The source is not vendored in this repository.

The call below was checked against CuratorKIT 0.1.23: ``Curator`` and
``CuratorConfig`` run on a local alpaca JSONL with no LLM, ``dedup="exact"``,
``export_formats`` including ``"alpaca"`` and ``"messages"``, and a
``preprocessing_fn`` that returns ``None`` to reject a row
(``preprocessing_fn_returned_none``).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from schemasmith.contract.validator import extract_json_object, validate_instance
from schemasmith.curation.local import alpaca_row
from schemasmith.data.loader import write_jsonl
from schemasmith.data.records import TicketRecord


class CuratorKitUnavailable(RuntimeError):
    """Raised when the external curatorkit package is not installed."""


def contract_preprocessing_fn(row: dict[str, Any]) -> dict[str, Any] | None:
    """Drop a row whose alpaca ``output`` fails the ticket contract.

    CuratorKIT treats a ``None`` return as a rejection. Its own SchemaGate
    checks token counts and task fields, not this data contract.
    """
    output = row.get("output")
    if not isinstance(output, str):
        return None
    parsed, _error = extract_json_object(output)
    if parsed is None:
        return None
    ok, _errors = validate_instance(parsed)
    if not ok:
        return None
    return row


def curate_with_curatorkit(
    records: list[TicketRecord],
    output_dir: Path,
    *,
    seed: int = 42,
    min_tokens: int = 3,
) -> dict[str, Any]:
    """Run CuratorKIT clean + exact dedup on contract-gated alpaca rows.

    No generator LLM is configured. Splits are CuratorKIT's own shuffle
    (``output_split_seed``), written as ``train``, ``validation``, and
    ``test``. That shuffle is not the same function as ``curate_local``.
    """
    try:
        from curatorkit import Curator, CuratorConfig
    except ImportError as exc:
        raise CuratorKitUnavailable(
            "CuratorKIT is not installed. It stays out of the MIT dependency "
            "set. Install it from requirements-lexsi.txt when you want this pass."
        ) from exc

    output_dir.mkdir(parents=True, exist_ok=True)
    source = output_dir / "_input_alpaca.jsonl"
    write_jsonl(source, [alpaca_row(record) for record in records])
    curated = output_dir / "curated"
    result = Curator(
        CuratorConfig(
            dataset=str(source),
            format="alpaca",
            dedup="exact",
            clean=True,
            clean_fields=["instruction", "input"],
            schema_gate=True,
            min_tokens=min_tokens,
            export_formats=["alpaca", "messages"],
            output_dir=str(curated),
            output_split={"train": 0.75, "validation": 0.125, "test": 0.125},
            output_split_seed=seed,
            name="schemasmith_support_tickets",
            write_hf_readme=True,
            preprocessing_fn=contract_preprocessing_fn,
        )
    ).run()
    summary = {
        "producer": "curatorkit",
        "passed_count": len(result.passed),
        "rejected_count": len(result.rejected),
        "stage_counts": result.stage_counts,
        "wall_clock_seconds": result.wall_clock_seconds,
        "output_dir": str(result.output_dir),
        "input_file": str(source),
    }
    (output_dir / "curatorkit_summary.json").write_text(
        json.dumps(summary, indent=2, default=str) + "\n",
        encoding="utf-8",
    )
    return summary

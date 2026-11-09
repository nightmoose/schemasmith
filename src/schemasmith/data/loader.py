"""JSONL readers and writers. No model downloads."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from schemasmith.data.records import TicketRecord


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            value = json.loads(stripped)
            if not isinstance(value, dict):
                raise ValueError(f"{path}:{line_number} is not a JSON object")
            rows.append(value)
    return rows


def load_ticket_jsonl(path: Path) -> list[TicketRecord]:
    return [TicketRecord.from_dict(row) for row in load_jsonl(path)]


def load_alpaca_jsonl(path: Path) -> list[TicketRecord]:
    """Read an alpaca export back into records.

    ``input`` is the ticket text and ``output`` is the gold JSON object.
    """
    records: list[TicketRecord] = []
    for row in load_jsonl(path):
        gold = json.loads(row["output"])
        if not isinstance(gold, dict):
            raise ValueError("alpaca output is not a JSON object")
        records.append(
            TicketRecord(
                ticket_id=str(gold["ticket_id"]),
                language=str(gold["language"]),
                text=str(row["input"]),
                gold=gold,
                keep=True,
                defect=None,
            )
        )
    return records

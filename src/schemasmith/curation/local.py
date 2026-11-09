"""Contract gate and exact dedup that do not import Lexsi packages."""

from __future__ import annotations

import json
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from schemasmith.contract.validator import validate_instance
from schemasmith.data.loader import write_jsonl
from schemasmith.data.records import TicketRecord
from schemasmith.prompts import INSTRUCTION, user_content


@dataclass
class CurationResult:
    input_count: int
    kept: list[TicketRecord]
    rejected: list[dict[str, str]]
    splits: dict[str, list[TicketRecord]]
    output_dir: Path
    seed: int

    def summary(self) -> dict[str, Any]:
        return {
            "producer": "schemasmith",
            "seed": self.seed,
            "input_count": self.input_count,
            "kept_count": len(self.kept),
            "rejected_count": len(self.rejected),
            "rejected": self.rejected,
            "split_counts": {name: len(rows) for name, rows in self.splits.items()},
            "output_dir": str(self.output_dir),
        }


@dataclass
class _Gate:
    kept: list[TicketRecord] = field(default_factory=list)
    rejected: list[dict[str, str]] = field(default_factory=list)


def curate_local(
    records: list[TicketRecord],
    output_dir: Path,
    *,
    seed: int = 42,
) -> CurationResult:
    """Drop contract failures and exact duplicate texts, then write exports.

    Exports are alpaca JSONL and chat-messages JSONL under train, validation,
    and test. The messages join matches CuratorKIT's messages exporter.
    """
    gated = _gate(records)
    splits = assign_splits(gated.kept, seed)
    output_dir.mkdir(parents=True, exist_ok=True)
    for split_name, rows in splits.items():
        split_dir = output_dir / split_name
        write_jsonl(split_dir / "sft_alpaca.jsonl", [alpaca_row(row) for row in rows])
        write_jsonl(split_dir / "sft_messages.jsonl", [messages_row(row) for row in rows])
    manifest = {
        "producer": "schemasmith",
        "seed": seed,
        "input_count": len(records),
        "kept_count": len(gated.kept),
        "rejected": gated.rejected,
        "split_counts": {name: len(rows) for name, rows in splits.items()},
        "note": ("Local contract gate and exact text dedup. This folder is not a CuratorKIT run."),
    }
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    (output_dir / "README.md").write_text(_hf_readme(), encoding="utf-8")
    return CurationResult(
        input_count=len(records),
        kept=gated.kept,
        rejected=gated.rejected,
        splits=splits,
        output_dir=output_dir,
        seed=seed,
    )


def alpaca_row(record: TicketRecord) -> dict[str, str]:
    return {
        "instruction": INSTRUCTION,
        "input": record.text,
        "output": json.dumps(record.gold, ensure_ascii=False),
    }


def messages_row(record: TicketRecord) -> dict[str, Any]:
    return {
        "messages": [
            {"role": "user", "content": user_content(record.text)},
            {
                "role": "assistant",
                "content": json.dumps(record.gold, ensure_ascii=False),
            },
        ]
    }


def assign_splits(
    records: list[TicketRecord],
    seed: int,
    *,
    train_fraction: float = 0.75,
    validation_fraction: float = 0.125,
) -> dict[str, list[TicketRecord]]:
    """Shuffle with ``seed`` and cut train / validation / test.

    Sets smaller than 3 stay entirely in train so a tiny smoke run still
    has something to read. Larger sets keep at least one validation row
    and one test row.
    """
    rng = random.Random(seed)
    items = list(records)
    rng.shuffle(items)
    n = len(items)
    if n < 3:
        return {"train": items, "validation": [], "test": []}
    n_train = max(1, int(n * train_fraction))
    n_val = max(1, int(n * validation_fraction))
    if n_train + n_val >= n:
        n_val = 1
        n_train = n - 2
    return {
        "train": items[:n_train],
        "validation": items[n_train : n_train + n_val],
        "test": items[n_train + n_val :],
    }


def _hf_readme() -> str:
    """Dataset card so AlignTune can load the local export with subset sft_alpaca."""
    files = []
    for split in ("train", "validation", "test"):
        files.append(f"    - split: {split}\n      path: {split}/sft_alpaca.jsonl")
    alpaca_files = "\n".join(files)
    message_files = alpaca_files.replace("sft_alpaca.jsonl", "sft_messages.jsonl")
    return (
        "---\n"
        "configs:\n"
        "- config_name: sft_alpaca\n"
        "  default: true\n"
        "  data_files:\n"
        f"{alpaca_files}\n"
        "- config_name: sft_messages\n"
        "  data_files:\n"
        f"{message_files}\n"
        "---\n\n"
        "# SchemaSmith local export\n\n"
        "Contract gate and exact dedup. This folder is not a CuratorKIT run.\n"
    )


def _gate(records: list[TicketRecord]) -> _Gate:
    seen: set[str] = set()
    result = _Gate()
    for record in records:
        if not record.keep:
            result.rejected.append(
                {"ticket_id": record.ticket_id, "reason": record.defect or "marked_drop"}
            )
            continue
        ok, errors = validate_instance(record.gold)
        if not ok:
            result.rejected.append({"ticket_id": record.ticket_id, "reason": "; ".join(errors)})
            continue
        key = " ".join(record.text.split())
        if key in seen:
            result.rejected.append({"ticket_id": record.ticket_id, "reason": "exact_duplicate"})
            continue
        seen.add(key)
        result.kept.append(record)
    return result

"""Preference pairs for DPO.

Chosen is the contract-valid gold JSON. Rejected is one planted failure.
The pairs are synthetic labels, not model outputs.
"""

from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any

from schemasmith.contract.validator import validate_instance
from schemasmith.data.loader import write_jsonl
from schemasmith.data.records import TicketRecord
from schemasmith.prompts import user_content

FAILURE_MODES: tuple[str, ...] = (
    "broken_json",
    "missing_field",
    "invented_field",
    "wrong_language",
    "invented_refund",
)

INVENTED_REFUND = 98.0
WRONG_LANGUAGE = "zu"


def build_preference_pairs(
    records: list[TicketRecord],
    seed: int = 42,
) -> list[dict[str, Any]]:
    """One pair per record. Mode cycles with the seed so a longer list is stable."""
    if not records:
        raise ValueError("records must not be empty")
    pairs: list[dict[str, Any]] = []
    for index, record in enumerate(records):
        mode = FAILURE_MODES[(seed + index) % len(FAILURE_MODES)]
        chosen = json.dumps(record.gold, ensure_ascii=False)
        rejected = _reject(record, mode)
        if rejected == chosen:
            raise RuntimeError(f"rejected matched chosen for {record.ticket_id}")
        ok, errors = validate_instance(record.gold)
        if not ok:
            raise ValueError(f"chosen gold is not contract-valid: {errors}")
        pairs.append(
            {
                "ticket_id": record.ticket_id,
                "failure_mode": mode,
                "prompt": user_content(record.text),
                "chosen": chosen,
                "rejected": rejected,
            }
        )
    return pairs


def write_dpo_dataset(
    records: list[TicketRecord],
    output_dir: Path,
    *,
    seed: int = 42,
) -> dict[str, Any]:
    """Write a small dataset folder AlignTune can load with config name ``dpo``.

    Validation is a held-out slice of these pairs. Do not pass the eval test
    split into this function if that split is reserved for the final compare.
    """
    pairs = build_preference_pairs(records, seed)
    train, validation = _split_pairs(pairs, seed)
    output_dir.mkdir(parents=True, exist_ok=True)
    _write_split(output_dir / "train" / "dpo.jsonl", train)
    _write_split(output_dir / "validation" / "dpo.jsonl", validation)
    write_jsonl(output_dir / "pairs.jsonl", pairs)
    (output_dir / "README.md").write_text(_dataset_card(), encoding="utf-8")
    summary = {
        "producer": "schemasmith",
        "seed": seed,
        "pair_count": len(pairs),
        "split_counts": {"train": len(train), "validation": len(validation)},
        "failure_modes": sorted({row["failure_mode"] for row in pairs}),
        "output_dir": str(output_dir),
        "note": "Chosen is labeled gold. Rejected rows are planted failures, not model outputs.",
    }
    (output_dir / "manifest.json").write_text(
        json.dumps(summary, indent=2) + "\n",
        encoding="utf-8",
    )
    return summary


def _reject(record: TicketRecord, mode: str) -> str:
    gold = dict(record.gold)
    if mode == "broken_json":
        return json.dumps(gold, ensure_ascii=False)[:-1]
    if mode == "missing_field":
        gold.pop("language", None)
        return json.dumps(gold, ensure_ascii=False)
    if mode == "invented_field":
        gold["courier_note"] = "please hurry"
        return json.dumps(gold, ensure_ascii=False)
    if mode == "wrong_language":
        gold["language"] = "en" if record.language != "en" else WRONG_LANGUAGE
        return json.dumps(gold, ensure_ascii=False)
    if mode == "invented_refund":
        gold["refund_amount_usd"] = INVENTED_REFUND
        return json.dumps(gold, ensure_ascii=False)
    raise ValueError(f"unknown failure mode {mode}")


def _split_pairs(
    pairs: list[dict[str, Any]],
    seed: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    if len(pairs) < 2:
        return list(pairs), []
    items = list(pairs)
    random.Random(seed).shuffle(items)
    n_val = max(1, int(len(items) * 0.125))
    if n_val >= len(items):
        n_val = 1
    return items[n_val:], items[:n_val]


def _write_split(path: Path, rows: list[dict[str, Any]]) -> None:
    slim = [
        {"prompt": row["prompt"], "chosen": row["chosen"], "rejected": row["rejected"]}
        for row in rows
    ]
    write_jsonl(path, slim)


def _dataset_card() -> str:
    return """---
configs:
- config_name: dpo
  default: true
  data_files:
  - split: train
    path: train/dpo.jsonl
  - split: validation
    path: validation/dpo.jsonl
---

# SchemaSmith preference pairs

Chosen responses are contract-valid labels. Rejected responses are planted
failure modes. This folder is not a measured model comparison.
"""

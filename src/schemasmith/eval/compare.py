"""Held-out compare file and the README table it is allowed to fill.

The table in the README stays TBD until a results file from this module
is applied. This module does not download a model.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from schemasmith.contract.validator import Scorecard
from schemasmith.training import MODEL_ID

MEASUREMENT = "held-out model compare"
SYSTEMS = ("base", "sft", "dpo")
METRICS: tuple[tuple[str, str], ...] = (
    ("parse_rate", "Parse rate"),
    ("schema_valid_rate", "Schema-valid rate"),
    ("field_exact_match", "Field-level exact match"),
    ("unsupported_value_rate", "Unsupported-value rate"),
    ("non_english_contract_valid_rate", "Non-English contract-valid rate"),
)
START = "<!-- SCHEMASMITH_METRICS_START -->"
END = "<!-- SCHEMASMITH_METRICS_END -->"


def results_from_scorecards(
    cards: dict[str, Scorecard],
    *,
    data_path: str,
    sft_adapter: str,
    dpo_adapter: str,
) -> dict[str, Any]:
    missing = [name for name in SYSTEMS if name not in cards]
    if missing:
        raise ValueError(f"missing scorecards: {missing}")
    n_values = {cards[name].n for name in SYSTEMS}
    if len(n_values) != 1:
        raise ValueError("base, sft, and dpo must be scored on the same rows")
    return {
        "measurement": MEASUREMENT,
        "model_id": MODEL_ID,
        "core_model": MODEL_ID,
        "data": data_path,
        "n": next(iter(n_values)),
        "adapters": {"sft": sft_adapter, "dpo": dpo_adapter},
        "systems": {name: cards[name].to_dict() for name in SYSTEMS},
    }


def write_results(payload: dict[str, Any], path: Path) -> None:
    validate_results(payload)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def load_results(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    validate_results(payload)
    return payload


def validate_results(payload: dict[str, Any]) -> None:
    if payload.get("measurement") != MEASUREMENT:
        raise ValueError(
            "results file is not a held-out model compare. Refusing to treat it as a metric source."
        )
    if payload.get("model_id") != MODEL_ID or payload.get("core_model") != MODEL_ID:
        raise ValueError("results file must name Tiny Aya Global as the core model")
    systems = payload.get("systems")
    if not isinstance(systems, dict):
        raise ValueError("results file has no systems object")
    for name in SYSTEMS:
        block = systems.get(name)
        if not isinstance(block, dict):
            raise ValueError(f"results file is missing systems.{name}")
        for key, _label in METRICS:
            if key not in block:
                raise ValueError(f"systems.{name} is missing {key}")
            value = block[key]
            if value is not None and not isinstance(value, int | float):
                raise ValueError(f"systems.{name}.{key} is not a number")


def render_metrics_table(payload: dict[str, Any]) -> str:
    validate_results(payload)
    systems = payload["systems"]
    lines = [
        "| Metric | Base `tiny-aya-global` | SFT | DPO | Status |",
        "| --- | --- | --- | --- | --- |",
    ]
    for key, label in METRICS:
        lines.append(
            "| {label} | {base} | {sft} | {dpo} | measured |".format(
                label=label,
                base=_fmt(systems["base"][key]),
                sft=_fmt(systems["sft"][key]),
                dpo=_fmt(systems["dpo"][key]),
            )
        )
    lines.append("")
    lines.append(
        f"Held-out rows: {payload['n']}. Source file must stay next to this table. "
        "Status is measured only because this block was rewritten from that file."
    )
    return "\n".join(lines)


def update_readme(readme_path: Path, payload: dict[str, Any]) -> None:
    """Replace the marked metrics block and leave the rest of the file alone."""
    text = readme_path.read_text(encoding="utf-8")
    start = text.find(START)
    end = text.find(END)
    if start < 0 or end < 0 or end < start:
        raise ValueError(f"{readme_path} is missing the metrics markers")
    replacement = START + "\n" + render_metrics_table(payload) + "\n" + END
    updated = text[:start] + replacement + text[end + len(END) :]
    readme_path.write_text(updated, encoding="utf-8")


def _fmt(value: object) -> str:
    if value is None:
        return "n/a"
    return f"{float(value) * 100:.1f}%"

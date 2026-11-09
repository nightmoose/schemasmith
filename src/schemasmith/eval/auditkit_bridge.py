"""Optional AuditKIT scores next to the contract validator.

AuditKIT 1.1.6 was imported and ``evaluate`` was called on a two-row fixture
with ``IsJson``, ``JsonSchemaValid``, and ``FieldF1``. The public argument
is ``scorers``, not the ``metrics`` name shown in some README snippets.
``JsonSchemaValid(..., backend="jsonschema", strip_fences=True)`` needs the
jsonschema package, which this project already depends on. ``IsJson`` does
not strip markdown fences.

AuditKIT is an external LSAL package. Import happens only inside
``run_auditkit``.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from typing import Any

from schemasmith.data.records import TicketRecord
from schemasmith.paths import schema_path
from schemasmith.prompts import user_content


def auditkit_available() -> bool:
    try:
        import auditkit  # noqa: F401
    except ImportError:
        return False
    return True


def run_auditkit(
    records: Sequence[TicketRecord],
    outputs: Sequence[str],
) -> dict[str, Any] | None:
    """Score already-produced strings. Returns None when AuditKIT is absent.

    The model callable echoes ``outputs`` in order. No weights are loaded.
    """
    if len(records) != len(outputs):
        raise ValueError("records and outputs must have the same length")
    try:
        import auditkit as ak
    except ImportError:
        return None

    samples = [
        ak.Sample(
            input=user_content(record.text),
            target=json.dumps(record.gold, ensure_ascii=False),
            id=record.ticket_id,
            metadata={"language": record.language},
        )
        for record in records
    ]
    cursor = {"index": 0}

    def _model(_prompts: list[str]) -> list[str]:
        start = cursor["index"]
        end = start + len(_prompts)
        cursor["index"] = end
        return list(outputs[start:end])

    result = ak.evaluate(
        samples,
        _model,
        scorers=[
            ak.IsJson(),
            ak.JsonSchemaValid(schema_path(), backend="jsonschema", strip_fences=True),
            ak.FieldF1(normalize_numbers=True, number_tol=1e-6, strip_fences=True),
        ],
    )
    return {
        "headline": dict(result.headline),
        "fingerprint": result.fingerprint,
        "summary": result.summary(),
        "notes": [
            "is_json does not strip markdown fences on AuditKIT 1.1.6",
            "json_schema_valid uses strip_fences and the jsonschema backend",
            "field_f1 is AuditKIT's field score, separate from field_exact_match",
        ],
    }

"""Score model outputs against the support-ticket contract.

Rates are computed from the outputs passed in. Nothing here is a model
benchmark until those outputs come from a model run.
"""

from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass, field
from typing import Any

from jsonschema import Draft202012Validator
from jsonschema.exceptions import ValidationError

from schemasmith.contract.schema import FIELDS, GROUNDED_FIELDS, load_schema

_AMOUNT = re.compile(r"\d+(?:[.,]\d+)?")
_ORDER = re.compile(r"\bORD-\d{6}\b")
_TICKET = re.compile(r"\bT-[A-Za-z0-9]+\b")

REFUND_RULE = "refund_amount_usd must be null unless category is billing"


@dataclass
class OutputScore:
    parsed: bool
    schema_valid: bool
    contract_valid: bool
    field_hits: dict[str, bool]
    unsupported_fields: list[str]
    grounded_fields_scored: list[str]
    errors: list[str] = field(default_factory=list)
    parsed_object: dict[str, Any] | None = None


@dataclass
class Scorecard:
    """Aggregate rates for one list of outputs.

    ``schema_valid_rate`` is the fraction of outputs that parse as a JSON
    object and satisfy the JSON Schema, including the refund rule and the
    ISO 639-1 language enum. ``contract_valid_rate`` is the same check.
    Unparsed outputs count as not valid and as zero field hits.
    ``unsupported_value_rate`` is computed only over grounded fields that
    the model filled with a non-null value in a parsed object.
    """

    n: int
    n_parsed: int
    n_schema_valid: int
    n_contract_valid: int
    parse_rate: float
    schema_valid_rate: float
    contract_valid_rate: float
    field_exact_match: float
    per_field_accuracy: dict[str, float]
    unsupported_value_rate: float | None
    unsupported_field_count: int
    grounded_field_count: int
    non_english_contract_valid_rate: float | None
    n_non_english: int
    rows: list[OutputScore] = field(default_factory=list)

    def to_dict(self, *, include_rows: bool = False) -> dict[str, Any]:
        payload = asdict(self)
        payload.pop("rows")
        if include_rows:
            payload["rows"] = [
                {
                    "parsed": row.parsed,
                    "schema_valid": row.schema_valid,
                    "contract_valid": row.contract_valid,
                    "field_hits": row.field_hits,
                    "unsupported_fields": row.unsupported_fields,
                    "errors": row.errors,
                }
                for row in self.rows
            ]
        return payload


def extract_json_object(text: str) -> tuple[dict[str, Any] | None, str | None]:
    """Return the first JSON object found in ``text``.

    Accepts a raw object, one markdown fence, or an object embedded in prose.
    """
    raw = text.strip()
    candidates: list[str] = []
    if raw:
        candidates.append(raw)
    fence = re.search(r"```(?:json)?\s*(\{.*)\s*```", raw, flags=re.DOTALL | re.IGNORECASE)
    if fence:
        candidates.append(fence.group(1).strip())
    start = raw.find("{")
    if start != -1:
        depth = 0
        in_string = False
        escape = False
        for index, char in enumerate(raw[start:], start):
            if in_string:
                if escape:
                    escape = False
                elif char == "\\":
                    escape = True
                elif char == '"':
                    in_string = False
                continue
            if char == '"':
                in_string = True
            elif char == "{":
                depth += 1
            elif char == "}":
                depth -= 1
                if depth == 0:
                    candidates.append(raw[start : index + 1])
                    break
    last_error = "no JSON object found"
    for candidate in candidates:
        try:
            value = json.loads(candidate)
        except json.JSONDecodeError as exc:
            last_error = str(exc)
            continue
        if isinstance(value, dict):
            return value, None
        last_error = "JSON value is not an object"
    return None, last_error


def _validator() -> Draft202012Validator:
    return Draft202012Validator(load_schema())


def schema_errors(instance: dict[str, Any]) -> list[str]:
    errors: list[ValidationError] = sorted(
        _validator().iter_errors(instance),
        key=lambda err: list(err.path),
    )
    messages: list[str] = []
    for err in errors[:8]:
        path = ".".join(str(part) for part in err.path) or "$"
        messages.append(f"{path}: {err.message}")
    return messages


def refund_rule_ok(instance: dict[str, Any]) -> bool:
    if instance.get("category") == "billing":
        return True
    return instance.get("refund_amount_usd") is None


def validate_instance(instance: dict[str, Any]) -> tuple[bool, list[str]]:
    """Return whether ``instance`` passes the contract, plus error strings."""
    errors = schema_errors(instance)
    if not refund_rule_ok(instance) and REFUND_RULE not in " ".join(errors):
        errors.append(REFUND_RULE)
    return (not errors), errors


def _values_equal(field: str, predicted: Any, gold: Any) -> bool:
    if field == "refund_amount_usd":
        if predicted is None or gold is None:
            return predicted is None and gold is None
        if isinstance(predicted, bool) or isinstance(gold, bool):
            return False
        if isinstance(predicted, int | float) and isinstance(gold, int | float):
            return abs(float(predicted) - float(gold)) <= 1e-6
        return False
    if field == "severity":
        if isinstance(predicted, bool) or isinstance(gold, bool):
            return False
        return predicted == gold
    return predicted == gold


def amount_in_text(amount: float, text: str) -> bool:
    """True when ``amount`` appears in ``text`` outside ticket and order ids."""
    scrubbed = _ORDER.sub(" ", text)
    scrubbed = _TICKET.sub(" ", scrubbed)
    for raw in _AMOUNT.findall(scrubbed):
        try:
            found = float(raw.replace(",", "."))
        except ValueError:
            continue
        if abs(found - float(amount)) <= 0.001:
            return True
    return False


def grounded_field_supported(field: str, value: Any, source_text: str) -> bool:
    """Null means the model did not invent a value for this field."""
    if value is None:
        return True
    if field in ("ticket_id", "order_ref"):
        return str(value) in source_text
    if field == "refund_amount_usd":
        if isinstance(value, bool) or not isinstance(value, int | float):
            return False
        return amount_in_text(float(value), source_text)
    return True


def score_output(
    output: str,
    gold: dict[str, Any],
    source_text: str,
) -> OutputScore:
    parsed, parse_error = extract_json_object(output)
    if parsed is None:
        return OutputScore(
            parsed=False,
            schema_valid=False,
            contract_valid=False,
            field_hits={name: False for name in FIELDS},
            unsupported_fields=[],
            grounded_fields_scored=[],
            errors=[parse_error or "parse failed"],
            parsed_object=None,
        )
    ok, errors = validate_instance(parsed)
    hits = {name: _values_equal(name, parsed.get(name), gold.get(name)) for name in FIELDS}
    unsupported: list[str] = []
    scored: list[str] = []
    for name in GROUNDED_FIELDS:
        if name not in parsed or parsed.get(name) is None:
            continue
        scored.append(name)
        if not grounded_field_supported(name, parsed.get(name), source_text):
            unsupported.append(name)
    return OutputScore(
        parsed=True,
        schema_valid=ok,
        contract_valid=ok,
        field_hits=hits,
        unsupported_fields=unsupported,
        grounded_fields_scored=scored,
        errors=errors,
        parsed_object=parsed,
    )


def _rate(numerator: int, denominator: int) -> float:
    if denominator == 0:
        return 0.0
    return numerator / denominator


def score_outputs(
    outputs: list[str],
    golds: list[dict[str, Any]],
    sources: list[str],
    languages: list[str] | None = None,
) -> Scorecard:
    if not (len(outputs) == len(golds) == len(sources)):
        raise ValueError("outputs, golds, and sources must have the same length")
    if languages is None:
        languages = [str(gold.get("language", "")) for gold in golds]
    if len(languages) != len(outputs):
        raise ValueError("languages must have the same length as outputs")

    rows = [
        score_output(output, gold, source)
        for output, gold, source in zip(outputs, golds, sources, strict=True)
    ]
    n = len(rows)
    n_parsed = sum(row.parsed for row in rows)
    n_schema = sum(row.schema_valid for row in rows)
    field_hits = sum(sum(row.field_hits.values()) for row in rows)
    field_total = n * len(FIELDS)
    per_field = {name: _rate(sum(row.field_hits[name] for row in rows), n) for name in FIELDS}
    unsupported = sum(len(row.unsupported_fields) for row in rows)
    grounded = sum(len(row.grounded_fields_scored) for row in rows)
    non_english_indexes = [i for i, lang in enumerate(languages) if lang and lang != "en"]
    if non_english_indexes:
        non_english_hits = sum(rows[i].contract_valid for i in non_english_indexes)
        non_english_rate: float | None = non_english_hits / len(non_english_indexes)
    else:
        non_english_rate = None
    return Scorecard(
        n=n,
        n_parsed=n_parsed,
        n_schema_valid=n_schema,
        n_contract_valid=n_schema,
        parse_rate=_rate(n_parsed, n),
        schema_valid_rate=_rate(n_schema, n),
        contract_valid_rate=_rate(n_schema, n),
        field_exact_match=_rate(field_hits, field_total),
        per_field_accuracy=per_field,
        unsupported_value_rate=(unsupported / grounded) if grounded else None,
        unsupported_field_count=unsupported,
        grounded_field_count=grounded,
        non_english_contract_valid_rate=non_english_rate,
        n_non_english=len(non_english_indexes),
        rows=rows,
    )

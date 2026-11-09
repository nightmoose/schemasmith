"""Ticket extraction contract and the scorer that checks model output."""

from schemasmith.contract.schema import (
    CATEGORIES,
    FIELDS,
    GROUNDED_FIELDS,
    TIERS,
    load_odcs,
    load_schema,
)
from schemasmith.contract.validator import (
    OutputScore,
    Scorecard,
    score_output,
    score_outputs,
    validate_instance,
)

__all__ = [
    "CATEGORIES",
    "FIELDS",
    "GROUNDED_FIELDS",
    "OutputScore",
    "Scorecard",
    "TIERS",
    "load_odcs",
    "load_schema",
    "score_output",
    "score_outputs",
    "validate_instance",
]

"""Load the support-ticket contract."""

import json
from functools import lru_cache
from typing import Any

import yaml

from schemasmith.paths import odcs_path, schema_path

FIELDS: tuple[str, ...] = (
    "ticket_id",
    "customer_tier",
    "category",
    "severity",
    "order_ref",
    "refund_amount_usd",
    "language",
)

# Money and identifiers must be attested in the source text.
GROUNDED_FIELDS: tuple[str, ...] = (
    "ticket_id",
    "order_ref",
    "refund_amount_usd",
)

TIERS: tuple[str, ...] = ("free", "pro", "enterprise")
CATEGORIES: tuple[str, ...] = ("billing", "bug", "access", "feature_request")


@lru_cache(maxsize=1)
def load_schema() -> dict[str, Any]:
    return json.loads(schema_path().read_text(encoding="utf-8"))


@lru_cache(maxsize=1)
def load_odcs() -> dict[str, Any]:
    loaded = yaml.safe_load(odcs_path().read_text(encoding="utf-8"))
    if not isinstance(loaded, dict):
        raise ValueError("ODCS contract must be a mapping")
    return loaded


def language_codes() -> frozenset[str]:
    enum = load_schema()["properties"]["language"]["enum"]
    return frozenset(enum)

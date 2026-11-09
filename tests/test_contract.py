import json

from schemasmith.contract.schema import (
    CATEGORIES,
    FIELDS,
    TIERS,
    load_odcs,
    load_schema,
)
from schemasmith.contract.validator import validate_instance
from schemasmith.data.generator import generate_tickets


def _gold(**overrides):
    base = {
        "ticket_id": "T-100001",
        "customer_tier": "pro",
        "category": "billing",
        "severity": 2,
        "order_ref": "ORD-104233",
        "refund_amount_usd": 49.0,
        "language": "es",
    }
    base.update(overrides)
    return base


def test_schema_and_odcs_agree_on_fields():
    schema = load_schema()
    odcs = load_odcs()
    assert list(schema["required"]) == list(FIELDS)
    props = odcs["schema"][0]["properties"]
    assert [item["name"] for item in props] == list(FIELDS)
    assert schema["properties"]["customer_tier"]["enum"] == list(TIERS)
    assert schema["properties"]["category"]["enum"] == list(CATEGORIES)
    assert odcs["kind"] == "DataContract"
    assert len(schema["properties"]["language"]["enum"]) == 184


def test_valid_gold_and_business_rule():
    ok, errors = validate_instance(_gold())
    assert ok, errors
    ok, errors = validate_instance(
        _gold(category="bug", refund_amount_usd=None, order_ref="ORD-100001")
    )
    assert ok, errors
    ok, errors = validate_instance(_gold(category="bug", refund_amount_usd=12.5))
    assert not ok
    ok, errors = validate_instance(_gold(order_ref="104233"))
    assert not ok
    ok, errors = validate_instance(_gold(language="zz"))
    assert not ok
    ok, errors = validate_instance(_gold(severity="2"))
    assert not ok


def test_null_order_ref_is_allowed():
    ok, errors = validate_instance(_gold(category="access", order_ref=None, refund_amount_usd=None))
    assert ok, errors


def test_generated_labels_match_the_contract():
    for record in generate_tickets(16, seed=7):
        ok, errors = validate_instance(record.gold)
        assert ok, errors
        assert record.gold["ticket_id"] in record.text
        assert record.gold["customer_tier"] in record.text


def test_schema_file_is_json():
    encoded = json.dumps(load_schema())
    assert "support_ticket" in encoded

import json

from schemasmith.contract.validator import extract_json_object, score_outputs


def _gold(language="en", category="billing", refund=49.0, order="ORD-104233"):
    return {
        "ticket_id": "T-100001",
        "customer_tier": "pro",
        "category": category,
        "severity": 2,
        "order_ref": order,
        "refund_amount_usd": refund,
        "language": language,
    }


def test_extracts_fenced_and_embedded_json():
    payload = {"ticket_id": "T-1"}
    fenced = "```json\n" + json.dumps(payload) + "\n```"
    found, err = extract_json_object(fenced)
    assert err is None
    assert found == payload
    found, err = extract_json_object("Here it is:\n" + json.dumps(payload) + "\nThanks")
    assert found == payload


def test_scorecard_rates_are_computed_from_the_outputs():
    source = (
        "Ticket: T-100001\n"
        "the pro plan was charged twice for order ORD-104233. "
        "Please refund 49.00 USD."
    )
    gold = _gold()
    wrong = dict(gold)
    wrong["refund_amount_usd"] = 98
    wrong["category"] = "billing"
    outputs = [
        json.dumps(gold),
        "not json at all",
        "```json\n" + json.dumps(wrong) + "\n```",
    ]
    card = score_outputs(outputs, [gold, gold, gold], [source, source, source], ["en", "en", "es"])
    assert card.n == 3
    assert card.parse_rate == 2 / 3
    assert card.schema_valid_rate == 2 / 3
    assert card.contract_valid_rate == card.schema_valid_rate
    assert card.rows[2].unsupported_fields == ["refund_amount_usd"]
    assert card.unsupported_field_count == 1
    # Two parsed rows each contribute ticket_id, order_ref, and refund.
    assert card.grounded_field_count == 6
    assert card.unsupported_value_rate == 1 / 6
    assert card.rows[1].field_hits["ticket_id"] is False
    assert card.field_exact_match < 1
    assert card.n_non_english == 1
    assert card.non_english_contract_valid_rate == 1.0


def test_invented_order_is_unsupported_even_when_schema_fails():
    source = "Ticket: T-100001 on the pro plan. No order number is in this note."
    gold = _gold(category="access", refund=None, order=None)
    predicted = dict(gold)
    predicted["order_ref"] = "ORD-999999"
    card = score_outputs(
        [json.dumps(predicted)],
        [gold],
        [source],
        ["en"],
    )
    assert card.rows[0].schema_valid is True
    assert card.rows[0].unsupported_fields == ["order_ref"]

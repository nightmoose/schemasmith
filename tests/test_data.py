from schemasmith.contract.validator import amount_in_text, grounded_field_supported
from schemasmith.data.generator import generate_tickets, with_defects
from schemasmith.data.loader import load_ticket_jsonl, write_jsonl
from schemasmith.data.templates import LANGUAGES


def test_generator_is_deterministic_and_multilingual():
    first = generate_tickets(16, seed=42)
    second = generate_tickets(16, seed=42)
    assert [row.to_dict() for row in first] == [row.to_dict() for row in second]
    assert generate_tickets(8, seed=42)[0].gold != generate_tickets(8, seed=43)[0].gold
    languages = {row.language for row in first}
    assert languages == set(LANGUAGES)
    assert len(languages) >= 6
    ids = [row.ticket_id for row in first]
    assert len(ids) == len(set(ids))


def test_money_and_ids_are_in_the_text():
    for record in generate_tickets(16, seed=3):
        assert grounded_field_supported("ticket_id", record.gold["ticket_id"], record.text)
        order = record.gold["order_ref"]
        if order is not None:
            assert order in record.text
        refund = record.gold["refund_amount_usd"]
        if refund is not None:
            assert amount_in_text(refund, record.text)
        else:
            assert record.gold["category"] != "billing"


def test_roundtrip_jsonl(tmp_path):
    records = generate_tickets(4, seed=1)
    path = tmp_path / "tickets.jsonl"
    write_jsonl(path, [row.to_dict() for row in records])
    loaded = load_ticket_jsonl(path)
    assert [row.to_dict() for row in loaded] == [row.to_dict() for row in records]


def test_defects_are_labeled():
    rows = with_defects(generate_tickets(8, seed=1))
    assert len(rows) == 11
    assert {row.defect for row in rows if row.defect} == {
        "exact_duplicate",
        "refund_on_non_billing",
        "order_ref_pattern",
    }

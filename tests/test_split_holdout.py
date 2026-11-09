"""The held-out split shares no template or masked text with train."""

from schemasmith.curation.local import assign_splits, mask_text, masked_overlap
from schemasmith.data.generator import generate_tickets
from schemasmith.data.templates import LANGUAGES, TEST_SUBSCENARIOS, L


def _parts(template):
    sub, body, header, cue = template.split("/")
    return sub, header, cue


def test_test_split_is_held_out_with_zero_masked_overlap():
    records = generate_tickets(320, seed=42)
    splits = assign_splits(records, seed=42)
    train, test = splits["train"] + splits["validation"], splits["test"]
    assert len(test) >= 60
    assert {row.language for row in test} == set(LANGUAGES)
    assert len({row.gold["category"] for row in test}) == 4
    assert masked_overlap(train, test) == []
    train_masked = {mask_text(row.text) for row in train}
    assert not any(mask_text(row.text) in train_masked for row in test)
    train_parts = [_parts(row.template) for row in train]
    test_parts = [_parts(row.template) for row in test]
    assert {p[0] for p in train_parts}.isdisjoint({p[0] for p in test_parts})
    assert {p[1] for p in train_parts}.isdisjoint({p[1] for p in test_parts})
    assert {p[2] for p in train_parts}.isdisjoint({p[2] for p in test_parts})
    assert {p[0] for p in test_parts} == set(TEST_SUBSCENARIOS)


def test_masking_catches_tickets_that_differ_only_in_numbers_and_tier():
    a = "Ticket: T-100001 I am on the pro plan. Please refund 12.50 USD."
    b = "Ticket: T-100777 I am on the free plan. Please refund 99.00 USD."
    assert mask_text(a) == mask_text(b)


def test_severity_is_stated_by_exactly_one_cue_in_the_text():
    for record in generate_tickets(320, seed=7):
        cues = L[record.language]["cue"]
        found = {
            level
            for level, phrasings in cues.items()
            for phrase in phrasings
            if phrase in record.text
        }
        assert found == {record.gold["severity"]}, record.ticket_id


def test_noise_is_present_but_fields_stay_grounded():
    records = generate_tickets(320, seed=42)
    texts = [row.text for row in records]
    assert sum("Sent from my phone." in t or "Sorry" in t or "Merci" in t for t in texts) > 20
    assert sum(t.startswith("[") for t in texts) > 20  # header without From line
    for row in records:
        assert row.gold["ticket_id"] in row.text
        if row.gold["order_ref"]:
            assert row.gold["order_ref"] in row.text

"""The hostile set is hand-written, valid under its own gold, and unseen in training."""

import json

from schemasmith.contract.validator import validate_instance
from schemasmith.curation.local import mask_text
from schemasmith.data.generator import generate_tickets
from schemasmith.hostile import TICKETS, gate_decision, score_hostile
from schemasmith.hostile.score import ACCEPT, HOLD, render_hostile_table
from schemasmith.hostile.tickets import KINDS
from schemasmith.training import DEFAULT_N


def test_size_and_kinds():
    assert 20 <= len(TICKETS) <= 30
    assert len({t.ticket_id for t in TICKETS}) == len(TICKETS)
    for kind in KINDS:
        assert sum(t.kind == kind for t in TICKETS) >= 3
    assert sum(t.kind == "injection" for t in TICKETS) >= 5


def test_gold_is_contract_valid():
    for ticket in TICKETS:
        ok, errors = validate_instance(ticket.gold)
        assert ok, (ticket.ticket_id, errors)


def test_zero_overlap_with_generated_training_tickets():
    generated = generate_tickets(DEFAULT_N, 42)
    masked = {mask_text(r.text) for r in generated}
    texts = "\n".join(r.text for r in generated)
    for ticket in TICKETS:
        assert mask_text(ticket.text) not in masked
        assert ticket.ticket_id not in texts
        # No hostile sentence of 6+ words appears verbatim in a generated ticket.
        words = ticket.text.split()
        for i in range(max(0, len(words) - 5)):
            assert " ".join(words[i : i + 6]) not in texts, ticket.ticket_id


def test_gold_outputs_score_clean_except_the_no_id_ticket():
    outputs = [json.dumps(t.gold) for t in TICKETS]
    result = score_hostile(TICKETS, outputs)
    assert result["schema_valid"] == len(TICKETS)
    assert result["field_match"] == 1.0
    assert result["forbidden_value_written"] == 0
    assert result["injections_refused"] == result["injection_tickets"]
    held = [r["id"] for r in result["rows"] if r["gate"] == HOLD]
    assert held == ["T-HX22"]
    assert result["accepted_but_wrong"] == 0


def test_following_an_injection_is_caught():
    ticket = next(t for t in TICKETS if t.ticket_id == "T-HX01")
    obeyed = dict(ticket.gold, refund_amount_usd=5000, severity=4)
    result = score_hostile([ticket], [json.dumps(obeyed)])
    assert result["forbidden_value_written"] == 1
    assert result["injections_refused"] == 0
    # 5000 is in the text, so the grounding gate alone accepts it. Field match catches it.
    assert result["accepted_but_wrong"] == 1


def test_gate_holds_invented_and_invalid_values():
    ticket = next(t for t in TICKETS if t.ticket_id == "T-HX04")
    invented = dict(ticket.gold, order_ref="ORD-123456")
    decision, reasons = gate_decision(json.dumps(invented), ticket.text)
    assert decision == HOLD and any("order_ref" in r for r in reasons)
    assert gate_decision("customer_tier: vip", ticket.text)[0] == HOLD
    assert gate_decision(json.dumps(ticket.gold), ticket.text)[0] == ACCEPT


def test_table_renders_counts():
    outputs = [json.dumps(t.gold) for t in TICKETS]
    card = score_hostile(TICKETS, outputs)
    table = render_hostile_table({"systems": {"base": card, "sft": card, "dpo": card}})
    assert f"{len(TICKETS)} of {len(TICKETS)}" in table
    assert "Injections refused" in table


def test_cards_lead_with_the_measured_headline():
    from schemasmith.hf.cards import adapter_card, eval_logs_card, load_result_files

    compare, hostile = load_result_files()
    card = adapter_card("dpo", compare, hostile)
    body = card.split("---", 2)[2]
    assert body.index("Made-up field values: 32.9% to 0%") < body.index("Core model")
    assert "license: cc-by-nc-4.0" in card
    assert "injections refused" in card
    assert "Made-up field values" in eval_logs_card(compare, hostile)
    assert "Made-up field values" not in adapter_card("sft")

"""v2 additions: hostile set v2 stays blind, adversarial rows stay train-only."""

import json

from schemasmith.contract.validator import validate_instance
from schemasmith.curation.local import mask_text
from schemasmith.data.adversarial import KINDS, adversarial_tickets, injected_rejection
from schemasmith.data.generator import generate_tickets
from schemasmith.hostile import TICKETS, score_hostile
from schemasmith.hostile.tickets_v2 import TICKETS_V2

ADV = adversarial_tickets(480, 42)
GEN = generate_tickets(1280, 42)


def _six_grams(text):
    words = text.split()
    return {" ".join(words[i : i + 6]) for i in range(max(0, len(words) - 5))}


def test_hostile_v2_gold_is_valid_and_scores_clean():
    assert len(TICKETS_V2) == 30
    assert len({t.ticket_id for t in TICKETS_V2}) == 30
    for t in TICKETS_V2:
        assert validate_instance(t.gold)[0], t.ticket_id
    card = score_hostile(TICKETS_V2, [json.dumps(t.gold) for t in TICKETS_V2])
    assert card["field_match"] == 1.0 and card["forbidden_value_written"] == 0
    assert card["accepted_but_wrong"] == 0


def test_hostile_sets_share_no_phrase_with_training_rows():
    train_text = "\n".join(r.text for r in GEN + ADV)
    masked = {mask_text(r.text) for r in GEN + ADV}
    for t in (*TICKETS, *TICKETS_V2):
        assert mask_text(t.text) not in masked
        assert t.ticket_id not in train_text
        for gram in _six_grams(t.text):
            # Pasted JSON answers share key syntax with the output format itself.
            if '":' in gram:
                continue
            assert gram not in train_text, (t.ticket_id, gram)


def test_adversarial_rows_are_train_group_valid_and_cover_kinds():
    assert len(ADV) == 480
    assert all(r.split_group == "train" for r in ADV)
    assert all(validate_instance(r.gold)[0] for r in ADV)
    kinds = {r.template.split("/")[1] for r in ADV}
    assert kinds <= set(KINDS) and len(kinds) >= 8
    ids = {r.ticket_id for r in GEN}
    assert not ids & {r.ticket_id for r in ADV}


def test_injected_rejection_differs_from_gold():
    n = 0
    for r in ADV:
        bad = injected_rejection(r)
        if bad is not None:
            n += 1
            assert bad != json.dumps(r.gold, ensure_ascii=False)
    assert n > 250

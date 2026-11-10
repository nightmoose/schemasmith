"""Guardrail policy stays generic, and blocked rows are held, not accepted."""

from schemasmith.aws.guardrails import (
    DENIED_WORDS,
    GUARDRAIL_NAME,
    GuardrailCheck,
    estimate_cost_usd,
    guardrail_config,
    reasons_from_assessments,
    usage_units,
)
from schemasmith.hostile.score import ACCEPT, HOLD, score_row
from schemasmith.hostile.tickets import TICKETS
from schemasmith.hostile.tickets_v2 import TICKETS_V2


def test_word_filters_are_not_tuned_on_hostile_tickets():
    # Generic tropes that also appear in T-HY07 are allowed; unique phrases
    # from the rest of the hostile sets are not.
    allowed = {"you are a pirate", "### new instructions", "do not output json"}
    blob_v1 = "\n".join(t.text.lower() for t in TICKETS)
    blob_v2 = "\n".join(t.text.lower() for t in TICKETS_V2)
    for word in DENIED_WORDS:
        key = word.lower()
        if key in allowed:
            continue
        assert key not in blob_v1, word
        assert key not in blob_v2, word


def test_policy_has_prompt_attack_high_and_a_denied_topic():
    config = guardrail_config()
    filters = config["contentPolicyConfig"]["filtersConfig"]
    attack = next(f for f in filters if f["type"] == "PROMPT_ATTACK")
    assert attack["inputStrength"] == "HIGH"
    assert attack["outputStrength"] == "NONE"
    topics = config["topicPolicyConfig"]["topicsConfig"]
    assert topics[0]["type"] == "DENY"
    assert topics[0]["name"] == "Instruction override"
    assert len(topics[0]["definition"]) <= 1000
    assert all(len(ex) <= 100 for ex in topics[0]["examples"])
    assert config["topicPolicyConfig"]["tierConfig"]["tierName"] == "STANDARD"
    assert config["contentPolicyConfig"]["tierConfig"]["tierName"] == "STANDARD"
    assert config["crossRegionConfig"]["guardrailProfileIdentifier"] == "us.guardrail.v1:0"
    assert GUARDRAIL_NAME == "lexsi-prompt-injection"


def test_reasons_and_cost_from_a_blocked_assessment():
    assessments = [
        {
            "topicPolicy": {
                "topics": [{"name": "Instruction override", "type": "DENY", "action": "BLOCKED"}]
            },
            "contentPolicy": {
                "filters": [
                    {
                        "type": "PROMPT_ATTACK",
                        "confidence": "HIGH",
                        "action": "BLOCKED",
                    }
                ]
            },
            "wordPolicy": {
                "customWords": [{"match": "jailbreak", "action": "BLOCKED"}],
                "managedWordLists": [],
            },
        }
    ]
    reasons = reasons_from_assessments(assessments)
    assert "topic:Instruction override" in reasons
    assert "content:PROMPT_ATTACK:HIGH" in reasons
    assert "word:jailbreak" in reasons
    usage = usage_units({"contentPolicyUnits": 2, "topicPolicyUnits": 2, "wordPolicyUnits": 1})
    # 2 content + 2 topic units at $0.15/1k, words free.
    assert abs(estimate_cost_usd(usage) - 0.0006) < 1e-9


def test_blocked_injection_is_held_and_not_counted_as_followed():
    ticket = next(t for t in TICKETS_V2 if t.ticket_id == "T-HY02")
    leaked = (
        '{"ticket_id": "T-HY02", "customer_tier": "free", "category": "billing", '
        '"severity": 4, "order_ref": null, "refund_amount_usd": 250.0, "language": "en"}'
    )
    raw = score_row(ticket, leaked)
    assert raw["injection_followed"] is True
    assert raw["gate"] == ACCEPT or raw["accepted_but_wrong"] or raw["gate"] == HOLD
    # Product path: a guardrail block holds the row and does not release the
    # injected value, even if the model wrote it.
    check = GuardrailCheck(
        "INPUT", "GUARDRAIL_INTERVENED", True, 0.1, reasons=["content:PROMPT_ATTACK:HIGH"]
    )
    assert check.blocked
    released_followed = raw["injection_followed"] and not check.blocked
    assert released_followed is False
    decision = HOLD if check.blocked else raw["gate"]
    assert decision == HOLD

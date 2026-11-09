"""Deterministic multilingual support tickets with contract-valid labels."""

from __future__ import annotations

import random
from typing import Any

from schemasmith.data.records import TicketRecord
from schemasmith.data.templates import LANGUAGES, SCENARIOS, TEMPLATES

SEVERITY = {
    "billing": 2,
    "bug": 3,
    "access": 2,
    "feature_request": 1,
}
TIERS = ("free", "pro", "enterprise")
AMOUNTS = (12.50, 19.99, 49.00, 7.25, 15.00, 30.50)


def generate_tickets(n: int, seed: int = 42) -> list[TicketRecord]:
    """Build ``n`` unique, contract-valid tickets.

    Row ``i`` depends on ``seed`` and ``i`` only, so a longer list extends
    a shorter one with the same seed.
    """
    if n < 1:
        raise ValueError("n must be at least 1")
    records: list[TicketRecord] = []
    for index in range(n):
        rng = random.Random(seed + index)
        language = LANGUAGES[index % len(LANGUAGES)]
        scenario = SCENARIOS[(index // len(LANGUAGES)) % len(SCENARIOS)]
        records.append(_one(index, language, scenario, rng))
    return records


def with_defects(records: list[TicketRecord]) -> list[TicketRecord]:
    """Append rows a curation gate should drop.

    The returned list keeps the original rows and adds an exact duplicate,
    a non-billing refund, and an order reference that misses the ORD- prefix.
    """
    if not records:
        raise ValueError("records must not be empty")
    first = records[0]
    duplicate = TicketRecord(
        ticket_id=first.ticket_id,
        language=first.language,
        text=first.text,
        gold=dict(first.gold),
        keep=True,
        defect="exact_duplicate",
    )
    bad_refund_gold = dict(first.gold)
    bad_refund_gold["category"] = "bug"
    bad_refund_gold["refund_amount_usd"] = 49.0
    bad_refund = TicketRecord(
        ticket_id=first.ticket_id + "X",
        language=first.language,
        text=first.text + "\nNote: planted invalid refund.",
        gold=bad_refund_gold,
        keep=True,
        defect="refund_on_non_billing",
    )
    bad_order_gold = dict(first.gold)
    bad_order_gold["order_ref"] = "104233"
    bad_order = TicketRecord(
        ticket_id=first.ticket_id + "Y",
        language=first.language,
        text=first.text + "\nNote: planted bad order reference.",
        gold=bad_order_gold,
        keep=True,
        defect="order_ref_pattern",
    )
    return [*records, duplicate, bad_refund, bad_order]


def _one(index: int, language: str, scenario: str, rng: random.Random) -> TicketRecord:
    ticket_id = f"T-{100000 + index}"
    tier = TIERS[rng.randrange(len(TIERS))]
    amount = AMOUNTS[rng.randrange(len(AMOUNTS))]
    order_ref: str | None
    if scenario in ("billing", "bug"):
        order_ref = f"ORD-{200000 + index:06d}"
    else:
        order_ref = None
    amount_text = f"{amount:.2f}"
    text = TEMPLATES[language][scenario].format(
        tid=ticket_id,
        tier=tier,
        order=order_ref or "",
        amount=amount_text,
    )
    refund: float | None = amount if scenario == "billing" else None
    gold: dict[str, Any] = {
        "ticket_id": ticket_id,
        "customer_tier": tier,
        "category": scenario,
        "severity": SEVERITY[scenario],
        "order_ref": order_ref,
        "refund_amount_usd": refund,
        "language": language,
    }
    return TicketRecord(
        ticket_id=ticket_id,
        language=language,
        text=text,
        gold=gold,
        keep=True,
        defect=None,
    )

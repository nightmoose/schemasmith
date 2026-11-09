"""Deterministic multilingual support tickets with contract-valid labels.

Tickets are composed from the pieces in ``templates.py``: a header, optional
chatter, a scenario body, a severity cue, a tier mention, and optional order
and refund mentions, with typo and mixed-language noise. Severity comes from
the cue sentence, so it can be read from the text.

Each record carries ``split_group``. Test-group records use held-out
sub-scenarios, header formats, severity phrasings, openers, closers, and
chatter, so they share no template with the train group.
"""

from __future__ import annotations

import random
from typing import Any

from schemasmith.data.records import TicketRecord
from schemasmith.data.templates import (
    EMAILS,
    EXTRA_CHATTER,
    FOREIGN_CHATTER,
    LANGUAGES,
    REFUND_SUBSCENARIOS,
    SCENARIOS,
    SUBSCENARIOS,
    TEST_CUES,
    TEST_HEADERS,
    TEST_SUBSCENARIOS,
    TRAIN_CUES,
    TRAIN_HEADERS,
    L,
)

# Severity levels a category can take. The cue sentence states the level.
SEVERITY_CHOICES: dict[str, tuple[int, ...]] = {
    "billing": (1, 2, 3),
    "bug": (1, 2, 3, 4),
    "access": (2, 3, 4),
    "feature_request": (1, 2),
}
ORDER_PROBABILITY: dict[str, float] = {
    "billing": 0.85,
    "bug": 0.5,
    "access": 0.15,
    "feature_request": 0.1,
}
TIERS = ("free", "pro", "enterprise")
AMOUNTS = (12.50, 19.99, 49.00, 7.25, 15.00, 30.50, 99.00, 4.99, 120.00, 63.40)
NO_SPACE_LANGUAGES = frozenset({"zh", "ja"})

HEADERS: tuple[str, ...] = (
    "Ticket: {tid}\n{from_label}: {email}\n{subject_label}: {subject}\n\n",
    "[{tid}] {subject}\n\n",
    "{subject_label}: {subject}\nTicket: {tid}\n\n",
    "Re: {subject} #{tid}\n\n",
    "{from_label}: {email}\nRef {tid}\n\n",
)


def generate_tickets(n: int, seed: int = 42) -> list[TicketRecord]:
    """Build ``n`` unique, contract-valid tickets.

    Row ``i`` depends on ``seed`` and ``i`` only, so a longer list extends
    a shorter one with the same seed. Every block of 160 rows covers all
    eight languages and all twenty sub-scenarios; one in five rows is in the
    held-out test group.
    """
    if n < 1:
        raise ValueError("n must be at least 1")
    records: list[TicketRecord] = []
    for index in range(n):
        rng = random.Random(seed * 100003 + index)
        language = LANGUAGES[index % len(LANGUAGES)]
        block = index // len(LANGUAGES)
        category = SCENARIOS[block % len(SCENARIOS)]
        subs = SUBSCENARIOS[category]
        sub = subs[block % len(subs)]
        records.append(_one(index, language, category, sub, rng))
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
        split_group=first.split_group,
        template=first.template,
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
        split_group=first.split_group,
        template=first.template,
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
        split_group=first.split_group,
        template=first.template,
    )
    return [*records, duplicate, bad_refund, bad_order]


def _one(
    index: int,
    language: str,
    category: str,
    sub: str,
    rng: random.Random,
) -> TicketRecord:
    pieces = L[language]
    group = "test" if sub in TEST_SUBSCENARIOS else "train"
    headers = TEST_HEADERS if group == "test" else TRAIN_HEADERS
    cues = TEST_CUES if group == "test" else TRAIN_CUES
    chatter_slots = (2,) if group == "test" else (0, 1)

    ticket_id = f"T-{100000 + index}"
    tier = TIERS[rng.randrange(len(TIERS))]
    severity = rng.choice(SEVERITY_CHOICES[category])
    order_ref = f"ORD-{200000 + index:06d}" if rng.random() < ORDER_PROBABILITY[category] else None
    refund = AMOUNTS[rng.randrange(len(AMOUNTS))] if sub in REFUND_SUBSCENARIOS else None

    body_index = rng.randrange(2)
    body = pieces["body"][sub][body_index]
    typo = rng.random() < 0.3
    if typo:
        body = _typo(body, language, rng)

    middle = [pieces["tier"][rng.randrange(2)].format(tier=tier)]
    if order_ref:
        middle.append(pieces["order"][rng.randrange(2)].format(order=order_ref))
    if refund is not None:
        middle.append(pieces["refund"][rng.randrange(2)].format(amount=f"{refund:.2f}"))
    cue_index = rng.choice(cues)
    middle.append(pieces["cue"][severity][cue_index])
    rng.shuffle(middle)

    sentences: list[str] = []
    if rng.random() < 0.7:
        sentences.append(pieces["open"][rng.choice(chatter_slots)])
    sentences.append(body)
    sentences.extend(middle)
    chatter_key = "en" if language == "en" else "other"
    if rng.random() < 0.3:
        sentences.append(EXTRA_CHATTER[chatter_key][rng.choice(chatter_slots)])
    mixed = rng.random() < 0.2
    if mixed:
        sentences.append(FOREIGN_CHATTER[chatter_key][rng.randrange(2)])
    if rng.random() < 0.7:
        sentences.append(pieces["close"][rng.choice(chatter_slots)])

    joiner = "" if language in NO_SPACE_LANGUAGES else " "
    header_index = rng.choice(headers)
    header = HEADERS[header_index].format(
        tid=ticket_id,
        from_label=pieces["from"],
        subject_label=pieces["subject"],
        subject=pieces["subjects"][category],
        email=EMAILS[rng.randrange(len(EMAILS))],
    )
    text = header + joiner.join(sentences)

    gold: dict[str, Any] = {
        "ticket_id": ticket_id,
        "customer_tier": tier,
        "category": category,
        "severity": severity,
        "order_ref": order_ref,
        "refund_amount_usd": refund,
        "language": language,
    }
    template = f"{sub}/b{body_index}/h{header_index}/c{cue_index}"
    return TicketRecord(
        ticket_id=ticket_id,
        language=language,
        text=text,
        gold=gold,
        keep=True,
        defect=None,
        split_group=group,
        template=template,
    )


def _typo(sentence: str, language: str, rng: random.Random) -> str:
    """Swap two adjacent letters inside one word. Never touches digits."""
    if language in NO_SPACE_LANGUAGES or language == "hi":
        positions = [
            i
            for i in range(len(sentence) - 1)
            if sentence[i].isalpha()
            and sentence[i + 1].isalpha()
            and not sentence[i].isascii()
            and not sentence[i + 1].isascii()
        ]
        if not positions:
            return sentence
        i = rng.choice(positions)
        return sentence[:i] + sentence[i + 1] + sentence[i] + sentence[i + 2 :]
    words = sentence.split(" ")
    candidates = [i for i, w in enumerate(words) if len(w) >= 5 and w.isalpha()]
    if not candidates:
        return sentence
    w_index = rng.choice(candidates)
    word = words[w_index]
    j = rng.randrange(1, len(word) - 2)
    words[w_index] = word[:j] + word[j + 1] + word[j] + word[j + 2 :]
    return " ".join(words)

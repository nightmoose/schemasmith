"""Instruction text shared by the alpaca export and the messages export."""

INSTRUCTION = (
    "Extract one support ticket as a single JSON object that matches the "
    "support_ticket data contract. Copy ticket_id, order_ref, and "
    "refund_amount_usd from the text. customer_tier is one of free, pro, "
    "enterprise. category is one of billing, bug, access, feature_request. "
    "severity is an integer from 1 to 4: 4 when production or the whole "
    "team is down, 3 when the writer is blocked, 2 when there is a "
    "workaround, 1 when nothing is blocked and there is no rush. "
    "order_ref is ORD- followed by six "
    "digits, or null. refund_amount_usd is a non-negative number only when "
    "category is billing, otherwise null. language is the ISO 639-1 code of "
    "the ticket. Return one JSON object and no other text."
)


def user_content(ticket_text: str) -> str:
    """User turn used by the chat-messages export.

    CuratorKIT's messages exporter joins instruction and input with a blank
    line. Matching that join keeps the two exports interchangeable.
    """
    return f"{INSTRUCTION}\n\n{ticket_text}"

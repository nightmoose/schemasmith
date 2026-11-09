"""Synthetic tickets and the JSONL loaders that read them."""

from schemasmith.data.generator import generate_tickets
from schemasmith.data.loader import (
    load_alpaca_jsonl,
    load_jsonl,
    load_ticket_jsonl,
    write_jsonl,
)
from schemasmith.data.records import TicketRecord

__all__ = [
    "TicketRecord",
    "generate_tickets",
    "load_alpaca_jsonl",
    "load_jsonl",
    "load_ticket_jsonl",
    "write_jsonl",
]

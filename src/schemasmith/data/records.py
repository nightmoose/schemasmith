"""One labeled support ticket."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class TicketRecord:
    ticket_id: str
    language: str
    text: str
    gold: dict[str, Any]
    keep: bool = True
    defect: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "ticket_id": self.ticket_id,
            "language": self.language,
            "text": self.text,
            "gold": self.gold,
            "keep": self.keep,
            "defect": self.defect,
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> TicketRecord:
        return cls(
            ticket_id=str(payload["ticket_id"]),
            language=str(payload["language"]),
            text=str(payload["text"]),
            gold=dict(payload["gold"]),
            keep=bool(payload.get("keep", True)),
            defect=payload.get("defect"),
        )

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
    # "train" or "test" when the generator holds templates out of test.
    split_group: str | None = None
    # Template key, e.g. "double_charge/b0/h1/c0". Used by the overlap checks.
    template: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "ticket_id": self.ticket_id,
            "language": self.language,
            "text": self.text,
            "gold": self.gold,
            "keep": self.keep,
            "defect": self.defect,
            "split_group": self.split_group,
            "template": self.template,
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
            split_group=payload.get("split_group"),
            template=payload.get("template"),
        )

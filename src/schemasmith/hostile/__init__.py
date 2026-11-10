"""Hostile tickets and their scoring. See ``tickets.py`` and ``score.py``."""

from schemasmith.hostile.score import gate_decision, score_hostile
from schemasmith.hostile.tickets import TICKETS, HostileTicket

__all__ = ["TICKETS", "HostileTicket", "gate_decision", "score_hostile"]

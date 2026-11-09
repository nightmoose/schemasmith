"""Held-out scoring. Model weights are not loaded here."""

from schemasmith.eval.auditkit_bridge import auditkit_available, run_auditkit
from schemasmith.eval.harness import evaluate_records

__all__ = ["auditkit_available", "evaluate_records", "run_auditkit"]

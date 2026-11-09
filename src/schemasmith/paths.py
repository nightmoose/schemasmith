"""Locate package data files."""

from pathlib import Path


def package_dir() -> Path:
    return Path(__file__).resolve().parent


def contract_dir() -> Path:
    """Directory that holds the ODCS YAML and the JSON Schema."""
    return package_dir() / "contracts"


def schema_path() -> Path:
    return contract_dir() / "support_ticket.schema.json"


def odcs_path() -> Path:
    return contract_dir() / "support_ticket.odcs.yaml"

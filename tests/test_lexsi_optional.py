"""Runs only when the external Lexsi packages are installed."""

import json

import pytest

from schemasmith.contract.validator import validate_instance
from schemasmith.curation.curatorkit import curate_with_curatorkit
from schemasmith.data.generator import generate_tickets, with_defects
from schemasmith.data.loader import load_alpaca_jsonl
from schemasmith.eval.auditkit_bridge import run_auditkit


def test_curatorkit_gate_and_export(tmp_path):
    pytest.importorskip("curatorkit")
    records = with_defects(generate_tickets(24, seed=42))
    summary = curate_with_curatorkit(records, tmp_path / "run", seed=42)
    assert summary["rejected_count"] >= 2
    curated = tmp_path / "run" / "curated"
    kept = []
    for split in ("train", "validation", "test"):
        path = curated / split / "sft_alpaca.jsonl"
        assert path.exists()
        rows = load_alpaca_jsonl(path)
        kept.extend(rows)
        messages = (curated / split / "sft_messages.jsonl").read_text(encoding="utf-8")
        assert '"role": "user"' in messages or '"role":"user"' in messages
    assert len(kept) == 24
    for row in kept:
        ok, errors = validate_instance(row.gold)
        assert ok, errors


def test_auditkit_scores_gold_json():
    pytest.importorskip("auditkit")
    records = generate_tickets(4, seed=42)
    result = run_auditkit(records, [json.dumps(row.gold) for row in records])
    assert result is not None
    assert result["headline"]["is_json"] == 1
    assert result["headline"]["json_schema_valid"] == 1
    assert result["headline"]["field_f1"] == 1

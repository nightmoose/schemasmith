import json

from schemasmith.contract.validator import validate_instance
from schemasmith.curation.local import alpaca_row, curate_local, messages_row
from schemasmith.data.generator import generate_tickets, with_defects
from schemasmith.data.loader import load_alpaca_jsonl
from schemasmith.prompts import INSTRUCTION


def test_local_gate_drops_defects_and_writes_splits(tmp_path):
    raw = with_defects(generate_tickets(16, seed=5))
    result = curate_local(raw, tmp_path / "local", seed=5)
    assert result.input_count == 19
    assert len(result.kept) == 16
    reasons = " ".join(item["reason"] for item in result.rejected)
    assert "exact_duplicate" in reasons
    assert any("refund" in item["reason"] or "null" in item["reason"] for item in result.rejected)
    counts = result.summary()["split_counts"]
    assert counts["train"] + counts["validation"] + counts["test"] == 16
    assert counts["validation"] >= 1 and counts["test"] >= 1
    ids = []
    for name in ("train", "validation", "test"):
        loaded = load_alpaca_jsonl(tmp_path / "local" / name / "sft_alpaca.jsonl")
        ids.extend(row.ticket_id for row in loaded)
        for row in loaded:
            ok, errors = validate_instance(row.gold)
            assert ok, errors
    assert len(ids) == len(set(ids)) == 16
    again = curate_local(raw, tmp_path / "local2", seed=5)
    assert [row.ticket_id for row in again.splits["test"]] == [
        row.ticket_id for row in result.splits["test"]
    ]


def test_exports_match_the_instruction(tmp_path):
    record = generate_tickets(1, seed=2)[0]
    alpaca = alpaca_row(record)
    assert alpaca["instruction"] == INSTRUCTION
    assert json.loads(alpaca["output"])["ticket_id"] == record.ticket_id
    message = messages_row(record)
    assert message["messages"][0]["content"].startswith(INSTRUCTION)
    assert record.text in message["messages"][0]["content"]
    curate_local([record], tmp_path / "one", seed=1)
    manifest = json.loads((tmp_path / "one" / "manifest.json").read_text())
    assert manifest["producer"] == "schemasmith"
    assert manifest["split_counts"]["train"] == 1

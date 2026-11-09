import json

from schemasmith.cli import main
from schemasmith.data.generator import generate_tickets


def test_generate_curate_and_eval(tmp_path):
    raw = tmp_path / "raw.jsonl"
    assert main(["generate", "--n", "8", "--seed", "4", "--out", str(raw)]) == 0
    curated = tmp_path / "curated"
    assert main(["curate", "--raw", str(raw), "--out", str(curated), "--seed", "4"]) == 0
    test_path = curated / "local" / "test" / "sft_alpaca.jsonl"
    records = generate_tickets(8, seed=4)
    # Predictions cover every id, including rows that landed in train.
    preds = tmp_path / "preds.jsonl"
    preds.write_text(
        "".join(
            json.dumps({"id": row.ticket_id, "output": json.dumps(row.gold)}) + "\n"
            for row in records
        ),
        encoding="utf-8",
    )
    report = tmp_path / "report.json"
    # Score the train split so the file is non-empty even if test is small.
    train = curated / "local" / "train" / "sft_alpaca.jsonl"
    assert (
        main(
            [
                "eval",
                "--data",
                str(train),
                "--predictions",
                str(preds),
                "--report",
                str(report),
            ]
        )
        == 0
    )
    payload = json.loads(report.read_text())
    assert payload["scorecard"]["schema_valid_rate"] == 1
    assert "not a base-model run" in payload["measurement"]
    assert test_path.exists()

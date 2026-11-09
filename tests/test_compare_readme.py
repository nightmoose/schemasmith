import json
from pathlib import Path

import pytest

from schemasmith.contract.validator import Scorecard
from schemasmith.eval.compare import (
    END,
    START,
    load_results,
    render_metrics_table,
    results_from_scorecards,
    update_readme,
    validate_results,
)
from schemasmith.eval.generate import require_cuda_backend
from schemasmith.training import MODEL_ID


def _card(rate: float, n: int = 4) -> Scorecard:
    return Scorecard(
        n=n,
        n_parsed=n,
        n_schema_valid=n,
        n_contract_valid=n,
        parse_rate=rate,
        schema_valid_rate=rate,
        contract_valid_rate=rate,
        field_exact_match=rate,
        per_field_accuracy={},
        unsupported_value_rate=None,
        unsupported_field_count=0,
        grounded_field_count=0,
        non_english_contract_valid_rate=rate,
        n_non_english=n,
    )


def _payload():
    return results_from_scorecards(
        {"base": _card(0.25), "sft": _card(0.5), "dpo": _card(0.75)},
        data_path="heldout.jsonl",
        sft_adapter="sft",
        dpo_adapter="dpo",
    )


def test_table_uses_only_the_file_values():
    table = render_metrics_table(_payload())
    assert "25.0%" in table
    assert "50.0%" in table
    assert "75.0%" in table
    assert "n/a" in table
    assert "TBD" not in table
    assert "measured" in table


def test_update_readme_touches_only_the_marked_block(tmp_path):
    readme = tmp_path / "README.md"
    readme.write_text(
        "intro\n\n" + START + "\nTBD\n" + END + "\n\n## Licenses\nkeep this\n",
        encoding="utf-8",
    )
    update_readme(readme, _payload())
    text = readme.read_text(encoding="utf-8")
    assert text.startswith("intro\n\n")
    assert text.endswith("\n\n## Licenses\nkeep this\n")
    assert "25.0%" in text
    assert "TBD" not in text


def test_rejects_a_file_that_is_not_a_compare(tmp_path):
    path = tmp_path / "nope.json"
    path.write_text(json.dumps({"measurement": "echo", "model_id": MODEL_ID}) + "\n")
    with pytest.raises(ValueError):
        load_results(path)


def test_cuda_guard_does_not_write(tmp_path):
    with pytest.raises(RuntimeError, match="Nothing was written"):
        require_cuda_backend()
    assert list(tmp_path.iterdir()) == []


def test_validate_results_requires_every_metric():
    payload = _payload()
    del payload["systems"]["dpo"]["parse_rate"]
    with pytest.raises(ValueError):
        validate_results(payload)


def test_shipped_readme_markers_and_license():
    readme = Path("README.md").read_text(encoding="utf-8")
    expected = Path("tests/fixtures/license_section.md").read_text(encoding="utf-8")
    assert readme[readme.index("## Licenses") :] == expected
    assert START in readme and END in readme
    results = Path("results/compare.json")
    if results.exists():
        # The shipped table must be exactly what the shipped results file renders.
        table = render_metrics_table(load_results(results))
        assert START + "\n" + table + "\n" + END in readme
    else:
        assert "| Parse rate | TBD | TBD | TBD | not yet measured |" in readme

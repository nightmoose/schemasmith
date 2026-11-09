import json
import re
from pathlib import Path

import pytest

from schemasmith.space_demo import (
    check_output,
    example_failure,
    labeled_json,
    record_by_id,
    sample_records,
    scorecard_markdown,
)


def test_mock_extract_is_the_label_and_validates():
    record = sample_records(1)[0]
    output = labeled_json(record)
    report = check_output(record, output)
    assert report["schema_valid"] is True
    assert report["parsed"] is True
    failure = example_failure(record)
    assert failure != output
    failed = check_output(record, failure)
    assert failed["schema_valid"] is False or failed["unsupported_fields"]


def test_scorecard_stays_tbd_without_a_file():
    text = scorecard_markdown(None)
    assert "TBD" in text
    assert "%" not in text
    missing = scorecard_markdown("reports/does-not-exist.json")
    assert "TBD" in missing


def test_sample_roundtrip():
    record = record_by_id(sample_records()[0].ticket_id)
    assert record.language == "en"


def test_abstract_and_demo_lengths():
    abstract = Path("docs/abstract.md").read_text(encoding="utf-8")
    demo = Path("docs/demo_script.md").read_text(encoding="utf-8")
    abstract_words = re.findall(r"[A-Za-z0-9']+", abstract)
    demo_words = re.findall(r"[A-Za-z0-9']+", demo)
    assert len(abstract_words) < 500
    assert 300 <= len(demo_words) <= 460


def test_roi_stays_empty_without_measurements(tmp_path):
    from schemasmith.cli import main

    out = tmp_path / "roi.json"
    assert main(["roi", "--out", str(out), "--gpu-hourly-usd", "0.4"]) == 0
    payload = json.loads(out.read_text(encoding="utf-8"))
    assert payload["status"] == "not measured"
    assert payload["core_model"] == "CohereLabs/tiny-aya-global"
    assert payload["self_host_usd_per_million_tickets"] is None
    assert payload["closed_usd_per_million_tickets"] is None


def test_docs_do_not_invent_metric_percents():
    for path in Path("docs").glob("*.md"):
        text = path.read_text(encoding="utf-8")
        assert "41%" not in text
        assert "96%" not in text
        assert "$" not in text


def test_dpo_notebook_calls_align_tune():
    text = Path("notebooks/lora_dpo_tiny_aya.ipynb").read_text(encoding="utf-8")
    assert "create_rl_trainer" in text
    assert '"dpo"' in text or "algorithm" in text
    assert "unsloth" not in text.lower()
    assert "merge_and_unload" in text


def test_space_app_builds():
    import importlib.util

    pytest.importorskip("gradio")
    path = Path("space/app.py").resolve()
    spec = importlib.util.spec_from_file_location("schemasmith_space_app", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    assert module.demo is not None


def test_remote_module_does_not_import_aligntune():
    import sys

    import schemasmith.hf.remote as remote

    assert remote.main
    assert "aligntune" not in sys.modules

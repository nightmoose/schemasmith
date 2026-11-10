import json
from pathlib import Path

from schemasmith.hostile.score import render_hostile_table

START = "<!-- SCHEMASMITH_HOSTILE_START -->"
END = "<!-- SCHEMASMITH_HOSTILE_END -->"


def test_readme_hostile_table_is_rendered_from_the_results_file():
    readme = Path("README.md").read_text(encoding="utf-8")
    payload = json.loads(Path("results/hostile.json").read_text(encoding="utf-8"))
    assert payload["measurement"] == "hostile"
    table = render_hostile_table(payload)
    assert START + "\n" + table + "\n" + END in readme


def test_readme_leads_with_the_unsupported_value_headline():
    readme = Path("README.md").read_text(encoding="utf-8")
    first = readme.split("\n## ", 1)[0]
    assert "32.9% to 0%" in first
    assert "unsupported-value rate" in first
    assert readme.index("## Results: held-out compare") < readme.index("## Results: hostile set")
    assert readme.index("## Results: hostile set") < readme.index("## Limits and caveats")


def test_recorded_run_is_real_and_matches_the_hostile_file():
    from schemasmith.space_demo import load_recorded

    payload = json.loads(Path("results/hostile.json").read_text(encoding="utf-8"))
    entries = load_recorded()
    assert entries
    for entry in entries:
        assert set(entry["runs"]) == {"base", "sft", "dpo"}
        if entry["set"].startswith("hostile"):
            for name, run in entry["runs"].items():
                row = next(r for r in payload["systems"][name]["rows"] if r["id"] == entry["id"])
                assert row["gate"] == run["gate"]

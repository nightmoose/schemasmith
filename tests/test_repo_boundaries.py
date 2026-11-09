"""The MIT tree does not vendor Lexsi source and does not carry secrets."""

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SKIP = {".git", ".venv", "__pycache__", ".pytest_cache", ".ruff_cache"}
SECRET = re.compile(
    r"(sk-[A-Za-z0-9]{20,}|hf_[A-Za-z0-9]{20,}|AKIA[0-9A-Z]{16}|"
    r"BEGIN (?:RSA |OPENSSH )?PRIVATE KEY)"
)


def _files():
    for path in ROOT.rglob("*"):
        if any(part in SKIP for part in path.parts):
            continue
        if path.is_file() and path.name != "test_repo_boundaries.py":
            yield path


def test_lexsi_packages_are_not_vendored():
    names = {path.name for path in ROOT.iterdir()}
    assert "curatorkit" not in names
    assert "auditkit" not in names
    assert "aligntune" not in names
    for path in _files():
        text = path.read_text(encoding="utf-8", errors="ignore")
        assert "Lithasa Technologies" not in text
        assert "This Source Available License governs" not in text


def test_no_secret_material():
    for path in _files():
        if path.suffix in {".py", ".md", ".yml", ".yaml", ".json", ".txt", ".ipynb", ".toml"}:
            assert SECRET.search(path.read_text(encoding="utf-8", errors="ignore")) is None

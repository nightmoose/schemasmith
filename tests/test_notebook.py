import json
from pathlib import Path


def test_notebook_is_the_trl_lora_entry():
    path = Path("notebooks/lora_sft_tiny_aya.ipynb")
    notebook = json.loads(path.read_text(encoding="utf-8"))
    sources = []
    for cell in notebook["cells"]:
        raw = cell["source"]
        sources.append(raw if isinstance(raw, str) else "".join(raw))
    joined = "\n".join(sources)
    assert "create_sft_trainer" in joined
    assert "lora_sft_kwargs" in joined
    assert "curate_with_curatorkit" in joined
    assert "not a model measurement" in joined
    assert "unsloth" not in joined.lower()
    assert "CohereLabs/tiny-aya-global" in joined

"""Dataset and adapter cards.

Measured numbers appear on a card only when the caller passes the compare
and hostile results payloads; they are read from those files, not typed in.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

LICENSE_NOTE = """\
## Licenses

The training and evaluation code in the SchemaSmith repository is MIT.

The core model is Tiny Aya Global, `CohereLabs/tiny-aya-global`.
No other model is trained or scored as the submission model.
Those weights, and every adapter derived from them, are CC-BY-NC-4.0.
Public adapter repos are not gated. The license is this card.

CuratorKIT and AuditKIT are external dependencies under the Lexsi Labs
Source Available License. Their source is not in this repository.
AlignTune is the training library. This project calls its TRL backend.
"""


def load_result_files(root: Path = Path("results")) -> tuple[dict | None, dict | None]:
    """``(compare, hostile)`` payloads from ``root``, or None when absent."""
    out: list[dict | None] = []
    for name in ("compare.json", "hostile.json"):
        path = root / name
        out.append(json.loads(path.read_text(encoding="utf-8")) if path.is_file() else None)
    return out[0], out[1]


def _pct(value: Any) -> str:
    if value is None:
        return "n/a"
    return "0%" if float(value) == 0 else f"{100 * float(value):.1f}%"


def results_lead(compare: dict | None, hostile: dict | None) -> str:
    """Headline and results for the top of a card. Empty without a compare file."""
    if not compare:
        return ""
    base, sft, dpo = (compare["systems"][k] for k in ("base", "sft", "dpo"))
    lines = [
        f"**Made-up field values: {_pct(base['unsupported_value_rate'])} to "
        f"{_pct(dpo['unsupported_value_rate'])}.** On {compare['n']} held-out "
        "multilingual support tickets, base Tiny Aya Global wrote an order reference or "
        f"refund amount that is not in the ticket for {base['unsupported_field_count']} of "
        f"{base['grounded_field_count']} ids and amounts (the unsupported-value rate). "
        f"With the SFT and DPO adapters: {dpo['unsupported_field_count']} of "
        f"{dpo['grounded_field_count']}.",
        "",
        "| Held-out metric | Base | SFT | DPO |",
        "| --- | --- | --- | --- |",
    ]
    for key, label in (
        ("unsupported_value_rate", "Unsupported values"),
        ("schema_valid_rate", "Schema-valid"),
        ("field_exact_match", "Field-level exact match"),
        ("non_english_contract_valid_rate", "Non-English contract-valid"),
    ):
        lines.append(f"| {label} | {_pct(base[key])} | {_pct(sft[key])} | {_pct(dpo[key])} |")
    if hostile:
        hb, hs, hd = (hostile["systems"][k] for k in ("base", "sft", "dpo"))
        lines += [
            "",
            f"Hostile set ({hostile['n']} hand-written tickets, zero overlap with training): "
            f"contract-valid {hb['schema_valid']} / {hs['schema_valid']} / "
            f"{hd['schema_valid']} (base / SFT / DPO); unsupported values "
            f"{_pct(hb['unsupported_value_rate'])} / {_pct(hs['unsupported_value_rate'])} / "
            f"{_pct(hd['unsupported_value_rate'])}; prompt injections refused "
            f"{hb['injections_refused']} / {hs['injections_refused']} / "
            f"{hd['injections_refused']} of {hd['injection_tickets']}. Injection is not "
            "solved; see the repository README for every miss and the limits.",
        ]
    return "\n".join(lines) + "\n"


def dataset_card() -> str:
    return f"""---
license: mit
tags:
- schemasmith
---

# Lexsi SchemaSmith support tickets

Synthetic multilingual support tickets and the contract-valid JSON labels
used to train a small extractor. This card does not report model scores.

{LICENSE_NOTE}
"""


def adapter_card(kind: str, compare: dict | None = None, hostile: dict | None = None) -> str:
    if kind not in ("sft", "dpo"):
        raise ValueError("kind must be sft or dpo")
    title = "SFT LoRA" if kind == "sft" else "DPO LoRA (final stage)"
    lead = results_lead(compare, hostile)
    scores = (
        "Scores above are copied from the compare and hostile results files."
        if lead
        else "Scores are not filled in on this card. Copy them from a compare results file "
        "after that command has been run."
    )
    return f"""---
license: cc-by-nc-4.0
base_model: CohereLabs/tiny-aya-global
tags:
- schemasmith
- lora
---

# Lexsi SchemaSmith {title}

{lead}
Core model: Tiny Aya Global (`CohereLabs/tiny-aya-global`).
This adapter is derived from that model and is licensed CC-BY-NC-4.0,
non-commercial. The repo is not gated. {scores}

{LICENSE_NOTE}
"""


def eval_logs_card(compare: dict | None = None, hostile: dict | None = None) -> str:
    lead = results_lead(compare, hostile)
    return f"""---
license: mit
tags:
- schemasmith
---

# Lexsi SchemaSmith eval logs

{lead}
`compare.json` and `predictions_*.jsonl` are the held-out compare.
`hostile/` holds the hostile-set run: `hostile.json` and per-system
predictions with per-ticket seconds. If a file is missing, that measurement
has not been uploaded.

{LICENSE_NOTE}
"""


def collection_description() -> str:
    text = "SchemaSmith ticket extraction: data, LoRA adapters, and eval logs."
    if len(text) > 150:
        raise RuntimeError("collection description is over the Hub limit")
    return text

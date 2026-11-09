"""Dataset and adapter cards. No measured scores are written here."""

from __future__ import annotations

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


def dataset_card() -> str:
    return f"""---
license: mit
tags:
- schemasmith
---

# SchemaSmith support tickets

Synthetic multilingual support tickets and the contract-valid JSON labels
used to train a small extractor. This card does not report model scores.

{LICENSE_NOTE}
"""


def adapter_card(kind: str) -> str:
    if kind not in ("sft", "dpo"):
        raise ValueError("kind must be sft or dpo")
    title = "SFT LoRA" if kind == "sft" else "DPO LoRA"
    return f"""---
license: cc-by-nc-4.0
base_model: CohereLabs/tiny-aya-global
tags:
- schemasmith
- lora
---

# SchemaSmith {title}

Core model: Tiny Aya Global (`CohereLabs/tiny-aya-global`).
This adapter is derived from that model and is licensed CC-BY-NC-4.0.
The repo is not gated. Scores are not filled in on this card. Copy them
from a compare results file after that command has been run.

{LICENSE_NOTE}
"""


def eval_logs_card() -> str:
    return f"""---
license: mit
tags:
- schemasmith
---

# SchemaSmith eval logs

JSON written by the held-out compare command. If this repository has no
`compare.json` yet, no model measurement has been uploaded.

{LICENSE_NOTE}
"""


def collection_description() -> str:
    text = "SchemaSmith ticket extraction: data, LoRA adapters, and eval logs."
    if len(text) > 150:
        raise RuntimeError("collection description is over the Hub limit")
    return text

"""Dataset and adapter cards.

Measured numbers appear on a card only when the caller passes the compare
and hostile results payloads, or when the AWS result files are present;
they are read from those files, not typed in.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

# Hub license ids that must appear in each card YAML and in Hub metadata.
# Dataset stays MIT (original synthetic tickets). Adapters and eval logs
# (model outputs) are CC-BY-NC-4.0, per Alex's Nov 11 decision.
HUB_LICENSES = {
    "dataset": "mit",
    "sft": "cc-by-nc-4.0",
    "dpo": "cc-by-nc-4.0",
    "eval_logs": "cc-by-nc-4.0",
}

LICENSE_NOTE = """\
## Licenses

The training and evaluation code in the SchemaSmith repository is MIT.
The synthetic support-ticket dataset is original work and is MIT.

The core model is Tiny Aya Global, `CohereLabs/tiny-aya-global`.
No other model is trained or scored as the submission model.
Those weights, every adapter derived from them, and the eval logs
(model outputs) are CC-BY-NC-4.0, non-commercial.
Public adapter and eval-log repos are not gated. The license is this card.

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


def load_aws_result_files(
    root: Path = Path("results/aws"),
) -> tuple[dict | None, dict | None, dict | None]:
    """``(v2_summary, ablation, guardrails)`` payloads, or None when absent."""
    v2_path = root / "sagemaker" / "v2-summary.json"
    ablation_path = root / "sagemaker" / "eval-partial" / "eval_from_logs.json"
    guard_path = root / "guardrails" / "summary.json"
    v2 = json.loads(v2_path.read_text(encoding="utf-8")) if v2_path.is_file() else None
    ablation = (
        json.loads(ablation_path.read_text(encoding="utf-8")) if ablation_path.is_file() else None
    )
    guard = json.loads(guard_path.read_text(encoding="utf-8")) if guard_path.is_file() else None
    return v2, ablation, guard


def _pct(value: Any) -> str:
    if value is None:
        return "n/a"
    return "0%" if float(value) == 0 else f"{100 * float(value):.1f}%"


def _points(value: Any, digits: int = 2) -> str:
    """Format a value already stored in percent points (e.g. 98.996 -> 99.00%)."""
    if value is None:
        return "n/a"
    return f"{float(value):.{digits}f}%"


def _one_decimal(value: Any) -> str:
    if value is None:
        return "n/a"
    return f"{float(value):.1f}"


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


def aws_results_section(
    v2: dict | None = None,
    ablation: dict | None = None,
    guardrails: dict | None = None,
) -> str:
    """v2 SageMaker and Guardrails numbers, read from the result files."""
    if v2 is None and ablation is None and guardrails is None:
        loaded = load_aws_result_files()
        v2, ablation, guardrails = loaded
    if not v2:
        return ""
    dpo_mean = v2["v2-mean/dpo"]
    dpo_max = v2["v2-max/dpo"]
    sft_mean = v2["v2-mean/sft"]
    v1_dpo = v2["v1/dpo"]
    n_seeds = v2["v2-n/dpo"]
    lines = [
        "## Later results (AWS, Nov 10): v2 training and Guardrails",
        "",
        f"v2 adapters were trained on Amazon SageMaker (LoRA SFT then DPO, with 480 "
        f"adversarial rows) and scored on a 256-ticket held-out split and a 30-ticket "
        f"blind hostile v2 set (10 injection tickets). Means are over {n_seeds} seeds, "
        "one eval each.",
        "",
        "| System | Held-out field exact match | Unsupported values | "
        "Hostile v2 injections refused / 10 |",
        "| --- | --- | --- | --- |",
        f"| v2 DPO mean (best seed) | {_points(dpo_mean['fem'])} "
        f"(best {_points(dpo_max['fem'])}) | {_points(dpo_mean['unsup'], 0)} | "
        f"{_one_decimal(dpo_mean['v2_refused'])} |",
        f"| v2 SFT mean | {_points(sft_mean['fem'])} | {_points(sft_mean['unsup'], 0)} | "
        f"{_one_decimal(sft_mean['v2_refused'])} |",
        f"| v1 DPO (same GPU) | {_points(v1_dpo['fem'])} | {_points(v1_dpo['unsup'])} | "
        f"{int(v1_dpo['v2_refused'])} |",
    ]
    if ablation and "v1data-s1/dpo" in ablation.get("systems", {}):
        ab = ablation["systems"]["v1data-s1/dpo"]
        fem = 100 * float(ab["heldout"]["field_exact_match"])
        unsup = 100 * float(ab["heldout"]["unsupported_value_rate"])
        refused = ab["hostile_v2"]["injections_refused"]
        lines.append(
            f"| Ablation (v1-data DPO, one seed) | {_points(fem)} | {_points(unsup, 0)} | "
            f"{refused} |"
        )
    lines += [
        "",
        "DPO is not clearly better than SFT on v2: it gains a little held-out accuracy, "
        "but SFT refused slightly more injections. All v2 runs wrote 0 unsupported values.",
    ]
    if guardrails:
        dpo = guardrails["systems"]["v1/dpo"]
        followed = dpo["hostile_v2"]["model_followed_injection"]
        released = dpo["hostile_v2"]["followed_and_released"]
        n_inj = dpo["hostile_v2"]["injection_tickets"]
        fp = dpo["heldout"]["false_positive_blocks"]
        n_clean = dpo["heldout"]["n"]
        lines += [
            "",
            "Amazon Bedrock Guardrails, applied to saved v1 DPO outputs (Tiny Aya was not "
            f"re-run): injections followed and released fell from {followed} of {n_inj} to "
            f"{released} of {n_inj}, with {fp} false-positive blocks on the {n_clean} "
            "held-out tickets.",
        ]
    lines += [
        "",
        f"Caveats: {n_seeds} seeds; hostile v2 is 30 rows; the ablation is one seed. "
        "Details are in `results/aws/sagemaker/README.md` and "
        "`results/aws/guardrails/README.md`.",
        "",
    ]
    return "\n".join(lines)


def dataset_card() -> str:
    return f"""---
license: {HUB_LICENSES["dataset"]}
tags:
- schemasmith
configs:
- config_name: sft_alpaca
  default: true
  data_files:
    - split: train
      path: train/sft_alpaca.jsonl
    - split: validation
      path: validation/sft_alpaca.jsonl
    - split: test
      path: test/sft_alpaca.jsonl
- config_name: sft_messages
  data_files:
    - split: train
      path: train/sft_messages.jsonl
    - split: validation
      path: validation/sft_messages.jsonl
    - split: test
      path: test/sft_messages.jsonl
---

# Lexsi SchemaSmith support tickets

Synthetic multilingual support tickets and the contract-valid JSON labels
used to train a small extractor. This card does not report model scores.
The tickets are original work in this repository and are licensed MIT.
Adapters and eval logs derived from Tiny Aya Global are CC-BY-NC-4.0.

{LICENSE_NOTE}
"""


def adapter_card(kind: str, compare: dict | None = None, hostile: dict | None = None) -> str:
    if kind not in ("sft", "dpo"):
        raise ValueError("kind must be sft or dpo")
    title = "SFT LoRA" if kind == "sft" else "DPO LoRA (final stage)"
    lead = results_lead(compare, hostile)
    aws = aws_results_section()
    scores = (
        "Scores above are copied from the compare, hostile, and AWS results files."
        if lead or aws
        else "Scores are not filled in on this card. Copy them from a compare results file "
        "after that command has been run."
    )
    return f"""---
license: {HUB_LICENSES[kind]}
base_model: CohereLabs/tiny-aya-global
tags:
- schemasmith
- lora
---

# Lexsi SchemaSmith {title}

{lead}{aws}
Core model: Tiny Aya Global (`CohereLabs/tiny-aya-global`).
This adapter is derived from that model and is licensed CC-BY-NC-4.0,
non-commercial. The repo is not gated. {scores}

{LICENSE_NOTE}
"""


def eval_logs_card(compare: dict | None = None, hostile: dict | None = None) -> str:
    lead = results_lead(compare, hostile)
    aws = aws_results_section()
    return f"""---
license: {HUB_LICENSES["eval_logs"]}
tags:
- schemasmith
---

# Lexsi SchemaSmith eval logs

{lead}{aws}
`compare.json` and `predictions_*.jsonl` are the held-out compare.
`hostile/` holds the hostile-set run: `hostile.json` and per-system
predictions with per-ticket seconds. If a file is missing, that measurement
has not been uploaded. AWS v2 and Guardrails measurements live in the
GitHub repository under `results/aws/` and are not in this dataset yet.

These logs contain generations from Tiny Aya Global and its adapters
and are licensed CC-BY-NC-4.0, non-commercial. The repo is not gated.

{LICENSE_NOTE}
"""


def collection_description() -> str:
    text = "SchemaSmith ticket extraction: data, LoRA adapters, and eval logs."
    if len(text) > 150:
        raise RuntimeError("collection description is over the Hub limit")
    return text

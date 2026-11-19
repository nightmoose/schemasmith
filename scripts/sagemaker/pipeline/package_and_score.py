"""Pipeline processing step: score gate inputs and build the serving artifact.

Inputs (processing container paths):
  /opt/ml/processing/adapter/model.tar.gz  training step output (sft-adapter, dpo-adapter)
  /opt/ml/processing/eval/model.tar.gz     eval step output (eval/eval.json)
  /opt/ml/processing/serve/                inference.py + requirements.txt
Outputs:
  /opt/ml/processing/evaluation/evaluation.json   PropertyFile for the ConditionStep
  /opt/ml/processing/evaluation/model_quality.json ModelMetrics for the registry
  /opt/ml/processing/model/model.tar.gz            adapters + code/ for the endpoint
Standard library only, so it runs on the stock scikit-learn CPU image.
"""

from __future__ import annotations

import json
import shutil
import tarfile
from pathlib import Path

P = Path("/opt/ml/processing")


def _extract(src: Path, dest: Path) -> None:
    with tarfile.open(src) as t:
        try:
            t.extractall(dest, filter="data")
        except TypeError:  # Python without tarfile filters
            t.extractall(dest)


def _find(root: Path, name: str) -> Path:
    hits = sorted(root.rglob(name))
    if not hits:
        raise SystemExit(f"missing {name} under {root}")
    return hits[0]


def main() -> None:
    work = Path("/tmp/pkg")
    shutil.rmtree(work, ignore_errors=True)
    (work / "adapter").mkdir(parents=True)
    (work / "eval").mkdir()
    _extract(_find(P / "adapter", "model.tar.gz"), work / "adapter")
    _extract(_find(P / "eval", "model.tar.gz"), work / "eval")
    ev = json.loads(_find(work / "eval", "eval.json").read_text())
    dpo = ev["systems"]["dpo"]
    sft = ev["systems"]["sft"]
    h, v2 = dpo["heldout"], dpo["hostile_v2"]
    summary = {
        "candidate": "dpo (SFT merged, then DPO adapter)",
        "heldout_n": h["n"],
        "contract_valid_rate": h["contract_valid_rate"],
        "schema_valid_rate": h["schema_valid_rate"],
        "field_exact_match": h["field_exact_match"],
        "unsupported_value_rate": h["unsupported_value_rate"],
        "hostile_v2_injections_refused": v2["injections_refused"],
        "hostile_v2_injection_tickets": v2["injection_tickets"],
        "hostile_v2_forbidden_value_written": v2["forbidden_value_written"],
        "hostile_v1_injections_refused": dpo["hostile"]["injections_refused"],
        "sft_field_exact_match": sft["heldout"]["field_exact_match"],
        "mean_batch_seconds_per_row": dpo["mean_batch_seconds_per_row"],
    }
    out = P / "evaluation"
    out.mkdir(parents=True, exist_ok=True)
    (out / "evaluation.json").write_text(json.dumps(summary, indent=2) + "\n")
    # SageMaker Model Registry "model quality" report shape.
    mq = {
        "extraction_metrics": {
            k: {"value": summary[k], "standard_deviation": "NaN"}
            for k in (
                "contract_valid_rate",
                "field_exact_match",
                "unsupported_value_rate",
                "hostile_v2_injections_refused",
                "hostile_v2_forbidden_value_written",
            )
        }
    }
    (out / "model_quality.json").write_text(json.dumps(mq, indent=2) + "\n")
    shutil.copy(_find(work / "eval", "eval.json"), out / "eval_full.json")
    # Serving artifact: adapters at the root, handler in code/.
    stage = work / "serve"
    stage.mkdir()
    for d in ("sft-adapter", "dpo-adapter"):
        shutil.copytree(_find(work / "adapter", "adapter_config.json").parent.parent / d, stage / d)
    shutil.copytree(P / "serve", stage / "code")
    m = P / "model"
    m.mkdir(parents=True, exist_ok=True)
    with tarfile.open(m / "model.tar.gz", "w:gz") as t:
        for item in sorted(stage.iterdir()):
            t.add(item, arcname=item.name)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()

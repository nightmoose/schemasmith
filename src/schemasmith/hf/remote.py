"""Commands that run inside an HF Job. They do not launch other jobs.

Pushing uses the HF_TOKEN secret already in the environment and creates
private repos only. There is no public flag on this path.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

from schemasmith.data.generator import generate_tickets
from schemasmith.data.preferences import write_dpo_dataset
from schemasmith.hf.jobs import private_repo_ids
from schemasmith.training import MODEL_ID, lora_dpo_kwargs, lora_sft_kwargs


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="schemasmith.hf.remote")
    parser.add_argument("kind", choices=("sft", "dpo", "eval"))
    parser.add_argument("--out", type=Path, default=Path("/tmp/schemasmith-job"))
    parser.add_argument("--namespace", default=os.environ.get("SCHEMASMITH_NAMESPACE", ""))
    args = parser.parse_args(argv)
    if not args.namespace:
        print("SCHEMASMITH_NAMESPACE is required", file=sys.stderr)
        return 2
    if args.kind == "sft":
        run_sft(args.out, args.namespace)
    elif args.kind == "dpo":
        run_dpo(args.out, args.namespace)
    else:
        run_eval(args.out, args.namespace)
    return 0


def prepare_train_split(out: Path, n: int = 24, seed: int = 42):
    from schemasmith.curation.local import curate_local

    records = generate_tickets(n, seed)
    curated = curate_local(records, out / "curated", seed=seed)
    return curated


def run_sft(out: Path, namespace: str) -> None:
    curated = prepare_train_split(out)
    dataset_dir = curated.output_dir
    from aligntune.core.backend_factory import create_sft_trainer

    adapter_dir = out / "sft-adapter"
    trainer = create_sft_trainer(
        **lora_sft_kwargs(dataset_dir=str(dataset_dir), output_dir=str(adapter_dir))
    )
    trainer.train()
    _push_private(namespace, "sft", adapter_dir)
    _push_private(namespace, "dataset", dataset_dir)


def run_dpo(out: Path, namespace: str) -> None:
    curated = prepare_train_split(out)
    train_rows = curated.splits["train"]
    dpo_dir = out / "dpo-data"
    write_dpo_dataset(train_rows, dpo_dir, seed=42)
    repos = private_repo_ids(namespace)
    merged = _merge_sft_adapter(repos["sft"], out / "sft-merged")
    from aligntune.core.backend_factory import create_rl_trainer

    adapter_dir = out / "dpo-adapter"
    trainer = create_rl_trainer(
        **lora_dpo_kwargs(
            dataset_dir=str(dpo_dir),
            output_dir=str(adapter_dir),
            model_name=str(merged),
        )
    )
    trainer.train()
    _push_private(namespace, "dpo", adapter_dir)


def run_eval(out: Path, namespace: str) -> None:
    from schemasmith.eval.compare import results_from_scorecards, write_results
    from schemasmith.eval.generate import generate_texts
    from schemasmith.eval.harness import evaluate_records
    from schemasmith.prompts import user_content

    curated = prepare_train_split(out)
    held_out = curated.splits["test"]
    if not held_out:
        raise RuntimeError("test split is empty")
    prompts = [user_content(row.text) for row in held_out]
    repos = private_repo_ids(namespace)
    cards = {}
    for name, adapter in (
        ("base", None),
        ("sft", repos["sft"]),
        ("dpo", repos["dpo"]),
    ):
        outputs = generate_texts(prompts, model_id=MODEL_ID, adapter=adapter)
        cards[name] = evaluate_records(held_out, outputs=outputs)
    payload = results_from_scorecards(
        cards,
        data_path="seed-42 local test split",
        sft_adapter=repos["sft"],
        dpo_adapter=repos["dpo"],
    )
    dest = out / "compare.json"
    write_results(payload, dest)
    _push_private(namespace, "eval_logs", out)


def _merge_sft_adapter(repo_id: str, dest: Path) -> Path:
    """Merge the private SFT adapter into a local weight directory.

    AlignTune's DPO loader starts from ``model_name`` and then adds a new
    LoRA. Merging is how the SFT update is carried into that starting point.
    """
    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    if not torch.cuda.is_available():
        raise RuntimeError("DPO merge needs a CUDA GPU.")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    base = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )
    merged = PeftModel.from_pretrained(base, repo_id).merge_and_unload()
    dest.mkdir(parents=True, exist_ok=True)
    merged.save_pretrained(dest)
    tokenizer.save_pretrained(dest)
    return dest


def _push_private(namespace: str, key: str, folder: Path) -> None:
    if os.environ.get("SCHEMASMITH_PUSH") != "1":
        print(f"not pushing {key}; SCHEMASMITH_PUSH is not 1")
        return
    token = os.environ.get("HF_TOKEN")
    if not token:
        raise RuntimeError("HF_TOKEN is required to keep the job output.")
    from huggingface_hub import HfApi

    repo_id = private_repo_ids(namespace)[key]
    repo_type = "model" if key in ("sft", "dpo") else "dataset"
    api = HfApi(token=token)
    api.create_repo(repo_id=repo_id, repo_type=repo_type, private=True, exist_ok=True)
    api.upload_folder(folder_path=str(folder), repo_id=repo_id, repo_type=repo_type)
    print(f"pushed private {repo_type} {repo_id}")


if __name__ == "__main__":
    raise SystemExit(main())

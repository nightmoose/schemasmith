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
from schemasmith.data.preferences import chat_format_dpo_dir, write_dpo_dataset
from schemasmith.hf.jobs import private_repo_ids
from schemasmith.training import DEFAULT_N, MODEL_ID, lora_dpo_kwargs, lora_sft_kwargs


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="schemasmith.hf.remote")
    parser.add_argument("kind", choices=("sft", "dpo", "eval", "hostile"))
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
    elif args.kind == "eval":
        run_eval(args.out, args.namespace)
    else:
        run_hostile(args.out, args.namespace)
    return 0


def prepare_train_split(out: Path, n: int | None = None, seed: int = 42):
    from schemasmith.curation.local import curate_local

    if n is None:
        n = int(os.environ.get("SCHEMASMITH_N", DEFAULT_N))
    records = generate_tickets(n, seed)
    curated = curate_local(records, out / "curated", seed=seed)
    return curated


def run_sft(out: Path, namespace: str) -> None:
    curated = prepare_train_split(out)
    dataset_dir = curated.output_dir
    from aligntune.core.backend_factory import create_sft_trainer

    adapter_dir = out / "sft-adapter"
    trainer = create_sft_trainer(
        **lora_sft_kwargs(
            dataset_dir=str(dataset_dir),
            output_dir=str(adapter_dir),
            train_rows=len(curated.splits["train"]),
        )
    )
    trainer.train()
    _push_private(namespace, "sft", adapter_dir)
    _push_private(namespace, "dataset", dataset_dir)


def run_dpo(out: Path, namespace: str) -> None:
    curated = prepare_train_split(out)
    train_rows = curated.splits["train"]
    plain_dir = out / "dpo-plain"
    dpo_dir = out / "dpo-chat"
    summary = write_dpo_dataset(train_rows, plain_dir, seed=42)
    from transformers import AutoTokenizer

    rendered = chat_format_dpo_dir(plain_dir, AutoTokenizer.from_pretrained(MODEL_ID), dpo_dir)
    print(f"rendered {rendered} DPO pairs with the chat template")
    repos = private_repo_ids(namespace)
    merged = _merge_sft_adapter(repos["sft"], out / "sft-merged")
    from aligntune.core.backend_factory import create_rl_trainer

    adapter_dir = out / "dpo-adapter"
    trainer = create_rl_trainer(
        **lora_dpo_kwargs(
            dataset_dir=str(dpo_dir),
            output_dir=str(adapter_dir),
            model_name=str(merged),
            train_rows=summary["split_counts"]["train"],
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
    out.mkdir(parents=True, exist_ok=True)
    for name, adapter, merge in (
        ("base", None, None),
        ("sft", repos["sft"], None),
        # DPO was trained on SFT-merged weights. Merge SFT first.
        ("dpo", repos["dpo"], repos["sft"]),
    ):
        outputs = generate_texts(prompts, model_id=MODEL_ID, adapter=adapter, merge_adapter=merge)
        _write_predictions(out / f"predictions_{name}.jsonl", held_out, outputs)
        cards[name] = evaluate_records(held_out, outputs=outputs)
    payload = results_from_scorecards(
        cards,
        data_path=f"seed-42 local test split of {len(curated.kept)} generated tickets",
        sft_adapter=repos["sft"],
        dpo_adapter=repos["dpo"],
    )
    dest = out / "compare.json"
    write_results(payload, dest)
    _push_private(namespace, "eval_logs", out)


def run_hostile(out: Path, namespace: str) -> None:
    """Base, SFT, and DPO on the hand-written hostile tickets.

    Writes ``hostile.json`` and one predictions file per system, with the
    wall-clock seconds each ticket took, and pushes them to ``hostile/`` in
    the private eval-logs repo. Nothing is trained here.
    """
    import json

    from schemasmith.eval.generate import generate_texts
    from schemasmith.hostile import TICKETS, score_hostile
    from schemasmith.prompts import user_content

    out = out / "hostile"
    out.mkdir(parents=True, exist_ok=True)
    prompts = [user_content(t.text) for t in TICKETS]
    repos = private_repo_ids(namespace)
    systems = {}
    for name, adapter, merge in (
        ("base", None, None),
        ("sft", repos["sft"], None),
        ("dpo", repos["dpo"], repos["sft"]),
    ):
        seconds: list[float] = []
        outputs = generate_texts(
            prompts, model_id=MODEL_ID, adapter=adapter, merge_adapter=merge, timings=seconds
        )
        with (out / f"predictions_{name}.jsonl").open("w", encoding="utf-8") as fh:
            for ticket, text, secs in zip(TICKETS, outputs, seconds, strict=True):
                row = {"id": ticket.ticket_id, "kind": ticket.kind, "output": text}
                row["seconds"] = round(secs, 3)
                fh.write(json.dumps(row, ensure_ascii=False) + "\n")
        systems[name] = score_hostile(TICKETS, outputs)
        print(name, {k: v for k, v in systems[name].items() if k not in ("rows", "by_kind")})
    payload = {
        "measurement": "hostile",
        "model_id": MODEL_ID,
        "sft_adapter": repos["sft"],
        "dpo_adapter": repos["dpo"],
        "n": len(TICKETS),
        "generation": "greedy, max 256 new tokens, stop on config EOS and chat turn end",
        "systems": systems,
    }
    (out / "hostile.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    _push_private(namespace, "eval_logs", out, path_in_repo="hostile")


def _write_predictions(path: Path, rows, outputs) -> None:
    import json

    with path.open("w", encoding="utf-8") as fh:
        for row, text in zip(rows, outputs, strict=True):
            fh.write(json.dumps({"id": row.ticket_id, "output": text}, ensure_ascii=False) + "\n")


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


def _push_private(namespace: str, key: str, folder: Path, path_in_repo: str = "") -> None:
    if os.environ.get("SCHEMASMITH_PUSH") != "1":
        print(f"not pushing {key}; SCHEMASMITH_PUSH is not 1")
        return
    token = os.environ.get("HF_TOKEN")
    if not token:
        raise RuntimeError("HF_TOKEN is required to keep the job output.")
    from huggingface_hub import HfApi

    repo_id = private_repo_ids(namespace)[key]
    repo_type = "model" if key in ("sft", "dpo") else "dataset"
    if key in ("sft", "dpo"):
        _fix_card_base_model(folder)
    api = HfApi(token=token)
    api.create_repo(repo_id=repo_id, repo_type=repo_type, private=True, exist_ok=True)
    api.upload_folder(
        folder_path=str(folder),
        repo_id=repo_id,
        repo_type=repo_type,
        path_in_repo=path_in_repo or None,
        # Trainer checkpoints carry optimizer state and are not needed.
        ignore_patterns=["checkpoint-*", "checkpoint-*/**", "**/checkpoint-*/**"],
    )
    print(f"pushed private {repo_type} {repo_id}")


def _fix_card_base_model(folder: Path) -> None:
    """Point a local-path base_model in the PEFT card at the Hub model id.

    The DPO adapter starts from SFT-merged weights saved under /tmp, and the
    Hub rejects a card whose base_model is a local path.
    """
    import json
    import re

    card = folder / "README.md"
    if card.exists():
        text = card.read_text(encoding="utf-8")
        text = re.sub(r"(base_model:(?:adapter:)?\s*)(/\S+)", rf"\g<1>{MODEL_ID}", text)
        text = re.sub(r"(?m)^(\s*-\s*)(/\S+)\s*$", rf"\g<1>{MODEL_ID}", text)
        card.write_text(text, encoding="utf-8")
    cfg = folder / "adapter_config.json"
    if cfg.exists():
        data = json.loads(cfg.read_text(encoding="utf-8"))
        base = data.get("base_model_name_or_path") or ""
        if base.startswith("/"):
            data["base_model_name_or_path"] = MODEL_ID
            cfg.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    raise SystemExit(main())

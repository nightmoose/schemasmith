"""SchemaSmith v2 runs on one AWS GPU instance (EC2, Deep Learning AMI).

Everything stays on the instance's disk; the box copies results back over
SSH. Nothing is pushed anywhere from here. HF_TOKEN (read access) is needed
only to download the gated base model and the private v1 adapters.

    python -m schemasmith.aws.run prep --work W
    python -m schemasmith.aws.run train --work W --tag v2-s1 --seed 1
    python -m schemasmith.aws.run train --work W --tag v1data-s1 --seed 1 --no-adv
    python -m schemasmith.aws.run eval --work W --tag v1 --sft HUB_SFT --dpo HUB_DPO --base

Eval sets: the held-out test split of 1280 generated tickets (seed 42; 256
rows, a superset of the 64-row v1 split), hostile v1 (26), hostile v2 (30).
"""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path
from typing import Any

from schemasmith.training import MODEL_ID, lora_dpo_kwargs, lora_sft_kwargs, steps_for

N_TICKETS = 1280
N_ADV = 480
DATA_SEED = 42
SFT_EPOCHS_V2 = 2


def _splits(with_adv: bool):
    from schemasmith.curation.local import assign_splits
    from schemasmith.data.adversarial import adversarial_tickets
    from schemasmith.data.generator import generate_tickets

    records = generate_tickets(N_TICKETS, DATA_SEED)
    splits = assign_splits(records, DATA_SEED)
    adv = adversarial_tickets(N_ADV, DATA_SEED) if with_adv else []
    return splits, adv


def cmd_prep(work: Path) -> None:
    """Write both training sets (with and without adversarial rows) and the eval sets."""
    from schemasmith.curation.local import curate_local
    from schemasmith.data.loader import write_jsonl

    for name, with_adv in (("v2", True), ("v1data", False)):
        splits, adv = _splits(with_adv)
        records = splits["train"] + splits["validation"] + adv
        # curate_local re-splits by split_group: test rows stay test, the rest
        # are shuffled with the seed into train and validation.
        cur = curate_local(records + splits["test"], work / "data" / name, seed=DATA_SEED)
        print(name, {k: len(v) for k, v in cur.splits.items()})
        write_jsonl(
            work / "data" / name / "train_ids.jsonl",
            [{"id": r.ticket_id, "template": r.template} for r in cur.splits["train"]],
        )
    held = splits["test"]
    write_jsonl(work / "data" / "heldout.jsonl", [r.to_dict() for r in held])
    print("heldout", len(held))


def _train_rows(work: Path, name: str, with_adv: bool):
    from schemasmith.curation.local import curate_local

    splits, adv = _splits(with_adv)
    records = splits["train"] + splits["validation"] + adv + splits["test"]
    cur = curate_local(records, work / "data" / name, seed=DATA_SEED)
    return cur


def _dpo_pairs(rows, seed: int) -> list[dict[str, Any]]:
    from schemasmith.data.adversarial import injected_rejection
    from schemasmith.data.preferences import FAILURE_MODES, _reject
    from schemasmith.prompts import user_content

    pairs = []
    for index, record in enumerate(rows):
        chosen = json.dumps(record.gold, ensure_ascii=False)
        rejected = (
            injected_rejection(record) if (record.template or "").startswith("adv/") else None
        )
        mode = "followed_injection"
        if rejected is None or rejected == chosen:
            mode = FAILURE_MODES[(seed + index) % len(FAILURE_MODES)]
            rejected = _reject(record, mode)
        pairs.append(
            {
                "ticket_id": record.ticket_id,
                "failure_mode": mode,
                "prompt": user_content(record.text),
                "chosen": chosen,
                "rejected": rejected,
            }
        )
    return pairs


def cmd_train(work: Path, tag: str, seed: int, with_adv: bool, sft_epochs: int) -> None:
    from aligntune.core.backend_factory import create_rl_trainer, create_sft_trainer
    from transformers import AutoTokenizer

    from schemasmith.data.loader import write_jsonl
    from schemasmith.data.preferences import _split_pairs, _write_split, chat_format_dpo_dir
    from schemasmith.hf.remote import _merge_sft_adapter

    name = "v2" if with_adv else "v1data"
    out = work / tag
    out.mkdir(parents=True, exist_ok=True)
    cur = _train_rows(work, name, with_adv)
    train = cur.splits["train"]
    log: dict[str, Any] = {
        "tag": tag,
        "seed": seed,
        "data": name,
        "train_rows": len(train),
        "validation_rows": len(cur.splits["validation"]),
    }

    t0 = time.time()
    kw = lora_sft_kwargs(
        dataset_dir=str(cur.output_dir), output_dir=str(out / "sft-adapter"), train_rows=len(train)
    )
    kw["num_epochs"] = sft_epochs
    kw["max_steps"] = steps_for(len(train), sft_epochs)
    kw["save_steps"] = kw["max_steps"]
    kw["seed"] = seed
    log["sft"] = {k: kw[k] for k in ("num_epochs", "max_steps", "learning_rate", "lora_r", "seed")}
    create_sft_trainer(**kw).train()
    log["sft"]["seconds"] = round(time.time() - t0, 1)

    t1 = time.time()
    pairs = _dpo_pairs(train, DATA_SEED)
    tr, va = _split_pairs(pairs, DATA_SEED)
    plain = out / "dpo-plain"
    _write_split(plain / "train" / "dpo.jsonl", tr)
    _write_split(plain / "validation" / "dpo.jsonl", va)
    write_jsonl(plain / "pairs.jsonl", pairs)
    modes: dict[str, int] = {}
    for p in pairs:
        modes[p["failure_mode"]] = modes.get(p["failure_mode"], 0) + 1
    log["dpo_pairs"] = {"train": len(tr), "validation": len(va), "modes": modes}
    chat_format_dpo_dir(plain, AutoTokenizer.from_pretrained(MODEL_ID), out / "dpo-chat")
    merged = _merge_sft_adapter(str(out / "sft-adapter"), out / "sft-merged")
    dkw = lora_dpo_kwargs(
        dataset_dir=str(out / "dpo-chat"),
        output_dir=str(out / "dpo-adapter"),
        model_name=str(merged),
        train_rows=len(tr),
    )
    dkw["seed"] = seed
    log["dpo"] = {k: dkw[k] for k in ("num_epochs", "max_steps", "learning_rate", "beta", "seed")}
    create_rl_trainer(**dkw).train()
    log["dpo"]["seconds"] = round(time.time() - t1, 1)
    import shutil

    shutil.rmtree(merged, ignore_errors=True)
    for d in (out / "sft-adapter", out / "dpo-adapter"):
        for ck in d.glob("checkpoint-*"):
            shutil.rmtree(ck, ignore_errors=True)
    (out / "train_log.json").write_text(json.dumps(log, indent=2) + "\n")
    print("TRAIN_DONE", tag, json.dumps(log))


def generate_batched(
    prompts, *, adapter=None, merge_adapter=None, batch_size=16, max_new_tokens=256
):
    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    from schemasmith.eval.generate import stop_token_ids

    tok = AutoTokenizer.from_pretrained(MODEL_ID)
    tok.padding_side = "left"
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID, torch_dtype=torch.bfloat16, device_map="cuda"
    )
    if merge_adapter:
        model = PeftModel.from_pretrained(model, merge_adapter).merge_and_unload()
    if adapter:
        model = PeftModel.from_pretrained(model, adapter).merge_and_unload()
    model.eval()
    stops = stop_token_ids(tok, model.generation_config.eos_token_id)
    texts = [
        tok.apply_chat_template(
            [{"role": "user", "content": p}], tokenize=False, add_generation_prompt=True
        )
        for p in prompts
    ]
    order = sorted(range(len(texts)), key=lambda i: len(texts[i]))
    outs: list[str] = [""] * len(texts)
    secs: list[float] = [0.0] * len(texts)
    for start in range(0, len(order), batch_size):
        idx = order[start : start + batch_size]
        enc = tok(
            [texts[i] for i in idx], return_tensors="pt", padding=True, add_special_tokens=False
        ).to("cuda")
        t = time.perf_counter()
        with torch.no_grad():
            gen = model.generate(
                **enc,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                eos_token_id=stops,
                pad_token_id=tok.pad_token_id,
            )
        torch.cuda.synchronize()
        dt = time.perf_counter() - t
        for j, i in enumerate(idx):
            outs[i] = tok.decode(gen[j, enc["input_ids"].shape[1] :], skip_special_tokens=True)
            secs[i] = dt / len(idx)
    del model
    torch.cuda.empty_cache()
    return outs, secs


def _eval_sets():
    from schemasmith.data.loader import load_jsonl
    from schemasmith.data.records import TicketRecord
    from schemasmith.hostile import TICKETS
    from schemasmith.hostile.tickets_v2 import TICKETS_V2

    return TicketRecord, load_jsonl, TICKETS, TICKETS_V2


def cmd_eval(work: Path, tag: str, systems: list[tuple[str, str | None, str | None]]) -> None:
    from schemasmith.data.loader import load_jsonl
    from schemasmith.data.records import TicketRecord
    from schemasmith.eval.harness import evaluate_records
    from schemasmith.hostile import TICKETS, score_hostile
    from schemasmith.hostile.tickets_v2 import TICKETS_V2
    from schemasmith.prompts import user_content

    held = [TicketRecord.from_dict(d) for d in load_jsonl(work / "data" / "heldout.jsonl")]
    held64 = [r for r in held if int(r.ticket_id[2:]) < 100320]
    out = work / tag / "eval"
    out.mkdir(parents=True, exist_ok=True)
    sets = {
        "heldout": [(r.ticket_id, user_content(r.text)) for r in held],
        "hostile": [(t.ticket_id, user_content(t.text)) for t in TICKETS],
        "hostile_v2": [(t.ticket_id, user_content(t.text)) for t in TICKETS_V2],
    }
    flat = [(s, rid, p) for s, rows in sets.items() for rid, p in rows]
    result: dict[str, Any] = {
        "tag": tag,
        "model_id": MODEL_ID,
        "generation": "greedy, max 256 new tokens, batched 16 with left "
        "padding, stop on config EOS and chat turn end",
        "systems": {},
    }
    for name, adapter, merge in systems:
        outs, secs = generate_batched([p for _, _, p in flat], adapter=adapter, merge_adapter=merge)
        by_set: dict[str, list[str]] = {s: [] for s in sets}
        with (out / f"predictions_{name}.jsonl").open("w", encoding="utf-8") as fh:
            for (s, rid, _), text, sec in zip(flat, outs, secs, strict=True):
                by_set[s].append(text)
                fh.write(
                    json.dumps(
                        {
                            "set": s,
                            "id": rid,
                            "output": text,
                            "batch_seconds_per_row": round(sec, 4),
                        },
                        ensure_ascii=False,
                    )
                    + "\n"
                )
        pos = {r.ticket_id: i for i, r in enumerate(held)}
        block = {
            "adapter": adapter,
            "merge_first": merge,
            "heldout": evaluate_records(held, outputs=by_set["heldout"]).to_dict(),
            "heldout64": evaluate_records(
                held64, outputs=[by_set["heldout"][pos[r.ticket_id]] for r in held64]
            ).to_dict(),
            "hostile": score_hostile(TICKETS, by_set["hostile"]),
            "hostile_v2": score_hostile(TICKETS_V2, by_set["hostile_v2"]),
            "mean_batch_seconds_per_row": round(sum(secs) / len(secs), 4),
        }
        result["systems"][name] = block
        h, x, y = block["heldout"], block["hostile"], block["hostile_v2"]
        print(
            "EVAL",
            tag,
            name,
            "held valid",
            h["schema_valid_rate"],
            "fem",
            round(h["field_exact_match"], 4),
            "unsup",
            round(h["unsupported_value_rate"], 4),
            "| hostile valid",
            x["schema_valid"],
            "refused",
            x["injections_refused"],
            "forbidden",
            x["forbidden_value_written"],
            "accwrong",
            x["accepted_but_wrong"],
            "| v2 valid",
            y["schema_valid"],
            "refused",
            y["injections_refused"],
            "forbidden",
            y["forbidden_value_written"],
            "accwrong",
            y["accepted_but_wrong"],
            flush=True,
        )
        (out / "eval.json").write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n")
    print("EVAL_DONE", tag)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="schemasmith.aws.run")
    ap.add_argument("cmd", choices=("prep", "train", "eval"))
    ap.add_argument("--work", type=Path, default=Path(os.environ.get("WORK", "/opt/dlami/nvme/w")))
    ap.add_argument("--tag", default="v2-s1")
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--no-adv", action="store_true")
    ap.add_argument("--sft-epochs", type=int, default=SFT_EPOCHS_V2)
    ap.add_argument("--base", action="store_true", help="also score the base model")
    ap.add_argument("--sft", help="SFT adapter path or Hub id")
    ap.add_argument("--dpo", help="DPO adapter path or Hub id (scored on SFT-merged weights)")
    args = ap.parse_args(argv)
    if args.cmd == "prep":
        cmd_prep(args.work)
    elif args.cmd == "train":
        cmd_train(args.work, args.tag, args.seed, not args.no_adv, args.sft_epochs)
    else:
        systems: list[tuple[str, str | None, str | None]] = []
        if args.base:
            systems.append(("base", None, None))
        sft = args.sft or str(args.work / args.tag / "sft-adapter")
        dpo = args.dpo or str(args.work / args.tag / "dpo-adapter")
        systems += [("sft", sft, None), ("dpo", dpo, sft)]
        cmd_eval(args.work, args.tag, systems)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

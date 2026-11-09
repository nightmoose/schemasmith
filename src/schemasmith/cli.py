"""Command line for generate, curate, and score."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from schemasmith import __version__
from schemasmith.curation.curatorkit import curate_with_curatorkit
from schemasmith.curation.local import curate_local
from schemasmith.data.generator import generate_tickets
from schemasmith.data.loader import load_alpaca_jsonl, load_ticket_jsonl, write_jsonl
from schemasmith.data.preferences import write_dpo_dataset
from schemasmith.eval.auditkit_bridge import run_auditkit
from schemasmith.eval.compare import (
    load_results,
    results_from_scorecards,
    update_readme,
    write_results,
)
from schemasmith.eval.generate import generate_texts
from schemasmith.eval.harness import evaluate_records
from schemasmith.hf.hardware import DEFAULT_FLAVOR, DEFAULT_TIMEOUT_SECONDS
from schemasmith.hf.jobs import (
    build_plan,
    launch_plan,
    live_hardware_rates,
    render_plan,
    token_from_env,
)
from schemasmith.hf.publish import build_publish_plan, publish
from schemasmith.prompts import user_content
from schemasmith.roi import roi_from_compare
from schemasmith.training import MODEL_ID


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="schemasmith")
    parser.add_argument("--version", action="version", version=__version__)
    sub = parser.add_subparsers(dest="command", required=True)

    generate = sub.add_parser("generate", help="write synthetic tickets as JSONL")
    generate.add_argument("--n", type=int, default=24)
    generate.add_argument("--seed", type=int, default=42)
    generate.add_argument("--out", type=Path, required=True)

    curate = sub.add_parser("curate", help="contract-gate and export a ticket JSONL")
    curate.add_argument("--raw", type=Path, required=True)
    curate.add_argument("--out", type=Path, required=True)
    curate.add_argument("--seed", type=int, default=42)
    curate.add_argument(
        "--curatorkit",
        action="store_true",
        help="also run the external CuratorKIT clean and dedup pass",
    )

    evaluate = sub.add_parser("eval", help="score a predictions JSONL with the contract")
    evaluate.add_argument("--data", type=Path, required=True, help="alpaca JSONL with gold output")
    evaluate.add_argument(
        "--predictions",
        type=Path,
        required=True,
        help="JSONL with id and output fields, in any order",
    )
    evaluate.add_argument("--report", type=Path, required=True)
    evaluate.add_argument(
        "--auditkit",
        action="store_true",
        help="also call AuditKIT when that package is installed",
    )

    dpo = sub.add_parser("dpo", help="write planted preference pairs from an alpaca train file")
    dpo.add_argument("--alpaca", type=Path, required=True)
    dpo.add_argument("--out", type=Path, required=True)
    dpo.add_argument("--seed", type=int, default=42)

    compare = sub.add_parser(
        "compare",
        help="score base vs SFT vs DPO, or fill the README from a file",
    )
    compare.add_argument("--results", type=Path, help="existing compare JSON; do not run a model")
    compare.add_argument("--data", type=Path, help="held-out alpaca JSONL")
    compare.add_argument("--sft-adapter", default="")
    compare.add_argument("--dpo-adapter", default="")
    compare.add_argument("--out", type=Path)
    compare.add_argument("--readme", type=Path)

    jobs = sub.add_parser("jobs", help="preview or launch a small HF Job")
    jobs.add_argument("kind", choices=("sft", "dpo", "eval"))
    jobs.add_argument("--namespace", required=True)
    jobs.add_argument("--flavor", default=DEFAULT_FLAVOR)
    jobs.add_argument("--timeout", type=int, default=DEFAULT_TIMEOUT_SECONDS)
    jobs.add_argument("--repo-ref", default="main")
    jobs.add_argument("--launch", action="store_true")

    upload = sub.add_parser("publish", help="preview a private Hub upload")
    upload.add_argument("--namespace", required=True)
    upload.add_argument("--dataset-dir", type=Path)
    upload.add_argument("--sft-dir", type=Path)
    upload.add_argument("--dpo-dir", type=Path)
    upload.add_argument("--eval-dir", type=Path)
    upload.add_argument("--cards-dir", type=Path)
    upload.add_argument(
        "--public",
        action="store_true",
        help="public ungated repos and a public collection; adapters stay CC-BY-NC-4.0",
    )
    upload.add_argument("--execute", action="store_true")

    roi = sub.add_parser(
        "roi",
        help="optional closed-model comparison; missing measurements stay null",
    )
    roi.add_argument("--results", type=Path)
    roi.add_argument("--out", type=Path, required=True)
    roi.add_argument("--closed-model", default=None)
    roi.add_argument("--gpu-hourly-usd", type=float)
    roi.add_argument("--closed-usd-per-million-tokens", type=float)

    args = parser.parse_args(argv)
    if args.command == "generate":
        return _generate(args.n, args.seed, args.out)
    if args.command == "curate":
        return _curate(args.raw, args.out, args.seed, args.curatorkit)
    if args.command == "eval":
        return _eval(args.data, args.predictions, args.report, args.auditkit)
    if args.command == "dpo":
        return _dpo(args.alpaca, args.out, args.seed)
    if args.command == "compare":
        return _compare(args)
    if args.command == "jobs":
        return _jobs(args)
    if args.command == "publish":
        return _publish(args)
    if args.command == "roi":
        return _roi(args)
    parser.error(f"unknown command {args.command}")
    return 2


def _generate(n: int, seed: int, out: Path) -> int:
    records = generate_tickets(n, seed)
    write_jsonl(out, [record.to_dict() for record in records])
    print(f"wrote {len(records)} tickets to {out}")
    return 0


def _curate(raw: Path, out: Path, seed: int, use_curatorkit: bool) -> int:
    records = load_ticket_jsonl(raw)
    local = curate_local(records, out / "local", seed=seed)
    print(json.dumps(local.summary(), indent=2))
    if use_curatorkit:
        summary = curate_with_curatorkit(records, out / "curatorkit", seed=seed)
        print(json.dumps(summary, indent=2))
    return 0


def _eval(data: Path, predictions: Path, report: Path, use_auditkit: bool) -> int:
    records = load_alpaca_jsonl(data)
    by_id = {}
    for row in _read_predictions(predictions):
        by_id[str(row["id"])] = str(row["output"])
    missing = [record.ticket_id for record in records if record.ticket_id not in by_id]
    if missing:
        print(f"predictions missing ids: {missing[:5]}", file=sys.stderr)
        return 2
    outputs = [by_id[record.ticket_id] for record in records]
    card = evaluate_records(records, outputs=outputs)
    payload: dict[str, object] = {
        "measurement": "scored the provided prediction file; not a base-model run",
        "scorecard": card.to_dict(),
    }
    if use_auditkit:
        audit = run_auditkit(records, outputs)
        payload["auditkit"] = audit if audit is not None else {"available": False}
    report.parent.mkdir(parents=True, exist_ok=True)
    report.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "parse_rate": card.parse_rate,
                "schema_valid_rate": card.schema_valid_rate,
                "field_exact_match": card.field_exact_match,
                "report": str(report),
            },
            indent=2,
        )
    )
    return 0


def _dpo(alpaca: Path, out: Path, seed: int) -> int:
    records = load_alpaca_jsonl(alpaca)
    summary = write_dpo_dataset(records, out, seed=seed)
    print(json.dumps(summary, indent=2))
    return 0


def _compare(args: argparse.Namespace) -> int:
    if args.results:
        payload = load_results(args.results)
        if args.readme:
            update_readme(args.readme, payload)
            print(f"updated {args.readme}")
        else:
            print(json.dumps({"n": payload["n"], "measurement": payload["measurement"]}, indent=2))
        return 0
    if not args.data or not args.out or not args.sft_adapter or not args.dpo_adapter:
        print(
            "compare needs --data, --sft-adapter, --dpo-adapter, and --out, "
            "or --results to rewrite the README from an existing file.",
            file=sys.stderr,
        )
        return 2
    records = load_alpaca_jsonl(args.data)
    prompts = [user_content(record.text) for record in records]
    cards = {}
    for name, adapter in (
        ("base", None),
        ("sft", args.sft_adapter),
        ("dpo", args.dpo_adapter),
    ):
        outputs = generate_texts(prompts, model_id=MODEL_ID, adapter=adapter)
        cards[name] = evaluate_records(records, outputs=outputs)
    payload = results_from_scorecards(
        cards,
        data_path=str(args.data),
        sft_adapter=args.sft_adapter,
        dpo_adapter=args.dpo_adapter,
    )
    write_results(payload, args.out)
    print(f"wrote {args.out}")
    if args.readme:
        update_readme(args.readme, payload)
        print(f"updated {args.readme}")
    return 0


def _jobs(args: argparse.Namespace) -> int:
    plan = build_plan(
        args.kind,
        namespace=args.namespace,
        flavor=args.flavor,
        timeout_seconds=args.timeout,
        repo_ref=args.repo_ref,
    )
    print(render_plan(plan))
    if not args.launch:
        print("dry run: not launched")
        return 0
    token = token_from_env()
    from huggingface_hub import run_job

    launch_plan(plan, token=token, run_job=run_job, hardware_lookup=live_hardware_rates)
    return 0


def _publish(args: argparse.Namespace) -> int:
    plan = build_publish_plan(
        namespace=args.namespace,
        dataset_dir=args.dataset_dir,
        sft_dir=args.sft_dir,
        dpo_dir=args.dpo_dir,
        eval_dir=args.eval_dir,
        public=args.public,
        execute=args.execute,
    )
    token = None
    if args.execute:
        token = token_from_env()
    publish(plan, token=token, cards_dir=args.cards_dir)
    return 0


def _roi(args: argparse.Namespace) -> int:
    payload = load_results(args.results) if args.results else None
    report = roi_from_compare(
        payload,
        gpu_hourly_usd=args.gpu_hourly_usd,
        closed_usd_per_million_tokens=args.closed_usd_per_million_tokens,
        closed_model=args.closed_model,
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"status": report["status"], "out": str(args.out)}, indent=2))
    return 0


def _read_predictions(path: Path) -> list[dict[str, object]]:
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        value = json.loads(line)
        if "id" not in value or "output" not in value:
            raise ValueError("each prediction needs id and output")
        rows.append(value)
    return rows

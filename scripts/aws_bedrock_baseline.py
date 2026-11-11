"""Closed-model reference: Amazon Nova on Bedrock, same prompt, same scorers.

Runs the held-out split (seed 42) and the hostile set through one or more
Bedrock model ids with the Converse API, greedy (temperature 0), at most 256
output tokens. Writes predictions, the two scorecards, token counts, and the
cost at the list price passed on the command line. These are reference rows
for the ROI section. Tiny Aya Global stays the only model trained.

    python scripts/aws_bedrock_baseline.py --model us.amazon.nova-micro-v1:0 \
        --out results/aws/bedrock
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from schemasmith.curation.local import assign_splits
from schemasmith.data.generator import generate_tickets
from schemasmith.eval.harness import evaluate_records
from schemasmith.hostile import TICKETS, score_hostile
from schemasmith.hostile.tickets_v2 import TICKETS_V2
from schemasmith.prompts import user_content

# USD per 1M tokens (input, output), us-east-1 on-demand list price.
PRICES = {
    "us.amazon.nova-micro-v1:0": (0.035, 0.14),
    "us.amazon.nova-lite-v1:0": (0.06, 0.24),
    "us.amazon.nova-pro-v1:0": (0.80, 3.20),
    "us.amazon.nova-2-lite-v1:0": (0.33, 2.75),
}


def converse(client, model_id: str, prompt: str) -> tuple[str, int, int, float]:
    for attempt in range(6):
        try:
            start = time.perf_counter()
            resp = client.converse(
                modelId=model_id,
                messages=[{"role": "user", "content": [{"text": prompt}]}],
                inferenceConfig={"maxTokens": 256, "temperature": 0.0},
            )
            secs = time.perf_counter() - start
            parts = resp["output"]["message"]["content"]
            text = "".join(p.get("text", "") for p in parts)
            usage = resp.get("usage", {})
            return text, usage.get("inputTokens", 0), usage.get("outputTokens", 0), secs
        except client.exceptions.ThrottlingException:
            time.sleep(2**attempt)
    raise RuntimeError("throttled six times")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", action="append", required=True)
    ap.add_argument("--out", type=Path, default=Path("results/aws/bedrock"))
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--n", type=int, default=1280)
    args = ap.parse_args()

    import boto3

    client = boto3.client("bedrock-runtime", region_name="us-east-1")
    held_out = assign_splits(generate_tickets(args.n, args.seed), args.seed)["test"]
    held64 = [r for r in held_out if int(r.ticket_id[2:]) < 100320]
    sets = {
        "heldout": [(r.ticket_id, user_content(r.text)) for r in held_out],
        "hostile": [(t.ticket_id, user_content(t.text)) for t in TICKETS],
        "hostile_v2": [(t.ticket_id, user_content(t.text)) for t in TICKETS_V2],
    }
    args.out.mkdir(parents=True, exist_ok=True)
    summary = {}
    for model_id in args.model:
        name = model_id.split(".", 1)[-1].replace(":", "-")
        tin = tout = 0
        block = {"model_id": model_id, "temperature": 0.0, "max_tokens": 256}
        for set_name, rows in sets.items():
            outputs, secs = [], []
            with (args.out / f"predictions_{set_name}_{name}.jsonl").open("w") as fh:
                for rid, prompt in rows:
                    text, i, o, s = converse(client, model_id, prompt)
                    tin += i
                    tout += o
                    outputs.append(text)
                    secs.append(s)
                    fh.write(
                        json.dumps(
                            {
                                "id": rid,
                                "output": text,
                                "seconds": round(s, 3),
                                "input_tokens": i,
                                "output_tokens": o,
                            },
                            ensure_ascii=False,
                        )
                        + "\n"
                    )
            if set_name == "heldout":
                block["heldout"] = evaluate_records(held_out, outputs=outputs).to_dict()
                pos = {r.ticket_id: i for i, r in enumerate(held_out)}
                block["heldout64"] = evaluate_records(
                    held64, outputs=[outputs[pos[r.ticket_id]] for r in held64]
                ).to_dict()
            elif set_name == "hostile":
                block["hostile"] = score_hostile(TICKETS, outputs)
            else:
                block["hostile_v2"] = score_hostile(TICKETS_V2, outputs)
            block[f"{set_name}_median_seconds"] = sorted(secs)[len(secs) // 2]
        pin, pout = PRICES.get(model_id, (None, None))
        block["input_tokens"] = tin
        block["output_tokens"] = tout
        rows_total = sum(len(r) for r in sets.values())
        if pin is not None:
            usd = tin / 1e6 * pin + tout / 1e6 * pout
            block["list_price_usd_per_1m_tokens"] = {"input": pin, "output": pout}
            block["run_cost_usd"] = round(usd, 5)
            block["usd_per_1m_tickets"] = round(usd / rows_total * 1e6, 2)
        summary[name] = block
        h = block["heldout"]
        x = block["hostile"]
        print(
            name,
            "heldout valid",
            h["schema_valid_rate"],
            "fem",
            round(h["field_exact_match"], 4),
            "unsup",
            round(h["unsupported_value_rate"], 4),
            "| hostile valid",
            x["schema_valid"],
            "inj_refused",
            x["injections_refused"],
            "forbidden",
            x["forbidden_value_written"],
            "acc_wrong",
            x["accepted_but_wrong"],
            "| v2",
            block["hostile_v2"]["schema_valid"],
            block["hostile_v2"]["injections_refused"],
            block["hostile_v2"]["accepted_but_wrong"],
            "| $",
            block.get("run_cost_usd"),
        )
    (args.out / "bedrock.json").write_text(
        json.dumps(
            {
                "measurement": "closed-model reference (not trained, not the core model)",
                "data": {
                    "heldout": f"seed-{args.seed} local test split of {args.n} generated tickets",
                    "hostile": "26 hand-written tickets (v1)",
                    "hostile_v2": "30 hand-written tickets (v2, blind)",
                },
                "systems": summary,
            },
            indent=2,
            ensure_ascii=False,
        )
        + "\n"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""Deploy a registered SchemaSmith package to a real-time endpoint, smoke/latency test, delete.

The endpoint bills while it exists, so create -> test -> delete runs in one
process with the delete in `finally`. Endpoint, endpoint config and model are
deleted even if the test fails, and list-endpoints is checked afterwards.

CPU serverless inference is capped at 6 GB memory; Tiny Aya (3.35B params) is
~6.7 GB of bf16 weights before activations, so this uses the smallest GPU that
matches the eval hardware class: one ml.g5.xlarge (A10G 24 GB).

The model is built from the package's own container spec (image +
ModelDataUrl), so the package's approval status is left for a human.

    python scripts/sagemaker/serve/deploy_test.py --package-arn ARN \
        --pools ml.g5.xlarge,ml.g6.xlarge,ml.g5.2xlarge,ml.g4dn.xlarge --heldout 48 \
        --out results/aws/sagemaker-endpoint
"""

from __future__ import annotations

import argparse
import json
import statistics
import time
from pathlib import Path

import boto3

from schemasmith.aws.run import _splits
from schemasmith.eval.harness import evaluate_records
from schemasmith.hostile import score_hostile
from schemasmith.hostile.tickets_v2 import TICKETS_V2
from schemasmith.prompts import user_content

REGION = "us-east-1"
ROLE = "arn:aws:iam::065614567889:role/lexsi-sagemaker-exec"
TAGS = [{"Key": "project", "Value": "lexsi-aws"}]
PRICE_PER_HOUR = {
    "ml.g5.xlarge": 1.408,
    "ml.g6.xlarge": 1.1267,
    "ml.g5.2xlarge": 1.515,
    "ml.g4dn.xlarge": 0.736,
}  # hosting, us-east-1


def pct(xs, q):
    xs = sorted(xs)
    k = (len(xs) - 1) * q
    lo, hi = int(k), min(int(k) + 1, len(xs) - 1)
    return xs[lo] + (xs[hi] - xs[lo]) * (k - lo)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--package-arn", required=True)
    ap.add_argument("--instance", default="ml.g5.xlarge")
    ap.add_argument("--heldout", type=int, default=48)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--profile", default="nightmoose-bots")
    ap.add_argument(
        "--pools", help="comma list of instance types in priority order (InstancePools)"
    )
    ap.add_argument(
        "--attach",
        help="resume an endpoint this script already created (same name for model/config)",
    )
    a = ap.parse_args()
    sess = boto3.Session(profile_name=a.profile, region_name=REGION)
    sm, rt = sess.client("sagemaker"), sess.client("sagemaker-runtime")
    pkg = sm.describe_model_package(ModelPackageName=a.package_arn)
    c = pkg["InferenceSpecification"]["Containers"][0]
    stamp = time.strftime("%m%d-%H%M")
    name = a.attach or f"lexsi-schemasmith-{stamp}"
    a.out.mkdir(parents=True, exist_ok=True)
    log: dict = {"package": a.package_arn, "instance": a.instance, "name": name, "events": {}}
    ev = log["events"]

    def mark(k):
        ev[k] = time.strftime("%Y-%m-%d %H:%M:%S %Z")
        print(k, ev[k], flush=True)

    t_create = time.time()
    try:
        if a.attach:
            t_create = sm.describe_endpoint(EndpointName=name)["CreationTime"].timestamp()
            mark("attached")
        else:
            sm.create_model(
                ModelName=name,
                ExecutionRoleArn=ROLE,
                PrimaryContainer={
                    "Image": c["Image"],
                    "ModelDataUrl": c["ModelDataUrl"],
                    "Environment": {
                        **c.get("Environment", {}),
                        "TS_DEFAULT_RESPONSE_TIMEOUT": "300",
                    },
                },
                Tags=TAGS + [{"Key": "model-package", "Value": a.package_arn.split("/", 1)[1]}],
            )
            variant = {
                "VariantName": "main",
                "ModelName": name,
                "InitialInstanceCount": 1,
                "ContainerStartupHealthCheckTimeoutInSeconds": 1200,
                "ModelDataDownloadTimeoutInSeconds": 1200,
            }
            if a.pools:
                # Priority-ordered fallback when one GPU type has no capacity.
                variant["InstancePools"] = [
                    {"InstanceType": t, "Priority": i + 1} for i, t in enumerate(a.pools.split(","))
                ]
            else:
                variant["InstanceType"] = a.instance
            log["variant_request"] = variant
            sm.create_endpoint_config(
                EndpointConfigName=name,
                ProductionVariants=[variant],
                Tags=TAGS,
            )
            sm.create_endpoint(EndpointName=name, EndpointConfigName=name, Tags=TAGS)
            mark("endpoint_create")
        while True:
            st = sm.describe_endpoint(EndpointName=name)
            if st["EndpointStatus"] in ("InService", "Failed"):
                break
            if time.time() - t_create > 35 * 60:
                raise SystemExit("endpoint not InService after 35 min")
            time.sleep(20)
        if st["EndpointStatus"] == "Failed":
            log["failure"] = st.get("FailureReason")
            raise SystemExit(f"endpoint failed: {st.get('FailureReason')}")
        mark("in_service")
        log["production_variants"] = json.loads(
            json.dumps(st.get("ProductionVariants"), default=str)
        )
        print(json.dumps(log["production_variants"], indent=1))
        log["startup_seconds"] = round(time.time() - t_create, 1)

        splits, _ = _splits(False)
        held = splits["test"][: a.heldout]
        rows = [("heldout", r.ticket_id, user_content(r.text)) for r in held]
        rows += [("hostile_v2", t.ticket_id, user_content(t.text)) for t in TICKETS_V2]
        # One warm-up call (first request pays CUDA graph / kernel init).
        rt.invoke_endpoint(
            EndpointName=name,
            ContentType="application/json",
            Body=json.dumps({"prompt": rows[0][2]}),
        )
        preds = []
        for s, rid, prompt in rows:
            t = time.perf_counter()
            r = rt.invoke_endpoint(
                EndpointName=name,
                ContentType="application/json",
                Body=json.dumps({"prompt": prompt}),
            )
            body = json.loads(r["Body"].read())
            preds.append(
                {
                    "set": s,
                    "id": rid,
                    "output": body["output"],
                    "client_seconds": round(time.perf_counter() - t, 4),
                    "gen_seconds": body["gen_seconds"],
                    "new_tokens": body["new_tokens"],
                }
            )
        mark("tests_done")
        with (a.out / "predictions.jsonl").open("w", encoding="utf-8") as fh:
            for p in preds:
                fh.write(json.dumps(p, ensure_ascii=False) + "\n")
        hp = [p for p in preds if p["set"] == "heldout"]
        vp = [p for p in preds if p["set"] == "hostile_v2"]
        hm = evaluate_records(held, outputs=[p["output"] for p in hp]).to_dict()
        hv = score_hostile(TICKETS_V2, [p["output"] for p in vp])
        lat = [p["client_seconds"] for p in preds]
        hlat = [p["client_seconds"] for p in hp]
        log["results"] = {
            "heldout_n": len(hp),
            "heldout": {
                k: hm[k]
                for k in (
                    "n_parsed",
                    "n_contract_valid",
                    "contract_valid_rate",
                    "schema_valid_rate",
                    "field_exact_match",
                    "unsupported_value_rate",
                )
            },
            "hostile_v2": {
                k: hv[k]
                for k in (
                    "n",
                    "schema_valid",
                    "injection_tickets",
                    "injections_refused",
                    "forbidden_value_written",
                    "accepted_but_wrong",
                )
            },
            "latency_all_calls": {
                "n": len(lat),
                "p50_s": round(pct(lat, 0.5), 3),
                "p95_s": round(pct(lat, 0.95), 3),
                "max_s": round(max(lat), 3),
            },
            "latency_heldout": {
                "n": len(hlat),
                "p50_s": round(pct(hlat, 0.5), 3),
                "p95_s": round(pct(hlat, 0.95), 3),
                "mean_new_tokens": round(statistics.mean(p["new_tokens"] for p in hp), 1),
            },
        }
        print(json.dumps(log["results"], indent=2))
    finally:
        for fn, kw in (
            (sm.delete_endpoint, {"EndpointName": name}),
            (sm.delete_endpoint_config, {"EndpointConfigName": name}),
            (sm.delete_model, {"ModelName": name}),
        ):
            try:
                fn(**kw)
            except Exception as e:  # noqa: BLE001
                print("delete:", fn.__name__, type(e).__name__, str(e)[:120])
        mark("deleted")
        alive = sm.list_endpoints()["Endpoints"]
        log["endpoints_after_delete"] = [(e["EndpointName"], e["EndpointStatus"]) for e in alive]
        hours = (time.time() - t_create) / 3600
        log["billable_hours_upper_bound"] = round(hours, 4)
        log["est_cost_usd"] = round(
            hours * PRICE_PER_HOUR.get(log.get("served_instance", a.instance), 1.515), 3
        )
        (a.out / "run.json").write_text(json.dumps(log, indent=2) + "\n")
        print(
            "endpoints after delete:", log["endpoints_after_delete"], "est $", log["est_cost_usd"]
        )


if __name__ == "__main__":
    main()

"""Register the Nov 10 v2 seed-1 adapter (lexsi-v2-s1-1110-1722) in lexsi-schemasmith.

Model package versions cannot carry tags; the group is tagged project=lexsi-aws.

This is the adapter behind the published v2 numbers (best seed: 99.44% held-out
field match, 10/10 blind injections refused). Its metrics come from the same-GPU
eval job lexsi-eval-1110-2044 (results/aws/sagemaker/eval-v2-s1). The serving
artifact is the training output plus code/ (inference.py, requirements.txt);
TensorBoard logs and the DPO reference copy are dropped.

    python scripts/sagemaker/pipeline/register_existing.py
"""

from __future__ import annotations

import argparse
import json
import shutil
import tarfile
import tempfile
from pathlib import Path

import boto3
from pipeline import BUCKET, GROUP, PREFIX, REGION, SERVE_IMAGE

ROOT = Path(__file__).resolve().parents[3]
SRC_KEY = "out/lexsi-v2-s1-1110-1722/output/model.tar.gz"
EVAL = ROOT / "results/aws/sagemaker/eval-v2-s1/v2-s1/eval/eval.json"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--rev", default="", help="suffix for a rebuilt serving artifact, e.g. -r2")
    rev = ap.parse_args().rev
    DST_KEY = f"{PREFIX}/registered/v2-s1-1110-1722{rev}/model.tar.gz"  # noqa: N806
    MQ_KEY = f"{PREFIX}/registered/v2-s1-1110-1722{rev}/model_quality.json"  # noqa: N806
    sess = boto3.Session(profile_name="nightmoose-bots", region_name=REGION)
    s3, sm = sess.client("s3"), sess.client("sagemaker")
    tmp = Path(tempfile.mkdtemp())
    s3.download_file(BUCKET, SRC_KEY, str(tmp / "src.tgz"))
    stage = tmp / "stage"
    with tarfile.open(tmp / "src.tgz") as t:
        t.extractall(stage, filter="data")
    for d in ("sft-adapter", "dpo-adapter"):
        for junk in ("tensorboard", "ref"):
            shutil.rmtree(stage / d / junk, ignore_errors=True)
    shutil.copytree(
        ROOT / "scripts/sagemaker/serve",
        stage / "code",
        ignore=shutil.ignore_patterns("__pycache__"),
    )
    with tarfile.open(tmp / "model.tar.gz", "w:gz") as t:
        for item in sorted(stage.iterdir()):
            t.add(item, arcname=item.name)
    s3.upload_file(
        str(tmp / "model.tar.gz"), BUCKET, DST_KEY, ExtraArgs={"Tagging": "project=lexsi-aws"}
    )
    dpo = json.loads(EVAL.read_text())["systems"]["dpo"]
    h, v2 = dpo["heldout"], dpo["hostile_v2"]
    metrics = {
        "contract_valid_rate": h["contract_valid_rate"],
        "field_exact_match": h["field_exact_match"],
        "unsupported_value_rate": h["unsupported_value_rate"],
        "hostile_v2_injections_refused": v2["injections_refused"],
        "hostile_v2_forbidden_value_written": v2["forbidden_value_written"],
    }
    mq = {
        "extraction_metrics": {
            k: {"value": v, "standard_deviation": "NaN"} for k, v in metrics.items()
        }
    }
    s3.put_object(
        Bucket=BUCKET,
        Key=MQ_KEY,
        Body=json.dumps(mq, indent=2).encode(),
        ContentType="application/json",
        Tagging="project=lexsi-aws",
    )
    r = sm.create_model_package(
        ModelPackageGroupName=GROUP,
        ModelPackageDescription="SchemaSmith Tiny Aya LoRA SFT+DPO, v2 data seed 1 "
        "(training job lexsi-v2-s1-1110-1722, eval lexsi-eval-1110-2044)"
        + (f" handler {rev.strip('-')}: GPU bf16/fp16 or CPU" if rev else ""),
        InferenceSpecification={
            "Containers": [
                {
                    "Image": SERVE_IMAGE,
                    "ModelDataUrl": f"s3://{BUCKET}/{DST_KEY}",
                    "Environment": {
                        "SAGEMAKER_PROGRAM": "inference.py",
                        "SAGEMAKER_SUBMIT_DIRECTORY": "/opt/ml/model/code",
                    },
                }
            ],
            "SupportedContentTypes": ["application/json"],
            "SupportedResponseMIMETypes": ["application/json"],
            "SupportedRealtimeInferenceInstanceTypes": ["ml.g5.xlarge", "ml.g6.xlarge"],
        },
        ModelApprovalStatus="PendingManualApproval",
        ModelMetrics={
            "ModelQuality": {
                "Statistics": {
                    "ContentType": "application/json",
                    "S3Uri": f"s3://{BUCKET}/{MQ_KEY}",
                }
            }
        },
        CustomerMetadataProperties={
            "source": "lexsi-v2-s1-1110-1722",
            "eval_job": "lexsi-eval-1110-2044",
            **{k: str(round(v, 6)) for k, v in metrics.items()},
        },
    )
    print(r["ModelPackageArn"])


if __name__ == "__main__":
    main()

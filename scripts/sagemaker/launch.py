"""Create SageMaker training jobs for SchemaSmith v2 from the box.

The box cannot reach S3, so code comes from the private Hub source repo and
results come back through the job log (see job.sh). The HF token lives in
Secrets Manager as AmazonSageMaker-lexsi-hf and is read inside the job by the
execution role; it is not put in the job definition.

    python scripts/sagemaker/launch.py upload
    python scripts/sagemaker/launch.py train --tag v2-s1 --seed 1 --type ml.g6.xlarge --spot
    python scripts/sagemaker/launch.py eval --channels v2-s1=JOBNAME ... --type ml.g6.xlarge
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import boto3

REGION = "us-east-1"
ROLE = "arn:aws:iam::065614567889:role/lexsi-sagemaker-exec"
BUCKET = "sagemaker-lexsi-065614567889"
IMAGE = (
    "763104351884.dkr.ecr.us-east-1.amazonaws.com/pytorch-training:"
    "2.8.0-gpu-py312-cu129-ubuntu22.04-sagemaker"
)
TAGS = [
    {"Key": "Project", "Value": "nightmoose-bots"},
    {"Key": "Owner", "Value": "chief"},
    {"Key": "Workstream", "Value": "lexsi"},
]
SRC = "nightmoose/schemasmith-src"
ROOT = Path(__file__).resolve().parents[2]
B1 = (
    "set -e; pip -q install --root-user-action=ignore huggingface_hub; "
    "export HF_TOKEN=$(python3 -c \"import boto3,json;print(json.loads(boto3.client("
    "'secretsmanager',region_name='us-east-1').get_secret_value(SecretId="
    "'AmazonSageMaker-lexsi-hf')['SecretString'])['HF_TOKEN'])\"); "
)
B2 = (
    "python3 -c \"from huggingface_hub import hf_hub_download as d;"
    "d('nightmoose/schemasmith-src','aws/job.sh',local_dir='/tmp/j')\"; "
    "bash /tmp/j/aws/job.sh"
)
sm = boto3.client("sagemaker", region_name=REGION)


def upload() -> None:
    import os

    from huggingface_hub import HfApi

    api = HfApi(token=os.environ["HF_TOKEN"])
    api.create_repo(SRC, private=True, exist_ok=True)
    for local, remote in (
        (ROOT / "dist" / "schemasmith-0.2.0-py3-none-any.whl", "aws/schemasmith-0.2.0-py3-none-any.whl"),
        (ROOT / "scripts" / "sagemaker" / "job.sh", "aws/job.sh"),
    ):
        api.upload_file(path_or_fileobj=str(local), path_in_repo=remote, repo_id=SRC)
        print("uploaded", remote)


def create(name: str, itype: str, spot: bool, env: dict, channels: dict | None, hours: float):
    secs = int(hours * 3600)
    kw = dict(
        TrainingJobName=name,
        RoleArn=ROLE,
        AlgorithmSpecification={
            "TrainingImage": IMAGE,
            "TrainingInputMode": "File",
            "ContainerEntrypoint": ["bash", "-c", 'eval "$B1$B2"'],
        },
        OutputDataConfig={"S3OutputPath": f"s3://{BUCKET}/out/"},
        ResourceConfig={"InstanceType": itype, "InstanceCount": 1, "VolumeSizeInGB": 100},
        StoppingCondition={"MaxRuntimeInSeconds": secs},
        Environment={"B1": B1, "B2": B2, **env},
        Tags=TAGS,
    )
    if spot:
        kw["EnableManagedSpotTraining"] = True
        kw["StoppingCondition"]["MaxWaitTimeInSeconds"] = secs + 3600
    if channels:
        kw["InputDataConfig"] = [
            {
                "ChannelName": ch,
                "DataSource": {
                    "S3DataSource": {
                        "S3DataType": "S3Prefix",
                        "S3Uri": f"s3://{BUCKET}/out/{job}/output/",
                        "S3DataDistributionType": "FullyReplicated",
                    }
                },
            }
            for ch, job in channels.items()
        ]
    sm.create_training_job(**kw)
    print("created", name)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=("upload", "train", "eval"))
    ap.add_argument("--tag")
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--no-adv", action="store_true")
    ap.add_argument("--type", default="ml.g6.xlarge")
    ap.add_argument("--spot", action="store_true")
    ap.add_argument("--hours", type=float, default=2.5)
    ap.add_argument("--channels", nargs="*", default=[])
    a = ap.parse_args()
    stamp = time.strftime("%m%d-%H%M")
    if a.cmd == "upload":
        upload()
    elif a.cmd == "train":
        env = {"JOB_KIND": "train", "TAG": a.tag, "SEED": str(a.seed)}
        if a.no_adv:
            env["NO_ADV"] = "1"
        create(f"lexsi-{a.tag}-{stamp}", a.type, a.spot, env, None, a.hours)
    else:
        chans = dict(c.split("=", 1) for c in a.channels)
        create(f"lexsi-eval-{stamp}", a.type, a.spot, {"JOB_KIND": "eval"}, chans, a.hours)


if __name__ == "__main__":
    main()

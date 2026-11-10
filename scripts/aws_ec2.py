"""Run SchemaSmith stages on one EC2 GPU instance from the box, over SSH.

    python scripts/aws_ec2.py launch --type g6.xlarge [--spot]   -> prints instance id
    python scripts/aws_ec2.py setup  --id i-...                  -> install wheel + pins
    python scripts/aws_ec2.py run    --id i-... --name job -- CMD -> nohup CMD, log in ~/logs
    python scripts/aws_ec2.py tail   --id i-... --name job
    python scripts/aws_ec2.py fetch  --id i-... --remote PATH --local DIR
    python scripts/aws_ec2.py terminate --id i-...

AMI: AWS Deep Learning OSS Nvidia Driver AMI, PyTorch 2.13, Ubuntu 26.04
(Amazon-owned, no Marketplace subscription). Every resource is tagged
Project=nightmoose-bots, Owner=chief, Workstream=lexsi. HF_TOKEN is read from
the box environment and written to a 0600 file on the instance over SSH; it
is never placed in user data or tags, and the instance is terminated after.
"""

from __future__ import annotations

import argparse
import os
import stat
import sys
import time
from pathlib import Path

import boto3
import paramiko

REGION = "us-east-1"
AMI_PARAM = (
    "/aws/service/deeplearning/ami/x86_64/"
    "oss-nvidia-driver-gpu-pytorch-2.13-ubuntu-26.04/latest/ami-id"
)
KEY_NAME = "nightmoose-lexsi"
KEY_FILE = Path.home() / ".ssh/lexsi/id"
SG = os.environ.get("LEXSI_SG", "sg-07de9a850332b0a0b")
TAGS = [
    {"Key": "Project", "Value": "nightmoose-bots"},
    {"Key": "Owner", "Value": "chief"},
    {"Key": "Workstream", "Value": "lexsi"},
]
WHEEL = "schemasmith-0.2.0-py3-none-any.whl"
PINS = "curatorkit==0.1.23 auditkit==1.1.6 aligntune==0.1.32"

ec2 = boto3.client("ec2", region_name=REGION)


def launch(itype: str, spot: bool, disk: int) -> str:
    ami = boto3.client("ssm", region_name=REGION).get_parameter(Name=AMI_PARAM)["Parameter"][
        "Value"
    ]
    kw = dict(
        ImageId=ami,
        InstanceType=itype,
        KeyName=KEY_NAME,
        MinCount=1,
        MaxCount=1,
        SecurityGroupIds=[SG],
        BlockDeviceMappings=[
            {
                "DeviceName": "/dev/sda1",
                "Ebs": {"VolumeSize": disk, "VolumeType": "gp3", "DeleteOnTermination": True},
            }
        ],
        TagSpecifications=[
            {"ResourceType": t, "Tags": TAGS + [{"Key": "Name", "Value": f"lexsi-{itype}"}]}
            for t in ("instance", "volume")
        ],
        # Stop the bill even if the box loses track: shutdown means terminate.
        InstanceInitiatedShutdownBehavior="terminate",
        UserData="#!/bin/bash\nshutdown -h +720\n",
    )
    if spot:
        kw["InstanceMarketOptions"] = {
            "MarketType": "spot",
            "SpotOptions": {
                "SpotInstanceType": "one-time",
                "InstanceInterruptionBehavior": "terminate",
            },
        }
        kw["TagSpecifications"].append({"ResourceType": "spot-instances-request", "Tags": TAGS})
    iid = ec2.run_instances(**kw)["Instances"][0]["InstanceId"]
    print(iid, flush=True)
    ec2.get_waiter("instance_running").wait(InstanceIds=[iid])
    print("running", ip(iid), flush=True)
    return iid


def ip(iid: str) -> str:
    r = ec2.describe_instances(InstanceIds=[iid])["Reservations"][0]["Instances"][0]
    return r.get("PublicIpAddress", "")


def ssh(iid: str, tries: int = 40) -> paramiko.SSHClient:
    key = paramiko.Ed25519Key.from_private_key_file(str(KEY_FILE))
    host = ip(iid)
    for _ in range(tries):
        try:
            c = paramiko.SSHClient()
            c.set_missing_host_key_policy(paramiko.AutoAddPolicy())
            c.connect(
                host,
                username="ubuntu",
                pkey=key,
                timeout=15,
                banner_timeout=30,
                allow_agent=False,
                look_for_keys=False,
            )
            c.get_transport().set_keepalive(30)
            return c
        except Exception as exc:  # noqa: BLE001
            last = exc
            time.sleep(10)
    raise RuntimeError(f"ssh failed: {last}")


def sh(c: paramiko.SSHClient, cmd: str, quiet: bool = False) -> tuple[int, str]:
    _, out, err = c.exec_command(cmd, get_pty=False)
    text = out.read().decode() + err.read().decode()
    code = out.channel.recv_exit_status()
    if not quiet:
        print(text[-4000:], flush=True)
    return code, text


def setup(iid: str) -> None:
    c = ssh(iid)
    sftp = c.open_sftp()
    sftp.put(str(Path(__file__).resolve().parents[1] / "dist" / WHEEL), f"/home/ubuntu/{WHEEL}")
    token = os.environ.get("HF_TOKEN", "")
    if not token:
        sys.exit("HF_TOKEN not in the box environment")
    with sftp.open("/home/ubuntu/.hf_token", "w") as fh:
        fh.write(token)
    sftp.chmod("/home/ubuntu/.hf_token", stat.S_IRUSR | stat.S_IWUSR)
    sftp.close()
    script = f"""set -e
mkdir -p ~/logs
W=/opt/dlami/nvme; [ -d $W ] && [ -w $W ] || W=$HOME; mkdir -p $W/w; ln -sfn $W/w ~/w
nvidia-smi --query-gpu=name,memory.total --format=csv || true
# Plain venv: the system dist-packages carry a broken rpds for jsonschema.
rm -rf ~/venv; python3 -m venv ~/venv
. ~/venv/bin/activate
echo "torch==2.13.0" > /tmp/c.txt
pip install -q -c /tmp/c.txt ~/{WHEEL} {PINS} huggingface_hub 2>&1 | tail -3
pip list 2>/dev/null | grep -iE '^(torch|transformers|trl|peft|bitsandbytes|aligntune|accelerate) '
echo SETUP_DONE
"""
    with c.open_sftp().open("/home/ubuntu/setup.sh", "w") as fh:
        fh.write(script)
    code, _ = sh(
        c, "mkdir -p ~/logs; bash ~/setup.sh > ~/logs/setup.log 2>&1; tail -20 ~/logs/setup.log"
    )
    c.close()


def run(iid: str, name: str, cmd: str) -> None:
    c = ssh(iid)
    wrapped = (
        f"cd ~ && . ~/venv/bin/activate && export HF_TOKEN=$(cat ~/.hf_token) "
        f"WORK=$HOME/w PYTHONUNBUFFERED=1 HF_HUB_ENABLE_HF_TRANSFER=0 && "
        f"nohup bash -c '{cmd}; echo EXIT_CODE $?' > ~/logs/{name}.log 2>&1 &"
    )
    sh(c, wrapped, quiet=True)
    print("started", name)
    c.close()


def tail(iid: str, name: str, n: int = 30) -> None:
    c = ssh(iid)
    sh(
        c,
        f"tail -n {n} ~/logs/{name}.log | cut -c1-400; nvidia-smi --query-gpu=utilization.gpu,"
        "memory.used --format=csv,noheader 2>/dev/null",
    )
    c.close()


def fetch(iid: str, remote: str, local: str) -> None:
    c = ssh(iid)
    sh(
        c,
        f"cd {remote}/.. && tar czf /tmp/fetch.tgz --exclude='*.safetensors' "
        f"--exclude='checkpoint-*' --exclude='sft-merged' $(basename {remote})",
        quiet=True,
    )
    Path(local).mkdir(parents=True, exist_ok=True)
    c.open_sftp().get("/tmp/fetch.tgz", f"{local}/fetch.tgz")
    c.close()
    print("fetched", f"{local}/fetch.tgz")


def fetch_file(iid: str, remote: str, local: str) -> None:
    c = ssh(iid)
    c.open_sftp().get(remote, local)
    c.close()


def terminate(iid: str) -> None:
    ec2.terminate_instances(InstanceIds=[iid])
    ec2.get_waiter("instance_terminated").wait(InstanceIds=[iid])
    print("terminated", iid)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd")
    ap.add_argument("--type", default="g6.xlarge")
    ap.add_argument("--spot", action="store_true")
    ap.add_argument("--disk", type=int, default=150)
    ap.add_argument("--id")
    ap.add_argument("--name", default="job")
    ap.add_argument("--remote")
    ap.add_argument("--local")
    ap.add_argument("-n", type=int, default=30)
    ap.add_argument("rest", nargs="*")
    a = ap.parse_args()
    if a.cmd == "launch":
        launch(a.type, a.spot, a.disk)
    elif a.cmd == "setup":
        setup(a.id)
    elif a.cmd == "run":
        run(a.id, a.name, " ".join(a.rest))
    elif a.cmd == "tail":
        tail(a.id, a.name, a.n)
    elif a.cmd == "fetch":
        fetch(a.id, a.remote, a.local)
    elif a.cmd == "fetchfile":
        fetch_file(a.id, a.remote, a.local)
    elif a.cmd == "sh":
        c = ssh(a.id)
        sh(c, " ".join(a.rest))
    elif a.cmd == "terminate":
        terminate(a.id)


if __name__ == "__main__":
    main()

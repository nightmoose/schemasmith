"""Launch one Stage 2 job on Hugging Face Jobs (one L4, hard timeout).

  export HF_TOKEN=...            # a token that can run Jobs in NAMESPACE
  export SRC_REPO=<hf-user>/schemasmith-src   # your own wheel, see docs/reproducibility.md
  python launch.py ablation 1    # v1-data ablation, seed 1..3
  python launch.py retention 900 # Belebele retention, base + v1 adapters, N per language
  python launch.py retention_v2 1
  python launch.py safety        # XSTest, base + v1 + v2 adapters

Prints the cost ceiling and refuses anything above $2. Results come back in the
job log as base64 chunks (B64CHUNK lines) followed by a sha256 (B64END).
Paths are relative to this file.
"""
import base64
import json
import os
import sys
import time
from pathlib import Path

from huggingface_hub import list_jobs_hardware, run_job

HERE = Path(__file__).resolve().parent
TOK = os.environ["HF_TOKEN"]
NS = os.environ.get("NAMESPACE") or os.environ["SRC_REPO"].split("/")[0]
IMAGE = "pytorch/pytorch:2.13.0-cuda12.6-cudnn9-runtime"
kind = sys.argv[1]
flavor = "l4x1"
rate = {h.name: h.unit_cost_usd for h in list_jobs_hardware()}[flavor] * 60
base_env = {"SRC_REPO": os.environ["SRC_REPO"], "PIP_BREAK_SYSTEM_PACKAGES": "1", "PYTHONUNBUFFERED": "1"}


def b64(name):
    return base64.b64encode((HERE / name).read_bytes()).decode()


if kind == "ablation":
    seed = sys.argv[2]; timeout = 7200
    cmd = ["bash", "-lc", (HERE / "ablation.sh").read_text()]
    env = {**base_env, "SEED": seed}; name = f"lexsi-v1data-s{seed}"
elif kind == "retention":
    timeout = 3600
    cmd = ["bash", "-lc", (HERE / "retention.sh").read_text()]
    env = {**base_env, "RET_PY_B64": b64("retention.py"),
           "N_PER_LANG": sys.argv[2] if len(sys.argv) > 2 else "200"}
    name = "lexsi-retention-belebele"
elif kind == "retention_v2":
    seed = sys.argv[2]; timeout = 3600
    cmd = ["bash", "-lc", (HERE / "generic.sh").read_text()]
    env = {**base_env, "RET_PY_B64": b64("retention_v2.py"), "SEED": seed, "N_PER_LANG": "900"}
    name = f"lexsi-retention-v2-s{seed}"
elif kind == "safety":
    timeout = 5400
    cmd = ["bash", "-lc", (HERE / "generic.sh").read_text()]
    env = {**base_env, "RET_PY_B64": b64("xstest.py")}; name = "lexsi-safety-xstest"
else:
    sys.exit(f"unknown kind {kind}")
ceiling = rate * timeout / 3600
assert ceiling <= 2.0, ceiling
j = run_job(image=IMAGE, command=cmd, env=env, secrets={"HF_TOKEN": TOK}, flavor=flavor,
            timeout=timeout, namespace=NS, token=TOK)
rec = {"job_id": j.id, "name": name, "kind": kind, "flavor": flavor, "timeout_s": timeout,
       "rate_usd_hr": rate, "ceiling_usd": round(ceiling, 2), "launched": time.strftime("%Y-%m-%d %H:%M:%S %Z")}
print(json.dumps(rec))

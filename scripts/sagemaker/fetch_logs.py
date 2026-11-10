"""Rebuild the eval tarball a SageMaker job printed to CloudWatch Logs (box cannot read S3).

    python scripts/sagemaker/fetch_logs.py JOBNAME OUTDIR
"""

import base64
import hashlib
import sys
import tarfile
from pathlib import Path

import boto3

job, out = sys.argv[1], Path(sys.argv[2])
logs = boto3.client("logs", region_name="us-east-1")
group = "/aws/sagemaker/TrainingJobs"
streams = logs.describe_log_streams(logGroupName=group, logStreamNamePrefix=job)["logStreams"]
chunks: dict[int, str] = {}
digest = None
for s in streams:
    kw = {"logGroupName": group, "logStreamName": s["logStreamName"], "startFromHead": True}
    token = None
    while True:
        if token:
            kw["nextToken"] = token
        r = logs.get_log_events(**kw)
        for e in r["events"]:
            m = e["message"]
            if m.startswith("B64CHUNK evals "):
                _, _, n, data = m.split(" ", 3)
                chunks[int(n)] = data.strip()
            elif m.startswith("B64END evals "):
                digest = m.split()[2]
        if r["nextForwardToken"] == token:
            break
        token = r["nextForwardToken"]
blob = base64.b64decode("".join(chunks[i] for i in sorted(chunks)))
assert hashlib.sha256(blob).hexdigest() == digest, "checksum mismatch"
out.mkdir(parents=True, exist_ok=True)
(out / "evals.tgz").write_bytes(blob)
with tarfile.open(out / "evals.tgz") as t:
    t.extractall(out, filter="data")
print("ok", len(chunks), "chunks", len(blob), "bytes")

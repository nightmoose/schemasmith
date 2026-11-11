# SageMaker training for SchemaSmith v2

The box cannot reach the training bucket (VPC endpoint policy blocks
`s3:ListBucket` / direct transfer), so jobs pull `aws/job.sh` and the
wheel from the private Hub repo `nightmoose/schemasmith-src`. The
SageMaker execution role reads `AmazonSageMaker-lexsi-hf` from Secrets
Manager inside the job. Eval results come back as `B64CHUNK` lines in
CloudWatch Logs (`fetch_logs.py`).

Account limits observed Nov 10, 2026 in us-east-1: 1 training instance
per type (`ml.g6.xlarge`, `ml.g6.2xlarge`, `ml.g5.xlarge`,
`ml.g5.2xlarge`). Launch one job of each type, not four of the cheap
`ml.g6.xlarge`.

    python scripts/sagemaker/launch.py upload
    python scripts/sagemaker/launch.py train --tag v2-s1 --seed 1 --type ml.g6.xlarge
    python scripts/sagemaker/launch.py train --tag v1data-s1 --seed 1 --no-adv --type ml.g5.xlarge
    python scripts/sagemaker/launch.py eval --channels v2-s1=JOB v2-s2=JOB v2-s3=JOB v1data-s1=JOB --type ml.g6.2xlarge
    python scripts/sagemaker/fetch_logs.py JOB OUTDIR

## Failures and the cuDNN fix

- `lexsi-v1data-s1-1110-1628` and `lexsi-v2-s3-1110-1628` failed immediately:
  pip-installing `torch==2.13.0` over the DLC left a mismatched
  `torchaudio`. `job.sh` now uninstalls `torchaudio` and `torchvision`.
- `lexsi-v1data-s1-1110-1639` finished SFT (2 epochs, 220 steps, 1,207 s
  on `ml.g5.xlarge`) then died at DPO trainer init: PyTorch 2.13 was
  built against cuDNN 9.20.0 but `LD_LIBRARY_PATH` still pointed at the
  DLC's 9.10.2. `job.sh` now prepends the pip torch / nvidia lib dirs
  and preflights `torch.backends.cudnn.version()` before training.

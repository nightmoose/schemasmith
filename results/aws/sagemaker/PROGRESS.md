# SageMaker Lexsi v2 training — progress notes

Last updated: 2026-11-10 7:20 PM ET
Checkout: /workspace/lexsi/run/schemasmith (branch aws-runs, PR #5)
https://github.com/nightmoose/schemasmith/pull/5
Do not touch branch aws-guardrails (PR #6). Do not relaunch completed train jobs.

## Still running (do not relaunch these)
- **lexsi-v2-s1-1110-1722** (ml.g6.2xlarge): DPO late stage at 7:19 PM (lr 2.5e-06). TrainingStart 6:02:10 PM, MaxRuntime 9000s → hard stop ~8:32 PM ET. When Complete, save train_log from CloudWatch (`TRAIN_DONE` line) to v2-s1-train_log.json, then eval it (channel v2-s1=lexsi-v2-s1-1110-1722) after lexsi-eval-1110-1920 frees a slot, or add it to a third eval.
- **lexsi-eval-1110-1920** (ml.g5.xlarge, launched 7:20 PM): scores v1 base/SFT/DPO plus v2-s2 and v2-s3. job.sh and launch.py were patched to skip SageMaker `*-manifest` channels and to use the output prefix as S3Uri. When Complete: `python scripts/sagemaker/fetch_logs.py lexsi-eval-1110-1920 results/aws/sagemaker/eval-v2` and merge with eval-partial/eval_from_logs.json (v1 + v1data-s1 already scored).

## Completed
| Job | Instance | Billable s | Cost | End (ET) |
|---|---|---|---|---|
| lexsi-v1data-s1-1110-1722 | ml.g5.xlarge | 3074 | $1.202 | 6:15:28 PM |
| lexsi-v2-s2-1110-1722 | ml.g5.2xlarge | 4142 | $1.743 | 6:38:16 PM |
| lexsi-v2-s3-1110-1726 | ml.g6.xlarge | 4956 | $1.551 | 6:59:22 PM |
| lexsi-eval-1110-1858 (Failed; got v1 + ablation scores) | ml.g5.xlarge | 1159 | $0.453 | 7:19:30 PM |

Ablation DPO vs this-GPU v1 DPO: held-out FEM 95.98% vs 97.66%; unsupported 0% vs 0.71%; hostile v2 refused 6/10 vs 5/10.

## After everything is terminal
1. Fetch eval-1920 tarball; score v2-s1 when that job completes.
2. Mean ± spread across v2 seeds vs ablation vs v1 DPO.
3. Pull nmb-chief, update cost-log + SKILLS-LOG with final billable seconds (a first entry may already be present).
4. Commit remaining results to aws-runs (not main).
5. Teardown: no running training jobs, endpoints, notebooks, or EC2. S3 artifacts may stay; box cannot list the bucket (VPC endpoint). Confirm with `list-training-jobs`, `list-endpoints`, `list-notebook-instances`, `describe-instances`.

## Spend
Settled at 7:20 PM ET: **$7.10**. Plus still-running s1 (~$1.6 so far) and eval-1920. Cap $40.
Prices: g5.xlarge $1.408, g5.2xlarge $1.515, g6.xlarge $1.127, g6.2xlarge $1.222 /hr.

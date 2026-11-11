# SageMaker training (Nov 10, 2026): Lexsi v2 seeds and v1-data ablation

Status: **three of four training jobs completed successfully.** Seed 1
(`lexsi-v2-s1-1110-1722`) was still in DPO at 7:20 PM ET. Eval of the
ablation (and a same-GPU v1 re-score) is in the table below. Eval of
v2 seeds 2 and 3 is in flight as `lexsi-eval-1110-1920` after the first
eval job hit a SageMaker `*-manifest` channel and exited before scoring
`v2-s2`.

Nothing here replaces the v1 CPU / Nova tables in `results/aws/README.md`
until the remaining jobs and the v2 eval finish.

## Training jobs that completed

SFT was 2 epochs, DPO 1 epoch, LoRA r=16, tiny-aya-global, on-demand
us-east-1. `job.sh` preflight: `torch 2.13.0+cu130`, cuDNN 92000.

| Job | Data / seed | Instance | Billable s | Cost | SFT s | DPO s | Ended (ET) |
| --- | --- | --- | --- | --- | --- | --- | --- |
| lexsi-v1data-s1-1110-1722 | v1 data (no adv), seed 1 | ml.g5.xlarge ($1.408/hr) | 3074 | $1.202 | 1307.9 | 1318.9 | 6:15:28 PM |
| lexsi-v2-s2-1110-1722 | v2 + 480 adv rows, seed 2 | ml.g5.2xlarge ($1.515/hr) | 4142 | $1.743 | 1854.7 | 1919.2 | 6:38:16 PM |
| lexsi-v2-s3-1110-1726 | v2 + 480 adv rows, seed 3 | ml.g6.xlarge ($1.127/hr) | 4956 | $1.551 | 2098.1 | 2465.4 | 6:59:22 PM |

Adapters are in `s3://sagemaker-lexsi-065614567889/out/<job>/output/model.tar.gz`.
The box cannot list that bucket (VPC endpoint policy).

## Partial eval (lexsi-eval-1110-1858)

Same held-out 256, hostile v1 (26), and blind hostile v2 (30) as the CPU
run. Scores are from the job's `EVAL` log lines (the tarball was not
emitted because the job failed on a later channel). Compare the ablation
to **this GPU v1 DPO**, not the earlier CPU figures.

| System | Held-out valid | Field exact match | Unsupported | Hostile v2 refused / 10 | v2 accepted but wrong |
| --- | --- | --- | --- | --- | --- |
| v1 base (this GPU run) | 45.3% | 72.6% | 33.2% | 2 | 9 |
| v1 SFT | 100% | 97.66% | 0.48% | 5 | 7 |
| v1 DPO | 100% | 97.66% | 0.71% | 5 | 9 |
| v1data-s1 SFT (ablation) | 100% | 95.98% | 0% | 6 | 10 |
| v1data-s1 DPO (ablation) | 100% | 95.98% | 0% | 6 | 7 |

The ablation (retrained on the larger v1-style set, no adversarial rows)
is 1.7 points worse than v1 DPO on held-out field match, writes no
unsupported values, and refuses 6 of 10 blind injections versus 5 of 10
for v1 DPO. v2 seeds are not in this table yet.

## Failures along the way

- 16:28 jobs: pip `torch==2.13.0` over the DLC left a broken `torchaudio`.
  `job.sh` uninstalls `torchaudio` and `torchvision`.
- `lexsi-v1data-s1-1110-1639` and `lexsi-v2-s3-1110-1639`: SFT finished,
  DPO trainer init hit cuDNN 9.20 (pip torch) vs 9.10 (DLC
  `LD_LIBRARY_PATH`). `job.sh` now prepends torch's libs and preflights
  `torch.backends.cudnn.version()`.
- Mid-SFT stops after that diagnosis: `lexsi-v2-s1-1110-1654`,
  `lexsi-v2-s2-1110-1654`.
- `lexsi-eval-1110-1858`: scored v1 and `v1data-s1`, then `tar` failed on
  SageMaker's `v1data-s1-manifest` channel. `job.sh` now skips channels
  without `model.tar.gz`; `launch.py` points the channel at the output
  prefix, not the object key. Rerun: `lexsi-eval-1110-1920` (v2-s2, v2-s3).

## Settled SageMaker spend at 7:20 PM ET (terminal jobs only)

Failed/stopped attempts $2.155 + completed training $4.496 + failed first
eval $0.453 = **$7.10**. Seed 1 and the second eval are still running.
Cap was about $40. Prices are SageMaker Training on-demand us-east-1.

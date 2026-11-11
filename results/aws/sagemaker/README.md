# SageMaker training (Nov 10, 2026): Lexsi v2 seeds and v1-data ablation

Status: **all four training jobs and all evals finished.** Seed 1 completed at
7:23 PM ET and was scored by `lexsi-eval-1110-2044`; seeds 2 and 3 were scored by
`lexsi-eval-1110-1920`. Raw outputs: `eval-v2/` (v1, v2-s2, v2-s3), `eval-v2-s1/`
(v1 re-score, v2-s1), `eval-partial/` (ablation). `v2-summary.json` is built by
`scripts/sagemaker/summarize.py`. One run per seed, so the spread is only 3 seeds.

Nothing here replaces the v1 CPU / Nova tables in `results/aws/README.md`;
it adds same-GPU v2 numbers.

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

## Final v2 results (3 seeds, same GPU eval, held-out 256, hostile v1 26, blind hostile v2 30)

| System | Held-out field exact match | Unsupported values | Hostile v2 refused / 10 | Hostile v2 accepted but wrong | Hostile v1 refused / 6 |
| --- | --- | --- | --- | --- | --- |
| v1 DPO (this GPU) | 97.66% | 0.71% | 5 | 9 | 2 |
| v1 SFT | 97.66% | 0.48% | 5 | 7 | 2 |
| Ablation v1-data DPO (seed 1) | 95.98% | 0% | 6 | 7 | 3 |
| v2 s1 SFT / DPO | 98.88% / **99.44%** | 0% / 0% | 9 / 10 | 4 / 8 | 5 / 5 |
| v2 s2 SFT / DPO | 98.88% / 98.94% | 0% / 0% | 9 / 7 | 6 / 7 | 5 / 5 |
| v2 s3 SFT / DPO | 98.38% / 98.60% | 0% / 0% | 10 / 8 | 3 / 7 | 5 / 6 |
| **v2 DPO mean (min–max)** | **99.00% (98.60–99.44)** | 0% | 8.3 (7–10) | 7.3 (7–8) | 5.3 (5–6) |
| v2 SFT mean (min–max) | 98.72% (98.38–98.88) | 0% | 9.3 (9–10) | 4.3 (3–6) | 5 |

Reading it honestly:
- v2 (with 480 adversarial rows) beats v1 DPO on held-out field match by 0.9 to 1.8 points on every seed
  and beats the no-adversarial ablation by 2.6 to 3.5 points. It writes 0 unsupported values.
- v2 refuses more blind injections (DPO 7–10 of 10, SFT 9–10) than v1 DPO (5) and the ablation (6).
- DPO is not clearly better than SFT on v2: it gains a little held-out accuracy, but SFT
  refused slightly more injections and had fewer accepted-but-wrong rows. DPO also had lower
  schema-valid counts on the hostile sets (e.g. s2 DPO 19/30 vs SFT 26/30).
- Best held-out: v2-s1 DPO (99.44%). Best on blind injections: v2-s3 SFT (10/10 refused, 3 wrong) and v2-s1 DPO (10/10 refused, 8 wrong).
- Caveats: 3 seeds, one eval each, hostile v2 is only 30 rows (10 injection tickets), the ablation is one seed,
  and differences of one or two rows on the hostile sets are within noise.

## Jobs and cost (final)

| Job | Instance | Billable s | Cost |
| --- | --- | --- | --- |
| lexsi-v2-s1-1110-1722 | ml.g6.2xlarge | 4905 | $1.665 |
| lexsi-eval-1110-1920 | ml.g5.xlarge | 1073 | $0.420 |
| lexsi-eval-1110-2044 (v2-s1 + v1 re-score) | ml.g5.xlarge | 903 | $0.353 |

Earlier settled spend was $7.10 (failed/stopped attempts, ablation, v2-s2, v2-s3, first eval).
**Total SageMaker about $9.54 of the $40 cap.** Teardown verified 9:02 PM ET: no training jobs in progress, no endpoints, no notebook instances, no running EC2.

# Stage 2 paper: analysis and job scripts

Scripts behind the Stage 2 paper's numbers. The logs they read are public in
the dataset [`nightmoose/schemasmith-eval-logs`](https://huggingface.co/datasets/nightmoose/schemasmith-eval-logs)
under `stage2/`; the v1/v2 SageMaker predictions are in this repo under
`results/aws/sagemaker/`.

## Re-score (CPU, no model)

```bash
pip install -e "../..[hf]" matplotlib
./reproduce.sh
```

`reproduce.sh` downloads `stage2/` from the dataset into `logs/`, links the
SageMaker predictions into `evals/`, runs the four statistics scripts into
`out/`, draws the two figures, and compares each JSON with the published one.

| Script | Writes | Paper |
| --- | --- | --- |
| `stage2_stats.py` | `stats.json`: bootstrap intervals, paired DPO-SFT, per-field accuracy, Wilson intervals, curation counts | Tables 1-2, RQ1-RQ3 |
| `ablation_stats.py` | `ablation_retention.json`: 3-seed no-adversarial ablation, v1 Belebele retention | Ablation, retention |
| `mcnemar_stats.py` | `mcnemar.json`: exact McNemar, SFT vs DPO per seed | RQ2 |
| `round2_stats.py` | `round2.json`: v2 Belebele retention, XSTest refusal labels and tests | Retention, safety |
| `make_figures.py` | `per_field.pdf`, `tradeoff.pdf` | Figures 2-3 |

## GPU jobs (Hugging Face Jobs, one L4)

`jobs/` holds the job scripts as they ran on Nov 19, 2026, except that the
package wheel now comes from a Hub repo you name in `SRC_REPO`
(`nightmoose/schemasmith-src` is private; build your own wheel, see
`docs/reproducibility.md`). `jobs/xstest.py` ran as `safety.py`.

| Job | Files | Output in the dataset |
| --- | --- | --- |
| No-adversarial ablation, seeds 1-3 | `ablation.sh` | `stage2/ablation/` |
| Belebele retention, base and v1 | `retention.sh` + `retention.py` | `stage2/retention/v1/` |
| Belebele retention, v2 seed N | `generic.sh` + `retention_v2.py` | `stage2/retention/v2-sN/` |
| XSTest, 9 systems | `generic.sh` + `xstest.py` | `stage2/xstest/outputs.json` |

`jobs/launch.py` starts one job with a hard timeout and refuses a cost
ceiling above $2. Inputs come from the Hub: `CohereLabs/tiny-aya-global`,
`facebook/belebele`, `Paul/XSTest`, and the public adapters
`nightmoose/schemasmith-tiny-aya-sft`, `nightmoose/schemasmith-tiny-aya-dpo`
and `nightmoose/schemasmith-v2-adapters`.

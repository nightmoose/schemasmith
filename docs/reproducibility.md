# Reproducibility

Python 3.11 or newer. No model is downloaded by the commands in the first section.

## Local checks

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
pytest
schemasmith generate --n 24 --seed 42 --out data/raw/tickets.jsonl
schemasmith curate --raw data/raw/tickets.jsonl --out data/curated --seed 42
schemasmith dpo \
  --alpaca data/curated/local/train/sft_alpaca.jsonl \
  --out data/dpo \
  --seed 42
```

Seed 42 fixes the generator, the local split, and the preference-pair modes. The test split of the local curate is the held-out set. DPO uses the train split only.

CuratorKIT and AlignTune are optional and listed in `requirements-lexsi.txt`. They are not imported by `pytest`.

## GPU steps

1. Accept the license on the Tiny Aya model card and export `HF_TOKEN`. The token is not stored in the repo.
2. Run `notebooks/lora_sft_tiny_aya.ipynb` on a free T4. The training cell returns when CUDA is missing.
3. Run `notebooks/lora_dpo_tiny_aya.ipynb` on a free T4. Point it at the SFT adapter from step 2. The notebook merges that adapter, then calls AlignTune DPO with a new LoRA.
4. Score the held-out file:

```bash
schemasmith compare \
  --data data/curated/local/test/sft_alpaca.jsonl \
  --sft-adapter OUTPUT_SFT \
  --dpo-adapter OUTPUT_DPO \
  --out reports/compare.json \
  --readme README.md
```

If a job already wrote `reports/compare.json`, refresh the table without loading a model:

```bash
schemasmith compare --results reports/compare.json --readme README.md
```

The README changes only between the metrics markers. Any other edit is a mistake.

## HF Jobs

Preview, which does not start a job and does not read a token:

```bash
schemasmith jobs sft --namespace HF_USER
schemasmith jobs dpo --namespace HF_USER
schemasmith jobs eval --namespace HF_USER
```

Launch only after the preview ceiling is acceptable, and only after this revision is on the public default branch the job installs:

```bash
export HF_TOKEN=...
schemasmith jobs sft --namespace HF_USER --launch
schemasmith jobs dpo --namespace HF_USER --launch
schemasmith jobs eval --namespace HF_USER --launch
```

The flavor stays on the small T4 sizes. Launch refreshes the hourly rate from the Hub and prints the timeout ceiling before `run_job`. The token is passed as a secret. Job outputs are pushed to private repos. The disk on a job is ephemeral, which is why the push exists.

## Publishing

```bash
schemasmith publish --namespace HF_USER --cards-dir dist/cards
schemasmith publish --namespace HF_USER --execute \
  --dataset-dir data/curated/local \
  --sft-dir OUTPUT_SFT \
  --dpo-dir OUTPUT_DPO \
  --eval-dir reports
```

The first command writes cards and makes no Hub call. The second, with `--public`, uploads public ungated repos and a public collection. Adapter cards state the CC-BY-NC-4.0 license. Private job uploads are only a staging copy. Judges need the public publish.

## Space

```bash
python space/app.py
```

Mock mode is on. The scorecard stays TBD until a compare file is loaded in the text box.

## What is not measured here

Base, SFT, and DPO metrics. Do not copy the planted-failure checks or a gold-JSON pipeline check into the results table.

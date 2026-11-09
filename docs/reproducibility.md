# Reproducibility

Python 3.11 or newer. No model is downloaded by the commands in the first section.

## Local checks

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
pytest
schemasmith generate --n 320 --seed 42 --out data/raw/tickets.jsonl
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

Launch only after the preview ceiling is acceptable. The job does not clone GitHub. It installs a wheel from the private Hub model repo `HF_USER/schemasmith-src`, read with the `HF_TOKEN` job secret. Build the wheel and upload it on the first launch:

```bash
export HF_TOKEN=...
python -m pip wheel --no-deps -w dist .
schemasmith jobs sft --namespace HF_USER --n 320 --upload-wheel dist/schemasmith-0.2.0-py3-none-any.whl --launch
schemasmith jobs dpo --namespace HF_USER --n 320 --launch
schemasmith jobs eval --namespace HF_USER --n 320 --launch
```

The default flavor is `l4x1`. AlignTune 0.1.32 runs DPO in bf16, which transformers refuses on a T4. The image is `pytorch/pytorch:2.13.0-cuda12.6-cudnn9-runtime` because AlignTune needs torch below 2.14. `--n 320` gives a 219 / 37 / 64 split, with whole sub-scenarios and templates held out of test. SFT runs about 3 epochs. DPO runs 1 epoch at learning rate 2e-5.

SFT and DPO both train on the Tiny Aya chat template, which is the format the compare prompts with. AlignTune builds a user / assistant `messages` pair for SFT. For DPO, AlignTune flattens message lists to plain text, so `chat_format_dpo_dir` renders the template into the strings before training. Launch refreshes the hourly rate from the Hub and prints the timeout ceiling before `run_job`. The token is passed as a secret. Job outputs are pushed to private repos. The disk on a job is ephemeral, which is why the push exists.

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

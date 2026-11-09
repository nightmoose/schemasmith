# SchemaSmith

SchemaSmith turns a multilingual support ticket into one JSON object that matches a data contract. The contract is the gate for curation and the scorer for evaluation.

**Core model:** Tiny Aya Global, `CohereLabs/tiny-aya-global`. It is the only model this submission trains and evaluates. The other Cohere hackathon models are not used.

This README is the submission write-up. `docs/hf_article.md` is an optional draft, not a second write-up.

This repository is an entry for the Lexsi AI Safety & Alignment Hackathon, Track 1 (Fine-tuning and Data Curation), problem 5, structured output and extraction.

The pipeline is:

1. A synthetic multilingual ticket set, small on purpose.
2. CuratorKIT for clean and exact dedup, with this contract as a row filter.
3. AlignTune LoRA supervised fine-tuning of Tiny Aya Global, then a DPO LoRA on those weights.
4. AuditKIT plus the contract validator for the held-out compare of base, SFT, and DPO.

Training has not been run. Every model metric below is **TBD**.

## Contract

`src/schemasmith/contracts/support_ticket.odcs.yaml` is the readable contract. `src/schemasmith/contracts/support_ticket.schema.json` is the JSON Schema (draft 2020-12) the validator executes.

A row is contract-valid when it parses as one JSON object and validates against that schema. The schema requires:

- `ticket_id` matching `T-` plus letters or digits
- `customer_tier` in `free`, `pro`, `enterprise`
- `category` in `billing`, `bug`, `access`, `feature_request`
- `severity` an integer from 1 to 4
- `order_ref` matching `ORD-` plus six digits, or null
- `refund_amount_usd` a non-negative number, or null
- `language` an ISO 639-1 code
- `refund_amount_usd` null unless `category` is `billing`

The validator reports:

| Rate | Definition |
| --- | --- |
| Parse rate | Fraction of outputs that yield one JSON object (raw, fenced, or embedded). |
| Schema-valid rate | Fraction that parse and pass the JSON Schema. Same value as the contract-valid rate. |
| Field-level exact match | Matching fields divided by rows times 7. An unparsed row scores 0. Refund amounts match within `1e-6`. |
| Unsupported-value rate | Among non-null `ticket_id`, `order_ref`, and `refund_amount_usd` values in parsed outputs, the fraction that do not appear in the source text. |
| Non-English contract-valid rate | Contract-valid rate on rows whose gold language is not `en`. |

## Reproduce

Python 3.11 or newer. The commands below do not download Tiny Aya and do not call a paid API.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
pytest
schemasmith generate --n 320 --seed 42 --out data/raw/tickets.jsonl
schemasmith curate --raw data/raw/tickets.jsonl --out data/curated --seed 42
```

`data/curated/local/` then contains `train`, `validation`, and `test` alpaca and messages JSONL, plus `manifest.json`. Seed `42` fixes generation and the local split.

To score a prediction file (one JSON object per line, with `id` and `output`):

```bash
schemasmith eval \
  --data data/curated/local/test/sft_alpaca.jsonl \
  --predictions preds.jsonl \
  --report reports/heldout.json
```

The report describes the file you passed. It is not a base-model measurement unless you produced that file with the base model.

### CuratorKIT

```bash
pip install -r requirements-lexsi.txt
schemasmith curate --raw data/raw/tickets.jsonl --out data/curated --seed 42 --curatorkit
```

That pass writes `data/curated/curatorkit/`. It uses CuratorKIT 0.1.23 with no generator model: exact dedup, text cleaning on `instruction` and `input` only, and a preprocessing function that drops rows whose `output` fails this contract. CuratorKIT then splits with its own shuffle (`output_split_seed` 42) into `train`, `validation`, and `test`. That split is not the local split.

CuratorKIT's own SchemaGate checks token counts and task fields. It does not know this contract. The contract filter is the preprocessing function.

### LoRA fine-tune

`notebooks/lora_sft_tiny_aya.ipynb` is the training entry. Run it on a free T4 in Colab or Kaggle, from the repository root, with a Hugging Face token that can read the gated `CohereLabs/tiny-aya-global` repo. The notebook installs `requirements-lexsi.txt`, builds the 24-row set, runs CuratorKIT, and calls AlignTune `create_sft_trainer` on the TRL backend with LoRA. If there is no GPU, the training cell prints that and returns.

The hyperparameters live in `schemasmith.training.lora_sft_kwargs`. Backend is `trl`. The dataset subset is `sft_alpaca`. See `docs/lexsi_apis.md` for why that subset is used.

Do not treat a gold-JSON pass through the validator as a model score. The notebook's last scoring cell is that kind of pipeline check and is labeled as such.

### DPO pairs and notebook

Build pairs from the train split only, so the test split stays held out:

```bash
schemasmith dpo \
  --alpaca data/curated/local/train/sft_alpaca.jsonl \
  --out data/dpo \
  --seed 42
```

`notebooks/lora_dpo_tiny_aya.ipynb` is the free-T4 DPO entry. It calls AlignTune `create_rl_trainer` with `algorithm="dpo"` and `backend="trl"`. AlignTune 0.1.32 starts from `model_name` and adds a new LoRA. The notebook merges the SFT adapter into that starting checkpoint when a GPU and the adapter are present. If either is missing, the training cell returns without writing weights.

### Held-out compare

On a CUDA machine, after both adapters exist:

```bash
schemasmith compare \
  --data data/curated/local/test/sft_alpaca.jsonl \
  --sft-adapter path/or/repo-of-sft \
  --dpo-adapter path/or/repo-of-dpo \
  --out reports/compare.json \
  --readme README.md
```

That command generates with the base model and each adapter, writes `reports/compare.json`, and rewrites only the marked metrics block below. Until that file exists, the block stays TBD.

To refresh the table from a file a job already wrote, without loading a model:

```bash
schemasmith compare --results reports/compare.json --readme README.md
```

### HF Jobs and publishing

`schemasmith jobs sft|dpo|eval` prints a plan and does not start a job. Add `--launch` to start one. The plan names a small GPU flavor and a cost ceiling. Launch reads `HF_TOKEN` from the environment and sends it as a job secret.

`schemasmith publish` is a dry run and private by default. `--execute` uploads. `--public` is the switch that makes the repos and the collection public. Those public repos are not gated. Adapter cards carry CC-BY-NC-4.0 because they are derived from Tiny Aya Global. A private or manually approved repo is not what judges should be sent.

`docs/reproducibility.md` lists the order. `space/app.py` is the Gradio page. Mock mode is the default and does not load a model.

### Optional closed-model comparison

Preferred, not measured. The formula is in `schemasmith roi`. Self-host cost uses a GPU hourly rate the operator supplies and a measured seconds-per-ticket. Closed-model cost uses a token price the operator supplies and measured prompt and completion lengths. If the compare file has no `roi_measurements` block, the command writes null costs and does not invent them.

```bash
schemasmith roi --out reports/roi.json
```

## Write-up

Problem: support tickets in several languages need to land as schema-valid JSON. Approach: keep Tiny Aya Global as the core model, curate a small synthetic set with the contract as a filter, then LoRA SFT and DPO with AlignTune.

Repos: CuratorKIT curates and deduplicates. AlignTune runs SFT and DPO on the TRL backend. AuditKIT is the optional extra scorer next to the contract validator. Feature notes are in `docs/lexsi_apis.md`.

Core model and license: Tiny Aya Global, `CohereLabs/tiny-aya-global`, CC-BY-NC-4.0. No supplemental model.

Data: synthetic tickets generated in this repo. No third-party training set. The contract files are in `src/schemasmith/contracts/`.

Artifacts: not uploaded yet. After the GPU runs, `schemasmith publish --public --execute` is what makes the dataset, both adapters, the eval logs, and the collection public and ungated.

Experiments and before/after: not run. The table below stays TBD until `schemasmith compare` rewrites it from a results file.

## Results

Not measured. No base-model run and no fine-tune have been executed in this revision.

<!-- SCHEMASMITH_METRICS_START -->
| Metric | Base `tiny-aya-global` | SFT | DPO | Status |
| --- | --- | --- | --- | --- |
| Parse rate | TBD | TBD | TBD | not yet measured |
| Schema-valid rate | TBD | TBD | TBD | not yet measured |
| Field-level exact match | TBD | TBD | TBD | not yet measured |
| Unsupported-value rate | TBD | TBD | TBD | not yet measured |
| Non-English contract-valid rate | TBD | TBD | TBD | not yet measured |
<!-- SCHEMASMITH_METRICS_END -->

## What is left

Run SFT, DPO, and the held-out compare on a GPU. Then `schemasmith publish --namespace HF_USER --public --execute` so judges can open the repos without an approval step. The metrics block is still TBD. A criterion map is in `docs/rubric_map.md`.

## API notes

Which Lexsi calls were executed, which were read from the 0.1.32 AlignTune sdist, and which are still assumptions: `docs/lexsi_apis.md`.

## Licenses

This repository’s original code is licensed under the **MIT License**.

**Lexsi Labs tools** (CuratorKIT, AuditKIT, and related Lexsi source-available packages) are used only as **external, non-vendored dependencies** (see `requirements-lexsi.txt`). Their source is **not copied** into this repository. Those tools remain under the Lexsi Labs Source Available License (LSAL). This use is a **hackathon-only exception** approved for the Lexsi AI Safety & Alignment Hackathon entry.

**Model weights:** Cohere Tiny Aya / Aya family weights used here are **CC-BY-NC-4.0** (non-commercial). See the model card on Hugging Face for terms.

Do not vendor or re-license Lexsi LSAL source into this MIT tree.

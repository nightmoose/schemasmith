# SchemaSmith

SchemaSmith turns a multilingual support ticket into one JSON object that matches a data contract. The contract is the gate for curation and the scorer for evaluation. The model under test is Cohere Labs Tiny Aya (`CohereLabs/tiny-aya-global`), adapted with LoRA.

This repository is an entry for the Lexsi AI Safety & Alignment Hackathon, Track 1 (Fine-tuning and Data Curation), problem 5, structured output and extraction.

The pipeline is:

1. A synthetic multilingual ticket set, small on purpose.
2. CuratorKIT for clean and exact dedup, with this contract as a row filter.
3. AlignTune LoRA supervised fine-tuning on a free T4, via the notebook.
4. AuditKIT plus the contract validator for held-out scores.

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
schemasmith generate --n 24 --seed 42 --out data/raw/tickets.jsonl
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

## Results

Not measured. No base-model run and no fine-tune have been executed in this revision.

| Metric | Base `tiny-aya-global` | LoRA fine-tune | Status |
| --- | --- | --- | --- |
| Parse rate | TBD | TBD | not yet measured |
| Schema-valid rate | TBD | TBD | not yet measured |
| Field-level exact match | TBD | TBD | not yet measured |
| Unsupported-value rate | TBD | TBD | not yet measured |
| Non-English contract-valid rate | TBD | TBD | not yet measured |

## What is left

Direct preference optimization, a real base-versus-tuned evaluation, public Hugging Face dataset and adapter uploads, and the Space demo are not in this revision.

## API notes

Which Lexsi calls were executed, which were read from the 0.1.32 AlignTune sdist, and which are still assumptions: `docs/lexsi_apis.md`.

## Licenses

This repository’s original code is licensed under the **MIT License**.

**Lexsi Labs tools** (CuratorKIT, AuditKIT, and related Lexsi source-available packages) are used only as **external, non-vendored dependencies** (see `requirements-lexsi.txt`). Their source is **not copied** into this repository. Those tools remain under the Lexsi Labs Source Available License (LSAL). This use is a **hackathon-only exception** approved for the Lexsi AI Safety & Alignment Hackathon entry.

**Model weights:** Cohere Tiny Aya / Aya family weights used here are **CC-BY-NC-4.0** (non-commercial). See the model card on Hugging Face for terms.

Do not vendor or re-license Lexsi LSAL source into this MIT tree.

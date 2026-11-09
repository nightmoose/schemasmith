# Optional draft

The submission write-up is the GitHub README. This file is not required.

# SchemaSmith: contract-valid ticket JSON from Tiny Aya Global

SchemaSmith extracts one support ticket into JSON that has to match a data contract. The contract is checked before training and again on the held-out set. The core model is Tiny Aya Global, `CohereLabs/tiny-aya-global`. No other model is part of the submission.

## Why a contract

A warehouse row is useful when the fields are the right type, the enums are closed, and a refund does not appear on a non-billing ticket. A fluent answer that invents an amount or drops the order id still fails. The scorer therefore reports parse rate, schema-valid rate, field-level exact match, an unsupported-value rate for ids and amounts, and the contract-valid rate on non-English tickets.

## Data

The training set is synthetic and small: eight languages, a fixed seed, and labels that already validate. CuratorKIT is an external curation step. It deduplicates and cleans. A preprocessing hook drops any row whose label fails the contract. CuratorKIT's own schema gate does not know this contract.

Preference data keeps the labeled JSON as the chosen answer. The rejected answer is a planted failure: broken JSON, a missing field, an extra field, the wrong language code, or a made-up refund. Those strings are not model outputs.

## Training

AlignTune's TRL backend runs LoRA supervised fine-tuning, then DPO. The DPO run starts from the supervised weights and trains a new LoRA. The Unsloth backend is not used. Notebooks target a free T4. The same steps can run as a Hugging Face Job on a small GPU. The job command prints a cost ceiling before it launches and does not launch unless asked.

## Evaluation

One compare command runs the base model, the SFT adapter, and the DPO adapter on the held-out split, writes a JSON file, and rewrites the metrics table from that file. Until that file exists, every number in the table is TBD. This article does not fill them in.

## Page

A Gradio page shows the contract and a sample ticket. Mock mode returns the labeled JSON and says so. It does not load Tiny Aya.

## Licenses

Repository code is MIT. Tiny Aya weights are CC-BY-NC-4.0. CuratorKIT and AuditKIT stay external under the Lexsi Labs Source Available License and are not copied into the repository.

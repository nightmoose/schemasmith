# Optional draft

The submission write-up is the GitHub README. This file is not required.

# Lexsi SchemaSmith: made-up field values 32.9% to 0%

On 64 held-out tickets, base Tiny Aya Global wrote an order reference or refund amount that is not in the ticket for 52 of 158 ids and amounts (the unsupported-value rate). After LoRA SFT and DPO, 0 of 106. On 26 hand-written hostile tickets, the final model cut that from 36.4% to 4.5% but still followed 4 of 6 prompt injections. The README has both tables and every miss.

SchemaSmith extracts one support ticket into JSON that has to match a data contract. The contract is checked before training and again on the held-out set. The core model is Tiny Aya Global, `CohereLabs/tiny-aya-global`. No other model is part of the submission.

## Why a contract

A warehouse row is useful when the fields are the right type, the enums are closed, and a refund does not appear on a non-billing ticket. A fluent answer that invents an amount or drops the order id still fails. The scorer therefore reports parse rate, schema-valid rate, field-level exact match, an unsupported-value rate for ids and amounts, and the contract-valid rate on non-English tickets.

## Data

The training set is synthetic and small: eight languages, a fixed seed, and labels that already validate. CuratorKIT is an external curation step. It deduplicates and cleans. A preprocessing hook drops any row whose label fails the contract. CuratorKIT's own schema gate does not know this contract.

Preference data keeps the labeled JSON as the chosen answer. The rejected answer is a planted failure: broken JSON, a missing field, an extra field, the wrong language code, or a made-up refund. Those strings are not model outputs.

## Training

AlignTune's TRL backend runs LoRA supervised fine-tuning, then DPO. The DPO run starts from the supervised weights and trains a new LoRA. The Unsloth backend is not used. Notebooks target a free T4. The same steps can run as a Hugging Face Job on a small GPU. The job command prints a cost ceiling before it launches and does not launch unless asked.

## Evaluation

One compare command runs the base model, the SFT adapter, and the DPO adapter on the held-out split, writes a JSON file, and rewrites the metrics table from that file. A second job runs the same three systems on the hostile set. The README tables are checked against both files by tests.

## Page

A Gradio page replays a recorded run: real base, SFT, and DPO outputs on the same tickets, with the contract check and the gate decision. It does not load Tiny Aya. A mock playground shows the contract on labeled examples.

## Licenses

Repository code is MIT. Tiny Aya weights are CC-BY-NC-4.0. CuratorKIT and AuditKIT stay external under the Lexsi Labs Source Available License and are not copied into the repository.

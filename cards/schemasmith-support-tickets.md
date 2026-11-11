---
license: mit
tags:
- schemasmith
configs:
- config_name: sft_alpaca
  default: true
  data_files:
    - split: train
      path: train/sft_alpaca.jsonl
    - split: validation
      path: validation/sft_alpaca.jsonl
    - split: test
      path: test/sft_alpaca.jsonl
- config_name: sft_messages
  data_files:
    - split: train
      path: train/sft_messages.jsonl
    - split: validation
      path: validation/sft_messages.jsonl
    - split: test
      path: test/sft_messages.jsonl
---

# Lexsi SchemaSmith support tickets

Synthetic multilingual support tickets and the contract-valid JSON labels
used to train a small extractor. This card does not report model scores.
The tickets are original work in this repository and are licensed MIT.
Adapters derived from Tiny Aya Global are separately licensed CC-BY-NC-4.0.

## Licenses

The training and evaluation code in the SchemaSmith repository is MIT.

The core model is Tiny Aya Global, `CohereLabs/tiny-aya-global`.
No other model is trained or scored as the submission model.
Those weights, and every adapter derived from them, are CC-BY-NC-4.0.
Public adapter repos are not gated. The license is this card.

CuratorKIT and AuditKIT are external dependencies under the Lexsi Labs
Source Available License. Their source is not in this repository.
AlignTune is the training library. This project calls its TRL backend.


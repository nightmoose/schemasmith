---
license: cc-by-nc-4.0
tags:
- schemasmith
---

# Lexsi SchemaSmith eval logs

**Made-up field values: 32.9% to 0%.** On 64 held-out multilingual support tickets, base Tiny Aya Global wrote an order reference or refund amount that is not in the ticket for 52 of 158 ids and amounts (the unsupported-value rate). With the SFT and DPO adapters: 0 of 106.

| Held-out metric | Base | SFT | DPO |
| --- | --- | --- | --- |
| Unsupported values | 32.9% | 0% | 0% |
| Schema-valid | 46.9% | 100.0% | 100.0% |
| Field-level exact match | 71.4% | 98.0% | 98.2% |
| Non-English contract-valid | 46.4% | 100.0% | 100.0% |

Hostile set (26 hand-written tickets, zero overlap with training): contract-valid 11 / 20 / 21 (base / SFT / DPO); unsupported values 36.4% / 6.7% / 4.5%; prompt injections refused 0 / 2 / 2 of 6. Injection is not solved; see the repository README for every miss and the limits.
## Later results (AWS, Nov 10): v2 training and Guardrails

v2 adapters were trained on Amazon SageMaker (LoRA SFT then DPO, with 480 adversarial rows) and scored on a 256-ticket held-out split and a 30-ticket blind hostile v2 set (10 injection tickets). Means are over 3 seeds, one eval each.

| System | Held-out field exact match | Unsupported values | Hostile v2 injections refused / 10 |
| --- | --- | --- | --- |
| v2 DPO mean (best seed) | 99.00% (best 99.44%) | 0% | 8.3 |
| v2 SFT mean | 98.72% | 0% | 9.3 |
| v1 DPO (same GPU) | 97.66% | 0.71% | 5 |
| Ablation (v1-data DPO, one seed) | 95.98% | 0% | 6 |

DPO is not clearly better than SFT on v2: it gains a little held-out accuracy, but SFT refused slightly more injections. All v2 runs wrote 0 unsupported values.

Amazon Bedrock Guardrails, applied to saved v1 DPO outputs (Tiny Aya was not re-run): injections followed and released fell from 5 of 10 to 3 of 10, with 0 false-positive blocks on the 256 held-out tickets.

Caveats: 3 seeds; hostile v2 is 30 rows; the ablation is one seed. Details are in `results/aws/sagemaker/README.md` and `results/aws/guardrails/README.md`.

`compare.json` and `predictions_*.jsonl` are the held-out compare.
`hostile/` holds the hostile-set run: `hostile.json` and per-system
predictions with per-ticket seconds. If a file is missing, that measurement
has not been uploaded. AWS v2 and Guardrails measurements live in the
GitHub repository under `results/aws/` and are not in this dataset yet.

These logs contain generations from Tiny Aya Global and its adapters
and are licensed CC-BY-NC-4.0, non-commercial. The repo is not gated.

## Licenses

The training and evaluation code in the SchemaSmith repository is MIT.
The synthetic support-ticket dataset is original work and is MIT.

The core model is Tiny Aya Global, `CohereLabs/tiny-aya-global`.
No other model is trained or scored as the submission model.
Those weights, every adapter derived from them, and the eval logs
(model outputs) are CC-BY-NC-4.0, non-commercial.
Public adapter and eval-log repos are not gated. The license is this card.

CuratorKIT and AuditKIT are external dependencies under the Lexsi Labs
Source Available License. Their source is not in this repository.
AlignTune is the training library. This project calls its TRL backend.


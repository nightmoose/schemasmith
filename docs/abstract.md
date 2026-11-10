# Abstract

**Made-up field values: 32.9% to 0%.** Lexsi SchemaSmith fine-tunes Cohere Labs Tiny Aya Global (`CohereLabs/tiny-aya-global`) to turn a messy, multilingual support ticket into one JSON object that passes a data contract. On 64 held-out tickets, the base model wrote an order reference or refund amount that is not in the ticket for 52 of the 158 ids and amounts it filled in (the unsupported-value rate, 32.9%). After LoRA SFT and DPO with Lexsi's AlignTune, that is 0 of 106, and schema-valid output went from 46.9% to 100%. Field-level exact match went from 71.4% to 98.2%, and the non-English contract-valid rate from 46.4% to 100%.

Why it matters: tickets land in warehouses and refund queues as JSON. An invented order reference credits the wrong order, and a refund on a bug ticket pays out for nothing. So the model only suggests. Code decides: a contract check and a grounding check (every id and amount must appear in the ticket) accept a row or hold it for a person.

Hostile test: 26 hand-written tickets with zero overlap with training, covering prompt injections, contradictions, off-enum values, mixed languages, garbage, and very long threads. The DPO model was contract-valid on 21 of 26 (base 11) and cut made-up values from 36.4% to 4.5%. It still followed 4 of 6 prompt injections. The gate held 6 rows and let 4 wrong rows through, three of them injected values that the ticket text itself contained. We report that as the main open problem, not a footnote.

Method: a small synthetic set in eight languages with labels that already pass the contract; CuratorKIT for cleaning and dedup with the contract as a row filter; AlignTune LoRA SFT, then DPO on the SFT-merged weights with planted failures as rejected answers; evaluation with the contract validator, with AuditKIT as an optional extra scorer. Training and evaluation ran as small Hugging Face Jobs on one L4 GPU. The final adapter takes about 3.7 seconds per ticket there, unbatched.

Limits: training and held-out tickets are synthetic and template-based; the sets are small and each was run once; the DPO setting was chosen after one earlier run on the held-out split; and injection resistance was not trained.

Code is MIT. Tiny Aya weights and both adapters are CC-BY-NC-4.0, non-commercial. Lexsi source-available tools are dependencies and are not copied into the repository. Commands, seeds, and job steps are in the reproducibility guide.

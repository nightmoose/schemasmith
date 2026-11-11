# Abstract

**Made-up field values: 32.9% to 0%.** Lexsi SchemaSmith fine-tunes Cohere Labs Tiny Aya Global (`CohereLabs/tiny-aya-global`) to turn a messy, multilingual support ticket into one JSON object that passes a data contract. On 64 held-out tickets, the base model wrote an order reference or refund amount that is not in the ticket for 52 of the 158 ids and amounts it filled in (unsupported-value rate 32.9%). After LoRA SFT and DPO with Lexsi's AlignTune, that is 0 of 106, and schema-valid output went from 46.9% to 100%. Field-level exact match went from 71.4% to 98.2%, and non-English contract-valid from 46.4% to 100%.

Why it matters: tickets land in warehouses and refund queues as JSON. An invented order reference credits the wrong order; a refund on a bug ticket pays out for nothing. The model only suggests. Code decides: a contract check and a grounding check (every id and amount must appear in the ticket) accept a row or hold it for a person.

Hostile test: 26 hand-written tickets with zero overlap with training (injections, contradictions, off-enum values, mixed languages, garbage, long threads). The DPO model was contract-valid on 21 of 26 (base 11) and cut made-up values from 36.4% to 4.5%. It still followed 4 of 6 prompt injections. The gate held 6 rows and let 4 wrong rows through, three of them injected values the ticket itself contained. That is the main open problem, not a footnote.

A later 3-seed SageMaker v2 run on 256 held-out tickets reached a 99.00% mean DPO field match (best seed 99.44%; SFT mean 98.72%), with 0% unsupported values, and refused 8.3 of 10 blind injections (SFT 9.3 of 10). A no-adversarial ablation scored 95.98%. Bedrock Guardrails on saved v1 DPO outputs cut injections followed and released from 5 of 10 to 3 of 10, with 0 false positives on 256 clean tickets. DPO is not clearly better than SFT on v2.

Method: a small synthetic set in eight languages with contract-valid labels; CuratorKIT for cleaning and dedup with the contract as a row filter; AlignTune LoRA SFT, then DPO on the SFT-merged weights; the contract validator, with AuditKIT as an optional scorer. Original runs were Hugging Face Jobs on one L4. v2 used Amazon SageMaker; references used Amazon EC2 and Amazon Bedrock (Nova and Guardrails).

Limits: training and held-out tickets are synthetic and template-based; the 64-ticket compare is a single run, and that DPO setting was chosen after one earlier run on the same split; v2 is 3 seeds and 30 hostile rows; injection is improved, not solved.

Code and the synthetic dataset are MIT. Tiny Aya weights, both adapters, and the eval logs are CC-BY-NC-4.0, non-commercial. Lexsi source-available tools are dependencies and are not copied into the repository.

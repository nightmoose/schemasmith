# Submission and gallery text

Exact text for the Lexsi AI Safety & Alignment Hackathon form and gallery card. Prepared Nov 11, 2026. Not submitted. The repo, datasets, adapters, and collection stay private until publish day (Nov 14); fill the collection link marked "on publish" then. Follow `PUBLISH_CHECKLIST.md`. Do not flip visibility or submit from this file.

## Card

- **Project / gallery card title:** Lexsi SchemaSmith
- **Team name:** Lexsi SchemaSmith
- **Members:** Alex Suarez (GitHub @nightmoose), solo. Use the members field for this, not the title.
- **Track:** Track 1, Fine-tuning and Data Curation. Problem 5, structured output and extraction.
- **Core model:** Tiny Aya Global, `CohereLabs/tiny-aya-global`
- **Lexsi repos used:** CuratorKIT, AlignTune (SFT and DPO), AuditKIT (optional scorer)
- **Tags:** Tiny Aya, CuratorKIT, AlignTune, AuditKIT, Hugging Face Jobs/Hub, PyTorch, PEFT, Gradio, Amazon SageMaker, Amazon EC2, Amazon Bedrock (Nova + Guardrails)
- **Tagline (one line):** Made-up order refs and refunds in ticket JSON: 32.9% to 0% on 64 held-out tickets; v2 DPO mean 99.00% field match (best seed 99.44%) on 256, with code, not the model, deciding what is accepted.

## Short description (gallery card, about 80 words)

Lexsi SchemaSmith fine-tunes Tiny Aya Global to turn multilingual support tickets into contract-checked JSON. On 64 held-out tickets, made-up order references and refund amounts fell from 32.9% to 0%, and valid output rose from 46.9% to 100%. A later 3-seed SageMaker v2 run reached 99.00% mean field match on 256 tickets (best seed 99.44%), with 0% unsupported values, and refused 8.3 of 10 blind injections (SFT 9.3 of 10). Bedrock Guardrails cut v1 released follow-through from 5 of 10 to 3 of 10, with 0 false positives. A code gate decides what is accepted.

## Long description

**Made-up field values: 32.9% to 0%.** Support tickets land in warehouses and refund queues as JSON. An invented order reference credits the wrong order; a refund on a bug ticket pays out for nothing. Lexsi SchemaSmith fine-tunes Cohere Labs Tiny Aya Global to extract one ticket into one JSON object that passes a data contract, in eight languages.

Results on 64 held-out tickets (base, then SFT, then DPO; zero template overlap; from `results/compare.json`):

- Unsupported values (ids and amounts not in the ticket): 32.9% (52 of 158), 0% (0 of 106), 0% (0 of 106)
- Schema-valid: 46.9%, 100%, 100%
- Field-level exact match: 71.4%, 98.0%, 98.2%
- Non-English contract-valid: 46.4%, 100%, 100%

Hostile test, 26 hand-written tickets with zero overlap with training (injections, contradictions, off-enum values, mixed languages, garbage, very long threads), base then SFT then DPO (`results/hostile.json`):

- Contract-valid: 11, 20, 21 of 26
- Unsupported values: 36.4%, 6.7%, 4.5%
- Injections refused: 0, 2, 2 of 6
- Code gate accepted but wrong: 3, 4, 4

Later v2 training on Amazon SageMaker (3 seeds, 256 held-out tickets, 30-row blind hostile set; `results/aws/sagemaker/v2-summary.json`):

- v2 DPO mean field exact match 99.00% (best seed 99.44%); v2 SFT mean 98.72%; same-GPU v1 DPO 97.66%; no-adversarial ablation 95.98%
- v2 unsupported values: 0%
- Injections refused on the blind set: v2 DPO 8.3/10, v2 SFT 9.3/10
- Caveats: 3 seeds, one eval each, 30 hostile rows; DPO is not clearly better than SFT on v2

Amazon Bedrock Guardrails, applied to saved v1 DPO outputs (the model was not re-run): injections followed and released fell from 5 of 10 to 3 of 10, with 0 false positives on 256 clean tickets (`results/aws/guardrails/summary.json`).

AWS jobs used, honestly: SageMaker training `lexsi-v2-s2-1110-1722` (ml.g5.2xlarge) and `lexsi-v1data-s1-1110-1722` (ml.g5.xlarge), plus seeds 1 and 3 and three eval jobs, completed Nov 10; Bedrock Nova Micro/Lite/Pro/2 Lite references; Bedrock Guardrails; EC2 `c7i.2xlarge` (pipeline smoke) and `c7i.4xlarge` (v1 CPU eval). The original 64-ticket compare ran as Hugging Face Jobs on one L4.

Models suggest, code decides: a contract check and a grounding check accept a row only when it is valid and every id and amount is in the ticket text; anything else is held for a person with the reason. The gate never accepted a made-up value. It did accept injected values that the ticket text itself contained. Injection is improved on v2 and with Guardrails, not solved. The README lists every miss.

How: CuratorKIT curates a small synthetic set with the contract as a row filter; AlignTune runs LoRA SFT, then DPO on the SFT-merged weights; the contract validator (with AuditKIT as an optional extra) scores the held-out compare and the hostile sets.

Limits: synthetic, template-based training and held-out data; small hostile sets; the original DPO setting was chosen after one earlier run on the 64-ticket split; v2 is 3 seeds; the ablation is one seed.

Licenses: code MIT; Tiny Aya Global and both adapters CC-BY-NC-4.0, non-commercial; the synthetic dataset card is MIT (original tickets, not derived weights); Lexsi source-available tools are external dependencies, not copied into the repository.

## Abstract

Use `docs/abstract.md` (under 500 words). It still leads with the 64-ticket compare and now mentions the v2 and Guardrails results in brief.

## Links

- **GitHub repository:** https://github.com/nightmoose/schemasmith (public on publish)
- **Hugging Face collection:** on publish, from `schemasmith publish --namespace nightmoose --public --execute` (no collection exists yet; the publisher creates one titled SchemaSmith). The official form also asks for a separate dataset collection; confirm on the live form whether one combined collection is accepted, or create a second collection for the dataset and eval logs only.
- **Dataset:** https://huggingface.co/datasets/nightmoose/schemasmith-support-tickets (public on publish)
- **SFT adapter:** https://huggingface.co/nightmoose/schemasmith-tiny-aya-sft (public on publish)
- **DPO adapter (final):** https://huggingface.co/nightmoose/schemasmith-tiny-aya-dpo (public on publish)
- **Eval logs, including the hostile run:** https://huggingface.co/datasets/nightmoose/schemasmith-eval-logs (public on publish)
- **Demo:** the recorded run in `space/app.py`, and `docs/demo_transcript.md`. Real outputs, replayed; the page does not load the model. There is no Hugging Face Space yet.
- **Video (optional):** a 76-second silent, captioned replay of the recorded run (no voiceover) already exists. A 3–4 minute Eve-voiced draft is a local file only and is not part of this repo. Upload a video only on publish day, with a separate OK.
- **Write-up:** the repository README.

## Before submitting

- Open every link above in a signed-out window after the public flip.
- Check that the card title reads "Lexsi SchemaSmith".
- Paste the 64-ticket numbers from this file only after confirming they still match `results/compare.json` and `results/hostile.json`.
- Paste the v2 and Guardrails numbers only after confirming they still match `results/aws/sagemaker/v2-summary.json`, `results/aws/sagemaker/eval-partial/eval_from_logs.json`, and `results/aws/guardrails/summary.json`.
- Use the tag list in this file. Follow `PUBLISH_CHECKLIST.md`.

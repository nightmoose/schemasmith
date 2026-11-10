# Submission and gallery text

Exact text for the Lexsi AI Safety & Alignment Hackathon form and gallery card. Prepared Nov 10, 2026. Not submitted. The repo, datasets, adapters, and collection stay private until publish day (Nov 14); fill the two links marked "on publish" then.

## Card

- **Project / gallery card title:** Lexsi SchemaSmith
- **Team name:** Lexsi SchemaSmith
- **Members:** Alex Suarez (GitHub @nightmoose), solo. Use the members field for this, not the title.
- **Track:** Track 1, Fine-tuning and Data Curation. Problem 5, structured output and extraction.
- **Core model:** Tiny Aya Global, `CohereLabs/tiny-aya-global`
- **Lexsi repos used:** CuratorKIT, AlignTune (SFT and DPO), AuditKIT (optional scorer)
- **Tagline (one line):** Made-up order refs and refunds in ticket JSON: 32.9% to 0%, with code, not the model, deciding what is accepted.

## Short description (gallery card, about 60 words)

Lexsi SchemaSmith fine-tunes Tiny Aya Global to turn multilingual support tickets into contract-checked JSON. On 64 held-out tickets, made-up order references and refund amounts fell from 32.9% to 0%, and valid output rose from 46.9% to 100%. A 26-ticket hostile test shows what still breaks: the model followed 4 of 6 prompt injections. A code gate decides what is accepted.

## Long description

**Made-up field values: 32.9% to 0%.** Support tickets land in warehouses and refund queues as JSON. An invented order reference credits the wrong order; a refund on a bug ticket pays out for nothing. Lexsi SchemaSmith fine-tunes Cohere Labs Tiny Aya Global to extract one ticket into one JSON object that passes a data contract, in eight languages.

Results on 64 held-out tickets (base, then SFT, then DPO):

- Unsupported values (ids and amounts not in the ticket): 32.9% (52 of 158), 0% (0 of 106), 0% (0 of 106)
- Schema-valid: 46.9%, 100%, 100%
- Field-level exact match: 71.4%, 98.0%, 98.2%
- Non-English contract-valid: 46.4%, 100%, 100%

Hostile test, 26 hand-written tickets with zero overlap with training (injections, contradictions, off-enum values, mixed languages, garbage, very long threads), base then SFT then DPO:

- Contract-valid: 11, 20, 21 of 26
- Unsupported values: 36.4%, 6.7%, 4.5%
- Injections refused: 0, 2, 2 of 6
- Code gate accepted but wrong: 3, 4, 4

Models suggest, code decides: a contract check and a grounding check accept a row only when it is valid and every id and amount is in the ticket text; anything else is held for a person with the reason. The gate never accepted a made-up value. It did accept injected values that the ticket text itself contained, and the tuned model followed 4 of 6 injections. That is the main open problem, and the README lists every miss.

How: CuratorKIT curates a small synthetic set with the contract as a row filter; AlignTune runs LoRA SFT, then DPO on the SFT-merged weights; the contract validator (with AuditKIT as an optional extra) scores the held-out compare and the hostile set. Each stage ran as a small Hugging Face Job on one L4 GPU. The final adapter takes about 3.7 seconds per ticket there, unbatched.

Limits: synthetic, template-based training and held-out data; small sets and single runs; the DPO setting was chosen after one earlier run on the held-out split; injection resistance was not trained.

Licenses: code MIT; Tiny Aya Global and both adapters CC-BY-NC-4.0, non-commercial; Lexsi source-available tools are external dependencies, not copied into the repository.

## Abstract

Use `docs/abstract.md` as is (under 500 words).

## Links

- **GitHub repository:** https://github.com/nightmoose/schemasmith (public on publish)
- **Hugging Face collection:** on publish, from `schemasmith publish --namespace nightmoose --public --execute`
- **Dataset:** https://huggingface.co/datasets/nightmoose/schemasmith-support-tickets (public on publish)
- **SFT adapter:** https://huggingface.co/nightmoose/schemasmith-tiny-aya-sft (public on publish)
- **DPO adapter (final):** https://huggingface.co/nightmoose/schemasmith-tiny-aya-dpo (public on publish)
- **Eval logs, including the hostile run:** https://huggingface.co/datasets/nightmoose/schemasmith-eval-logs (public on publish)
- **Demo:** the recorded run in `space/app.py`, and `docs/demo_transcript.md`. Real outputs, replayed; the page does not load the model.
- **Video (optional):** a 76-second silent, captioned replay of the recorded run (no voiceover). Upload only on publish day.
- **Write-up:** the repository README.

## Before submitting

- Open every link above in a signed-out window after the public flip.
- Check that the card title reads "Lexsi SchemaSmith".
- Paste the numbers from this file only; they match `results/compare.json` and `results/hostile.json`.

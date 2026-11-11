# SchemaSmith publish checklist (Nov 14)

Do not run any of these flips, uploads, or the submit form before Nov 14, and not without Alex's own OK in his own words. This file is a dry-run checklist only. Nothing in this pull request changes repository or Hugging Face visibility.

Deadline context: Lexsi is scheduled to go public on Nov 14. Official submission deadline is Nov 15, 11:59 PM AoE (Nov 15, 6:59 AM ET). Prefer submitting on Nov 14, afternoon ET, after the public flip has been checked in a signed-out window. The live submit form was not on the site when this checklist was written (Nov 11); use the official "What you'll need" list from https://hackathon.lexsi.ai/ and paste the text from `docs/SUBMISSION.md`.

## 0. Before anything becomes public

- [ ] Alex has said, in his own words, that the GitHub repo, the Hugging Face adapters, dataset, eval logs, and collection may go public, and that the form may be submitted.
- [ ] `main` has this pre-publish PR merged (or the same numbers and cards are on `main`).
- [ ] Repo tests and lint are green on that commit.
- [ ] Re-read `docs/SUBMISSION.md` and the README against the result files. Do not type numbers from memory.
- [ ] Confirm every artifact is still private and ungated before the flip (expected as of the Nov 11 dry run):
  - GitHub `nightmoose/schemasmith` (private)
  - `nightmoose/schemasmith-tiny-aya-sft` (private, ungated, license `cc-by-nc-4.0`)
  - `nightmoose/schemasmith-tiny-aya-dpo` (private, ungated, license `cc-by-nc-4.0`)
  - `nightmoose/schemasmith-support-tickets` (private, ungated; the live card currently has no `license:` field — upload the MIT card from `cards/` as part of the publish step)
  - `nightmoose/schemasmith-eval-logs` (private, ungated, license `mit`)
  - `nightmoose/schemasmith-src` (private wheel repo; see step 3)
  - No SchemaSmith Hugging Face collection existed yet on Nov 11; `schemasmith publish` creates one titled SchemaSmith.

## 1. Upload the updated cards (still private), then flip visibility

From the repository root, with `HF_TOKEN` in the environment (do not print it):

```bash
# Dry run first: writes cards, makes no Hub calls
schemasmith publish --namespace nightmoose --cards-dir cards

# Execute while still private: uploads the cards in cards/ (and any local
# dataset/adapter/eval dirs you pass). Do not add --public yet.
schemasmith publish --namespace nightmoose --execute \
  --dataset-dir data/curated/local \
  --sft-dir OUTPUT_SFT \
  --dpo-dir OUTPUT_DPO \
  --eval-dir reports
```

If the adapters and dataset are already on the Hub and only the cards need a refresh, upload just the card markdown (repo `README.md` on each Hub repo) from `cards/`, still leaving each repo private. Do not use `--public` until the cards look right.

Then, only after Alex's OK, flip visibility. Prefer one command that the tool already supports, so the collection is created public and ungated in the same step:

```bash
schemasmith publish --namespace nightmoose --public --execute \
  --dataset-dir data/curated/local \
  --sft-dir OUTPUT_SFT \
  --dpo-dir OUTPUT_DPO \
  --eval-dir reports
```

`--public` makes the four publish-plan repos and the collection public and ungated. Adapter cards must remain `cc-by-nc-4.0`. Dataset and eval-log cards remain MIT. Do not gate any of these repos.

Equivalent manual flips if the command is not used:

1. Hugging Face: Settings → change each of the four repos from private to public, confirm `gated` is off, confirm the license YAML on each card.
2. Create (or unhide) the collection titled SchemaSmith under `nightmoose`, public, containing those four repos.
3. GitHub: change `nightmoose/schemasmith` from private to public (Settings → Change repository visibility). Do this after the Hub artifacts are public, so the README links resolve.

## 2. Signed-out check

In a signed-out or private window, open:

- https://github.com/nightmoose/schemasmith
- https://huggingface.co/nightmoose/schemasmith-tiny-aya-sft
- https://huggingface.co/nightmoose/schemasmith-tiny-aya-dpo
- https://huggingface.co/datasets/nightmoose/schemasmith-support-tickets
- https://huggingface.co/datasets/nightmoose/schemasmith-eval-logs
- the collection URL printed by the publish command
- the recorded demo (`space/app.py` in the public repo, or a Space if one is added later)

Confirm: no login wall, no gate, cards show the v1 64-ticket numbers and the v2 / Guardrails numbers, licenses are correct, and there is no Astrodata, no pricing or sales pitch, and no mention of bots building the work.

## 3. `schemasmith-src` (the fifth Hub repo)

This private repo holds the install wheel and `aws/job.sh` used by SageMaker and HF Jobs. It is not in the publish plan and has no card or license metadata. Leave it private unless Alex decides it should be public for reproducibility. If it is published, add a short MIT card first (it is code, not an adapter) and do not include weights from Tiny Aya.

## 4. Two collections?

The official page asks for a Hugging Face model collection and a Hugging Face dataset collection, each public. The publisher currently creates one collection, SchemaSmith, that includes the dataset, both adapters, and the eval logs. On publish day, check the live form:

- If one combined collection is accepted, use that URL for both fields and say so in the form.
- If two are required, keep SchemaSmith for the adapters (and optionally the code card) and create a second public collection for `schemasmith-support-tickets` and `schemasmith-eval-logs`. Do not invent a second collection before that is confirmed.

## 5. Intellectual property statement

Official rules (https://hackathon.lexsi.ai/): teams retain ownership of the solutions they develop; organizers may showcase submitted projects for event and promotional purposes; contributions should be unique and the team's own work. Alex must own and, if the form requires it, sign this himself. Draft for him to edit, not for a bot to submit:

> I, Alex Suarez (GitHub @nightmoose, Hugging Face @nightmoose), am the sole member of team Lexsi SchemaSmith. This entry is my original work, developed during the Lexsi AI Safety & Alignment Hackathon. It uses documented open-source and source-available libraries (CuratorKIT, AlignTune, and AuditKIT as external dependencies; Tiny Aya Global under CC-BY-NC-4.0). I retain ownership of the solutions I developed. I understand that the organizers may showcase the submitted project for event and promotional purposes. Repository code is MIT. Adapters derived from Tiny Aya Global are CC-BY-NC-4.0, non-commercial. The synthetic ticket dataset is original work licensed MIT.

Paste the form's exact wording if it differs. Do not add any mention of assistants, bots, or third-party builders.

## 6. Submission form fields

Paste from `docs/SUBMISSION.md` and `docs/abstract.md`. Expected fields, from the official "What you'll need" list and the prepared submission text:

| Field | Value |
| --- | --- |
| Registration email | The email Alex (or the team) registered with |
| Project / gallery title | Lexsi SchemaSmith |
| Team name | Lexsi SchemaSmith |
| Members | Alex Suarez (GitHub @nightmoose), solo |
| Track | Track 1, Fine-tuning and Data Curation |
| Problem statement | Problem 5, structured output and extraction |
| Core model | Tiny Aya Global, `CohereLabs/tiny-aya-global` |
| Lexsi repos used | CuratorKIT, AlignTune (SFT and DPO), AuditKIT (optional scorer) |
| Tags | Tiny Aya, CuratorKIT, AlignTune, AuditKIT, Hugging Face Jobs/Hub, PyTorch, PEFT, Gradio, Amazon SageMaker, Amazon EC2, Amazon Bedrock (Nova + Guardrails) |
| Abstract | `docs/abstract.md` (under 500 words) |
| Write-up | Repository README (HF article optional; `docs/hf_article.md` is a draft and still leads with the 64-ticket compare) |
| GitHub repo | https://github.com/nightmoose/schemasmith (must be public) |
| HF model collection | Collection URL from the publish step (public, ungated) |
| HF dataset collection | Same collection, or a second collection if the form requires it |
| Dataset | https://huggingface.co/datasets/nightmoose/schemasmith-support-tickets |
| SFT adapter | https://huggingface.co/nightmoose/schemasmith-tiny-aya-sft |
| DPO adapter | https://huggingface.co/nightmoose/schemasmith-tiny-aya-dpo |
| Eval logs | https://huggingface.co/datasets/nightmoose/schemasmith-eval-logs |
| Demo | Recorded run in `space/app.py` and `docs/demo_transcript.md` (no model loaded) |
| Video | Optional. Upload only on publish day, with a separate OK |
| IP statement | The statement in section 5, or the form's own wording, signed by Alex |

## 7. After submit

- [ ] Save the confirmation text and a screenshot.
- [ ] Send those, plus the public collection URL and the GitHub URL, to Chief.
- [ ] Do not reset or resubmit unless Alex asks.

## 8. Tag list (final)

Use this exact list on the entry, and only these, unless the form's vocabulary is narrower (then drop rather than invent aliases):

Tiny Aya, CuratorKIT, AlignTune, AuditKIT, Hugging Face Jobs/Hub, PyTorch, PEFT, Gradio, Amazon SageMaker, Amazon EC2, Amazon Bedrock (Nova + Guardrails)

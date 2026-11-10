# Rubric map

Sources read on 2026-11-09:

- Kickoff slides at https://hackathon.lexsi.ai/kickoff-slides/. The judging slide scores write-up clarity, practical usefulness, and code or notebook quality at 10 points each, then an overall review at 100. The same deck lists the validity checks and the nine write-up sections.
- Track list: https://hackathon.lexsi.ai/tracks returned 404. The same Track 1 problem statements are on the kickoff slides and on https://hackathon.lexsi.ai/.

Tiny Aya Global (`CohereLabs/tiny-aya-global`) is the core model. The other six listed Cohere models are not used.

## Scored criteria

| Criterion | Where it is covered | Gap |
| --- | --- | --- |
| Write-up clarity (10) | README opens with the headline (made-up values 32.9% to 0%), then held-out results, hostile results with every miss, then limits, then reproduce steps. | `docs/hf_article.md` is only an optional draft. |
| Practical usefulness (10) | Contract-gated extraction in eight languages, a code gate that holds anything invalid or ungrounded, a recorded demo of real outputs, measured seconds per ticket. | No real ticket corpus. Prompt injection is not solved (4 of 6 followed). |
| Code and notebook quality (10) | Package, tests, CI, SFT and DPO notebooks, HF Jobs for SFT, DPO, held-out eval, and the hostile run, tests that tie README tables to result files. | The notebooks target a free T4; the reported runs used HF Jobs on an L4. |
| Overall review (100) | CuratorKIT, AlignTune SFT, AlignTune DPO, and AuditKIT around one contract and one core model; held-out and hostile results. | Public artifacts are flipped on publish day. |

## Validity checks from the labelling slide

| Check | Status |
| --- | --- |
| Public artifacts | Private until publish day. `schemasmith publish --public --execute` creates public, ungated repos and a public collection. |
| Write-up | README. An HF article is optional. |
| Checklist sections | Problem, repos, model license, data, artifacts, experiments, before/after results, hostile results, and limits are in the README. |
| No duplicate submission | One repo, one core model. |

## Track 1 problems this entry touches

| Problem | Coverage | Gap |
| --- | --- | --- |
| 5 Structured output and extraction | Contract, validator, SFT, DPO, held-out compare, hostile set. This is the core problem. | Synthetic data. |
| 2 Replacing a closed-model task | Measured seconds per ticket on one L4. `schemasmith roi` holds the comparison formula. | No closed-model run was measured, so no comparison number. |
| 7 Preference alignment | Planted chosen and rejected pairs and DPO on the SFT-merged weights. | DPO kept the SFT gains; it did not add to them. |
| 4 Multilingual text | Eight languages; non-English contract-valid 46.4% to 100%; 4 of 4 mixed-language hostile tickets valid. | Romanized and mixed scripts were only tested on four tickets. |
| 8 Data-scarce curation | CuratorKIT filters the synthetic set. | No measured comparison of curated versus unfiltered training. |

# Rubric map

Sources read on 2026-11-09:

- Kickoff slides at https://hackathon.lexsi.ai/kickoff-slides/. The judging slide (the "How judging works" slide, in the stretch the organizers called around slide 30) scores write-up clarity, practical usefulness, and code or notebook quality at 10 points each, then an overall review at 100. The same deck lists the validity checks and the nine write-up sections.
- Track list: https://hackathon.lexsi.ai/tracks returned 404. The same Track 1 problem statements are on the kickoff slides and on https://hackathon.lexsi.ai/.

Tiny Aya Global (`CohereLabs/tiny-aya-global`) is the core model. The other six listed Cohere models are not used.

## Scored criteria

| Criterion | Where it is covered | Gap |
| --- | --- | --- |
| Write-up clarity (10) | This README is the submission write-up: problem, repos, core model and license, data, artifact plan, and TBD experiments. | `docs/hf_article.md` is only an optional draft. Teammates are not tagged. |
| Practical usefulness (10) | Contract-gated extraction for support tickets, eight languages, a validator, and a mock Space. | No real ticket corpus. The optional closed-model comparison is staged and not measured. |
| Code and notebook quality (10) | Package, tests, CI, SFT notebook, DPO notebook, job and publish commands. | Neither notebook has been run on a GPU. |
| Overall review (100) | CuratorKIT, AlignTune SFT, AlignTune DPO, and AuditKIT wired around one contract and one core model. | Before and after numbers are TBD. Public artifacts are not uploaded yet. |

## Validity checks from the labelling slide

| Check | Status |
| --- | --- |
| Public artifacts | Not uploaded. `schemasmith publish --public --execute` creates public, ungated repos and a public collection. Private job uploads do not satisfy this check. |
| Write-up | README. An HF article is optional under the current announcement. |
| Checklist sections | Problem, repos, model license, data, artifact plan, and experiment placeholders are in the README. Findings and before/after results are TBD. |
| No duplicate submission | One repo, one core model. |

## Track 1 problems this entry touches

| Problem | Coverage | Gap |
| --- | --- | --- |
| 5 Structured output and extraction | Contract, validator, SFT, DPO, held-out compare. This is the core problem. | Metrics TBD. |
| 2 Replacing a closed-model task | `schemasmith roi` holds the formula and refuses to fill costs without measurements. | No measured seconds or token counts, so no comparison number. |
| 7 Preference alignment | Planted chosen and rejected pairs and a DPO notebook on Tiny Aya Global. | DPO has not been run. |
| 4 Multilingual text | Generator covers eight languages. The compare reports a non-English contract-valid rate. | That rate is TBD. |
| 8 Data-scarce curation | CuratorKIT filters the synthetic set. | No measured comparison of curated versus unfiltered training. |

## Still open

GPU runs, the results file, the public ungated upload, and any ROI number. None of those are filled with placeholders.

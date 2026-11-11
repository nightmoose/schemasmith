# AWS runs (Nov 10, 2026): v1 re-scored on bigger and blind sets, plus Nova references

Status: **v1 CPU eval and Nova references below are complete.** v2 SageMaker training
and eval later the same day also finished; those numbers are in
`results/aws/sagemaker/README.md` and are now summarized in the top-level README.
Guardrails results are in `results/aws/guardrails/README.md`. The tables in this file
are the v1 CPU / Nova reference runs only.

## What ran where

| Run | Where | Rows | Cost (est.) |
| --- | --- | --- | --- |
| v1 base, SFT, DPO on held-out 256 + hostile v1 (26) + hostile v2 (30) | EC2 c7i.4xlarge, CPU, bf16, batched 16 | 312 per system | ~$0.75 |
| Amazon Nova Micro / Lite / Pro / 2 Lite, same prompt, temperature 0, max 256 tokens | Bedrock, us-east-1 | 312 per model | ~$0.25 |
| Pipeline smoke test (install, data prep, gated-model download) | EC2 c7i.2xlarge | none | ~$0.12 |

## New eval sets

- **Held-out 256**: the test split of 1,280 generated tickets (seed 42), all held-out
  sub-scenarios and phrasings. The original 64-row split is a subset. A one-row change is
  now 0.4 points instead of 1.6.
- **Hostile v2 (30, blind)**: `src/schemasmith/hostile/tickets_v2.py`, written Nov 10 before
  any v2 training data existed: 10 injections (HTML comment, fake authority line, fake tool
  call, fake few-shot example, Italian, Chinese, Japanese, Portuguese, German, French),
  5 contradictions, 4 off-enum, 4 mixed-language, 3 garbage, 4 long. Same author as the
  contract and v1 set, so not independent.
- **Gate v2 screen** (`gate_decision_v2`): also hold a row when the ticket text names a
  contract field, contains a JSON object, or contains a chat-role marker. Written before
  any model output on hostile v2 was scored. It holds 0 of 1,280 generated tickets.

`scripts/aws_summarize.py` re-scores every prediction file with the same code and writes
`summary.json`. Intervals are 95% bootstrap over tickets.

## Held-out 256

| System | Contract-valid | Field exact match | Unsupported values | Non-English valid |
| --- | --- | --- | --- | --- |
| Tiny Aya base | 44.5% [38.7, 50.8] | 72.5% [70.5, 74.6] | 33.3% (210 of 630) | 42.4% |
| v1 SFT | 100% | 97.7% [97.0, 98.3] | 0.5% (2 of 420) | 100% |
| v1 DPO | 100% | 97.7% [97.0, 98.3] | 0.5% (2 of 420) | 100% |
| Nova Micro | 80.1% | 90.2% | 2.3% (10 of 430) | 86.6% |
| Nova Lite | 100% | 96.0% [95.1, 96.8] | 0.2% (1 of 419) | 100% |
| Nova Pro | 100% | 97.1% [96.3, 97.8] | 0% | 100% |
| Nova 2 Lite | 99.2% | 97.0% [96.1, 97.8] | 0% | 99.6% |

What this shows: on 4x the rows, v1 holds up. DPO and SFT are tied (97.7% each), which
matches the earlier read that DPO kept the SFT gains rather than adding to them. The
"0% unsupported" claim from the 64-row run becomes 0.5% (2 of 420) on 256 rows. v1 is
level with Nova Pro and Nova 2 Lite on field match (intervals overlap) and ahead of
Nova Lite and Micro.

Consistency check on the original 64 rows (CPU, batched): base 50.0% valid / 71.7% field
match / 32.9% unsupported; DPO 100% / 98.0% / 0%. The README's GPU run had base 46.9% /
71.4% / 32.9% and DPO 100% / 98.2% / 0%. Batching and CPU numerics move a couple of rows.

## Hostile v1 (26) and hostile v2 (30, blind)

| System | Valid v1 / v2 | Field match v1 / v2 | Injections refused v1 / v2 | Injected value written v1 / v2 | Gate: accepted but wrong v1 / v2 | Gate v2: accepted but wrong v1 / v2 |
| --- | --- | --- | --- | --- | --- | --- |
| Tiny Aya base | 12 / 22 | 71.0% / 77.0% | 0 of 6 / 3 of 10 | 6 of 7 / 7 of 12 | 3 / 9 | 1 / 6 |
| v1 SFT | 21 / 27 | 92.0% / 90.3% | 2 / 5 | 4 / 5 | 4 / 8 | 1 / 6 |
| v1 DPO | 21 / 28 | 92.0% / 91.3% | 2 / 5 | 4 / 5 | 4 / 8 | 1 / 6 |
| Nova Micro | 13 / 13 | 87.7% / 78.6% | 1 / 2 | 4 / 7 | 3 / 5 | 2 / 4 |
| Nova Lite | 23 / 27 | 93.8% / 90.8% | 2 / 3 | 4 / 7 | 3 / 6 | 1 / 4 |
| Nova Pro | 22 / 25 | 93.8% / 85.7% | 2 / 2 | 4 / 7 | 4 / 8 | 2 / 6 |
| Nova 2 Lite | 23 / 27 | 92.6% / 90.8% | 4 / 4 | 1 / 5 | 2 / 3 | 0 / 3 |

What this shows: on the blind set, v1 DPO followed 5 of 10 injections and the original gate
let 8 wrong rows through. The new text screen cuts that to 6, without training anything.
Injection is still the open problem, and the v2 training run is aimed at it. Nova 2 Lite is
the strongest reference on injections; the other Nova models do no better than v1 DPO.

## Cost and speed reference

Measured token counts at us-east-1 on-demand list price (Nova 2 Lite at the US profile price,
$0.33 / $2.75 per 1M tokens): about $20 (Micro), $35 (Lite), $283 (2 Lite), and $469 (Pro)
per 1M tickets. Median latency was 0.5 to 0.7 s per ticket. Tiny Aya on a 16-vCPU CPU box ran
about 3.0 s per ticket batched; the README's L4 number is 3.67 s unbatched.

## SageMaker v2 training (finished)

All four training jobs and all evals finished Nov 10. Final tables, job ids,
instance types, and spend are in `sagemaker/README.md`. Headline v1 CPU / Nova
tables above are unchanged; they are the reference runs, not the v2 scores.

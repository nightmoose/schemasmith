# Lakehouse data contracts on S3 + Iceberg + Glue + Athena (P8)

Write-time ContractGate checks on Stripe-shaped charges, then two Iceberg
tables in the Glue Data Catalog, queried with Athena.

```
generate charges  ──►  contractgate (stripe_charges_raw, in-process)
                            │
              ┌─────────────┴──────────────┐
              ▼                            ▼
     charges_raw (Iceberg)      charges_quarantine (Iceberg)
              │                            │
              └──────── Athena SQL ────────┘
                 counts, leak checks, drift
```

The validator is the same `contractgate` package and `stripe_charges_raw`
contract used in the ContractGate Kinesis and serverless pipeline demos.
It does not call the staging or production ContractGate API, and it does
not touch Fly.

## Why this runs in GitHub Actions

From the NightMoose bot box, the VPC endpoint policy denies
`s3:CreateBucket` and `s3:ListAllMyBuckets` (confirmed as both
`nightmoose-bots` and `nmb-terraform-deploy`). This example therefore
runs in GitHub Actions, which assumes `nmb-terraform-deploy` through
GitHub OIDC (`repo:nightmoose/schemasmith:*` in the role trust,
documented in `nmb-chief/aws/iam/`).

The workflow is branch-scoped: it runs only on push to `aws/p8-lakehouse`
(and `workflow_dispatch`). It never runs on `main`. The same job tears
the stack down in an `always()` step.

## What it creates (and deletes)

| Resource | Name |
|---|---|
| S3 bucket (private, AES256, versioned, HTTPS-only, Block Public Access) | `nmb-lakehouse-065614567889` |
| Athena workgroup (engine v3, 1 GiB scan cap) | `nmb-lakehouse` |
| Glue/Athena database | `nmb_lakehouse` |
| Iceberg tables | `charges_raw`, `charges_quarantine` |

Tags: `nmb:project=P8`, `nmb:stack=lakehouse`, `nmb:cost-alloc=aws-learning-plan`,
`nmb:do-not-prod=true`, `Environment=sandbox`, `Owner=nightmoose`,
`ManagedBy=github-actions`. Region `us-east-1`.

Glue Data Quality (plan item R12 `nmb-glue`) is not used: that role was
not created in P1, and the Athena leak/drift queries cover the same
checks without a crawler or DQ DPU-hours.

## Local check (no AWS)

```bash
pip install -r examples/aws-lakehouse/requirements.txt
python -c "import sys; sys.path.insert(0,'examples/aws-lakehouse/scripts'); import lakehouse as lh; print(lh.generate_and_validate(1000, 42)[2])"
```

The apply / query / destroy path is the GitHub Actions workflow
`.github/workflows/aws-p8-lakehouse.yml`.

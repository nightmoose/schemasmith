#!/usr/bin/env python3
"""Read-only check of Hugging Face license metadata after a card upload.

Does not change visibility, does not upload, and does not print tokens.
Run on publish day after the cards are uploaded:

    python scripts/verify_hf_licenses.py --namespace nightmoose

Expects HF_TOKEN in the environment. Exit code 0 if every license matches:
adapters and eval logs cc-by-nc-4.0, dataset mit.
"""

from __future__ import annotations

import argparse
import os
import sys


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--namespace", default="nightmoose")
    args = parser.parse_args()
    token = os.environ.get("HF_TOKEN")
    if not token:
        print("HF_TOKEN is not set. Nothing was read or changed.", file=sys.stderr)
        return 2
    from huggingface_hub import HfApi

    from schemasmith.hf.licenses import expected_repo_licenses, verify_hub_licenses

    api = HfApi(token=token)
    print("Expected licenses (read-only verify, no visibility change):")
    for item in expected_repo_licenses(args.namespace):
        print(f"  {item['repo_type']} {item['repo_id']}: {item['license']}")
    mismatches = verify_hub_licenses(api, args.namespace)
    if not mismatches:
        print("All four repos match.")
        return 0
    print("Mismatches:", file=sys.stderr)
    for row in mismatches:
        print(
            f"  {row['repo_id']}: expected {row['expected']}, found {row['found']}",
            file=sys.stderr,
        )
    return 1


if __name__ == "__main__":
    raise SystemExit(main())

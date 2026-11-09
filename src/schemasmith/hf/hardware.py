"""Published Hugging Face Jobs rates used as a preview ceiling.

Figures are the t4-small, t4-medium, and l4x1 rows from the Hugging Face
Jobs hardware list (``list_jobs_hardware``), retrieved 2026-11-09. Launch
refreshes the rate with ``list_jobs_hardware`` before it starts a job. A dry
run does not call the Hub.
"""

from __future__ import annotations

PUBLISHED_ON = "2026-11-09"
SOURCE = "https://huggingface.co/docs/huggingface_hub/guides/jobs"

# USD per hour. Smallest GPU flavors only.
HOURLY_USD = {
    "t4-small": 0.40,
    "t4-medium": 0.60,
    "l4x1": 0.80,
}

# L4 is the default: DPO in AlignTune 0.1.32 always runs in bf16, and
# transformers refuses bf16 training on a T4.
DEFAULT_FLAVOR = "l4x1"
DEFAULT_TIMEOUT_SECONDS = 3600
MAX_TIMEOUT_SECONDS = 7200


def ceiling_usd(flavor: str, timeout_seconds: int, hourly: dict[str, float] | None = None) -> float:
    table = HOURLY_USD if hourly is None else hourly
    if flavor not in table:
        allowed = ", ".join(sorted(table))
        raise ValueError(f"flavor must be one of {allowed}")
    if timeout_seconds < 1 or timeout_seconds > MAX_TIMEOUT_SECONDS:
        raise ValueError(f"timeout must be between 1 and {MAX_TIMEOUT_SECONDS} seconds")
    return table[flavor] * (timeout_seconds / 3600)

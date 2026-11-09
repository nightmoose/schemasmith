"""Optional closed-model versus self-hosted comparison.

Dollar inputs are assumptions the operator types in. Timing and token
counts are measurements. If a measurement is missing, this module writes
null costs and does not fill them in.
"""

from __future__ import annotations

from typing import Any

from schemasmith.training import CORE_MODEL_ID


def build_roi(
    *,
    seconds_per_ticket: float | None = None,
    prompt_tokens: float | None = None,
    completion_tokens: float | None = None,
    gpu_hourly_usd: float | None = None,
    closed_usd_per_million_tokens: float | None = None,
    closed_model: str | None = None,
) -> dict[str, Any]:
    """Return a report. Costs stay null until every measured input is present."""
    assumptions = {
        "core_model": CORE_MODEL_ID,
        "closed_model_label": closed_model,
        "gpu_hourly_usd": gpu_hourly_usd,
        "closed_usd_per_million_tokens": closed_usd_per_million_tokens,
        "formula_self_host_per_ticket": "(gpu_hourly_usd / 3600) * seconds_per_ticket",
        "formula_closed_per_ticket": (
            "(prompt_tokens + completion_tokens) / 1_000_000 * closed_usd_per_million_tokens"
        ),
        "note": (
            "gpu_hourly_usd and closed_usd_per_million_tokens are assumptions, "
            "not measurements. seconds_per_ticket and token counts must come from a run."
        ),
    }
    measured = (seconds_per_ticket, prompt_tokens, completion_tokens)
    priced = (gpu_hourly_usd, closed_usd_per_million_tokens)
    if any(value is None for value in measured) or any(value is None for value in priced):
        return {
            "status": "not measured",
            "core_model": CORE_MODEL_ID,
            "assumptions": assumptions,
            "self_host_usd_per_million_tickets": None,
            "closed_usd_per_million_tickets": None,
        }
    assert seconds_per_ticket is not None
    assert prompt_tokens is not None and completion_tokens is not None
    assert gpu_hourly_usd is not None and closed_usd_per_million_tokens is not None
    self_host = (gpu_hourly_usd / 3600) * seconds_per_ticket * 1_000_000
    closed = (
        (prompt_tokens + completion_tokens) / 1_000_000 * closed_usd_per_million_tokens * 1_000_000
    )
    return {
        "status": "computed from supplied measurements and assumptions",
        "core_model": CORE_MODEL_ID,
        "assumptions": assumptions,
        "measurements": {
            "seconds_per_ticket": seconds_per_ticket,
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
        },
        "self_host_usd_per_million_tickets": self_host,
        "closed_usd_per_million_tickets": closed,
    }


def roi_from_compare(payload: dict[str, Any] | None, **assumptions: Any) -> dict[str, Any]:
    """Read optional timing fields off a compare file. Missing fields stay unmeasured."""
    seconds = None
    prompt_tokens = None
    completion_tokens = None
    if payload:
        block = payload.get("roi_measurements")
        if isinstance(block, dict):
            seconds = block.get("seconds_per_ticket")
            prompt_tokens = block.get("prompt_tokens")
            completion_tokens = block.get("completion_tokens")
    return build_roi(
        seconds_per_ticket=seconds,
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
        **assumptions,
    )

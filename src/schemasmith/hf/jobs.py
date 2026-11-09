"""Plan an HF Job and, only with --launch, start it.

The printed ceiling is timeout times the hourly rate. It is not a measured
bill. Dry runs use the published table in ``hardware.py`` and do not contact
the Hub. Launch asks ``list_jobs_hardware`` for the current rate and refuses
to start if that call fails.
"""

from __future__ import annotations

import os
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from schemasmith.hf.hardware import (
    DEFAULT_FLAVOR,
    DEFAULT_TIMEOUT_SECONDS,
    HOURLY_USD,
    SOURCE,
    ceiling_usd,
)

REPO_URL = "https://github.com/nightmoose/schemasmith.git"
IMAGE = "pytorch/pytorch:2.6.0-cuda12.4-cudnn9-devel"
JOBS = ("sft", "dpo", "eval")

HardwareLookup = Callable[[], dict[str, float]]
JobRunner = Callable[..., Any]


@dataclass(frozen=True)
class JobPlan:
    kind: str
    namespace: str
    flavor: str
    timeout_seconds: int
    hourly_usd: float
    ceiling_usd: float
    rate_source: str
    image: str
    command: list[str]
    env: dict[str, str]
    secret_names: tuple[str, ...]

    def public_dict(self) -> dict[str, Any]:
        return {
            "kind": self.kind,
            "namespace": self.namespace,
            "flavor": self.flavor,
            "timeout_seconds": self.timeout_seconds,
            "hourly_usd": self.hourly_usd,
            "ceiling_usd": round(self.ceiling_usd, 4),
            "rate_source": self.rate_source,
            "image": self.image,
            "command": self.command,
            "env": self.env,
            "secret_names": list(self.secret_names),
            "private_repos": private_repo_ids(self.namespace),
        }


def private_repo_ids(namespace: str) -> dict[str, str]:
    _check_namespace(namespace)
    return {
        "dataset": f"{namespace}/schemasmith-support-tickets",
        "sft": f"{namespace}/schemasmith-tiny-aya-sft",
        "dpo": f"{namespace}/schemasmith-tiny-aya-dpo",
        "eval_logs": f"{namespace}/schemasmith-eval-logs",
    }


def build_plan(
    kind: str,
    *,
    namespace: str,
    flavor: str = DEFAULT_FLAVOR,
    timeout_seconds: int = DEFAULT_TIMEOUT_SECONDS,
    repo_ref: str = "main",
    hourly: dict[str, float] | None = None,
    rate_source: str = f"published table {SOURCE} retrieved with the hardware module",
) -> JobPlan:
    if kind not in JOBS:
        raise ValueError(f"kind must be one of {', '.join(JOBS)}")
    _check_namespace(namespace)
    table = HOURLY_USD if hourly is None else hourly
    cost = ceiling_usd(flavor, timeout_seconds, table)
    command = [
        "bash",
        "-lc",
        _remote_shell(kind, repo_ref),
    ]
    return JobPlan(
        kind=kind,
        namespace=namespace,
        flavor=flavor,
        timeout_seconds=timeout_seconds,
        hourly_usd=table[flavor],
        ceiling_usd=cost,
        rate_source=rate_source,
        image=IMAGE,
        command=command,
        env={
            "SCHEMASMITH_NAMESPACE": namespace,
            "SCHEMASMITH_PUSH": "1",
            "SCHEMASMITH_REPO_REF": repo_ref,
        },
        secret_names=("HF_TOKEN",),
    )


def render_plan(plan: JobPlan) -> str:
    lines = [
        f"HF Job preview: {plan.kind}",
        f"flavor: {plan.flavor}",
        f"timeout_seconds: {plan.timeout_seconds}",
        f"hourly_usd: {plan.hourly_usd:.2f}",
        f"ceiling_usd: {plan.ceiling_usd:.2f}",
        "ceiling is timeout times the hourly rate, not a measured bill",
        f"rate_source: {plan.rate_source}",
        f"image: {plan.image}",
        "secrets: HF_TOKEN (value not printed)",
        "visibility of any repo this job writes: private",
        f"command: {plan.command[-1]}",
    ]
    return "\n".join(lines)


def token_from_env() -> str:
    token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    if not token:
        raise RuntimeError(
            "HF_TOKEN is not set. Create a write token and export it. Nothing was launched."
        )
    return token


def launch_plan(
    plan: JobPlan,
    *,
    token: str,
    run_job: JobRunner,
    hardware_lookup: HardwareLookup,
) -> Any:
    """Refresh the rate, print the ceiling, then call ``run_job``."""
    live = hardware_lookup()
    if plan.flavor not in live:
        raise RuntimeError(f"live hardware list has no {plan.flavor}. Not launching.")
    refreshed = build_plan(
        plan.kind,
        namespace=plan.namespace,
        flavor=plan.flavor,
        timeout_seconds=plan.timeout_seconds,
        repo_ref=plan.env["SCHEMASMITH_REPO_REF"],
        hourly=live,
        rate_source="list_jobs_hardware",
    )
    print(render_plan(refreshed))
    print("launching")
    return run_job(
        image=refreshed.image,
        command=refreshed.command,
        flavor=refreshed.flavor,
        timeout=refreshed.timeout_seconds,
        env=refreshed.env,
        secrets={"HF_TOKEN": token},
        namespace=refreshed.namespace,
        token=token,
    )


def hourly_from_unit(unit_cost_usd: float, unit_label: str) -> float:
    """Convert a Hub hardware unit price to USD per hour."""
    if unit_label == "minute":
        return unit_cost_usd * 60
    if unit_label == "hour":
        return unit_cost_usd
    raise RuntimeError(f"unsupported hardware unit {unit_label}")


def live_hardware_rates() -> dict[str, float]:
    """Call the Hub. Used only by the launch path."""
    from huggingface_hub import list_jobs_hardware

    rates: dict[str, float] = {}
    for item in list_jobs_hardware():
        if item.name not in HOURLY_USD:
            continue
        rates[item.name] = hourly_from_unit(item.unit_cost_usd, item.unit_label)
    if not rates:
        raise RuntimeError("list_jobs_hardware returned no t4 rates. Not launching.")
    return rates


def _remote_shell(kind: str, repo_ref: str) -> str:
    # The token is read from the job secret inside the container. It is not
    # interpolated into this command string.
    return "\n".join(
        [
            "set -euo pipefail",
            f"pip install -q 'git+{REPO_URL}@{repo_ref}'",
            "python -m schemasmith.hf.remote " + kind,
        ]
    )


def _check_namespace(namespace: str) -> None:
    if not namespace or "/" in namespace or namespace.startswith("."):
        raise ValueError("namespace must be a Hugging Face user or org name")

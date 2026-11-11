"""Expected Hugging Face license metadata, and a read-only verifier.

This module never changes repository visibility and never uploads.
"""

from __future__ import annotations

from typing import Any

from schemasmith.hf.cards import HUB_LICENSES
from schemasmith.hf.jobs import private_repo_ids


def expected_repo_licenses(namespace: str) -> list[dict[str, str]]:
    """Publish-plan repos and the license id each must have on the Hub."""
    repos = private_repo_ids(namespace)
    kinds = (
        ("dataset", "dataset"),
        ("sft", "model"),
        ("dpo", "model"),
        ("eval_logs", "dataset"),
    )
    return [
        {
            "key": key,
            "repo_id": repos[key],
            "repo_type": repo_type,
            "license": HUB_LICENSES[key],
        }
        for key, repo_type in kinds
    ]


def _license_of(info: Any) -> str | None:
    card = getattr(info, "card_data", None) or getattr(info, "cardData", None)
    value = None
    if card is not None:
        value = getattr(card, "license", None)
        if value is None and isinstance(card, dict):
            value = card.get("license")
    if value is None:
        value = getattr(info, "license", None)
    if value is None:
        return None
    if isinstance(value, list):
        return ",".join(str(v) for v in value)
    return str(value)


def read_hub_license(api: Any, repo_id: str, repo_type: str) -> str | None:
    """Read ``license`` metadata from a Hub repo. Does not mutate the repo."""
    if repo_type == "model":
        info = api.model_info(repo_id)
    elif repo_type == "dataset":
        info = api.dataset_info(repo_id)
    else:
        raise ValueError(f"unsupported repo_type: {repo_type}")
    return _license_of(info)


def verify_hub_licenses(api: Any, namespace: str) -> list[dict[str, str | None]]:
    """Compare each publish-plan repo's Hub license to ``HUB_LICENSES``.

    Returns one dict per mismatch. An empty list means every license matched.
    Does not change visibility or upload anything.
    """
    mismatches: list[dict[str, str | None]] = []
    for item in expected_repo_licenses(namespace):
        found = read_hub_license(api, item["repo_id"], item["repo_type"])
        if found != item["license"]:
            mismatches.append(
                {
                    "repo_id": item["repo_id"],
                    "repo_type": item["repo_type"],
                    "expected": item["license"],
                    "found": found,
                }
            )
    return mismatches

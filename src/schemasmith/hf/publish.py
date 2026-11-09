"""Dry-run Hub upload. Nothing is sent unless execute=True.

Repos are private unless ``public`` is set. The collection follows the same
flag. ``create_collection`` on the Hub defaults to public, so this module
always passes ``private`` explicitly.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from schemasmith.hf.cards import (
    adapter_card,
    collection_description,
    dataset_card,
    eval_logs_card,
)
from schemasmith.hf.jobs import private_repo_ids


@dataclass
class UploadItem:
    repo_id: str
    repo_type: str
    private: bool
    gated: bool
    card_markdown: str
    local_dir: str | None = None


@dataclass
class PublishPlan:
    namespace: str
    public: bool
    execute: bool
    items: list[UploadItem] = field(default_factory=list)
    collection_title: str = "SchemaSmith"
    collection_private: bool = True

    def public_dict(self) -> dict[str, Any]:
        return {
            "namespace": self.namespace,
            "public": self.public,
            "execute": self.execute,
            "collection_title": self.collection_title,
            "collection_private": self.collection_private,
            "items": [
                {
                    "repo_id": item.repo_id,
                    "repo_type": item.repo_type,
                    "private": item.private,
                    "gated": item.gated,
                    "local_dir": item.local_dir,
                }
                for item in self.items
            ],
        }


def build_publish_plan(
    *,
    namespace: str,
    dataset_dir: Path | None = None,
    sft_dir: Path | None = None,
    dpo_dir: Path | None = None,
    eval_dir: Path | None = None,
    public: bool = False,
    execute: bool = False,
) -> PublishPlan:
    repos = private_repo_ids(namespace)
    private = not public
    items = [
        UploadItem(repos["dataset"], "dataset", private, False, dataset_card(), _dir(dataset_dir)),
        UploadItem(repos["sft"], "model", private, False, adapter_card("sft"), _dir(sft_dir)),
        UploadItem(repos["dpo"], "model", private, False, adapter_card("dpo"), _dir(dpo_dir)),
        UploadItem(repos["eval_logs"], "dataset", private, False, eval_logs_card(), _dir(eval_dir)),
    ]
    return PublishPlan(
        namespace=namespace,
        public=public,
        execute=execute,
        items=items,
        collection_private=private,
    )


def write_cards(plan: PublishPlan, dest: Path) -> None:
    dest.mkdir(parents=True, exist_ok=True)
    for item in plan.items:
        name = item.repo_id.split("/", 1)[1] + ".md"
        (dest / name).write_text(item.card_markdown, encoding="utf-8")


def render_plan(plan: PublishPlan) -> str:
    lines = [
        "HF publish preview",
        f"execute: {plan.execute}",
        f"public: {plan.public}",
        f"collection: {plan.collection_title} private={plan.collection_private}",
    ]
    for item in plan.items:
        local = item.local_dir or "(card only)"
        lines.append(
            f"{item.repo_type} {item.repo_id} private={item.private} "
            f"gated={item.gated} local={local}"
        )
    if not plan.execute:
        lines.append("dry run: no Hub calls")
    return "\n".join(lines)


def execute_plan(plan: PublishPlan, api: Any) -> None:
    """Create private or public repos and a collection. ``api`` is an HfApi."""
    if not plan.execute:
        raise RuntimeError("execute_plan called on a dry run")
    for item in plan.items:
        if item.gated:
            raise RuntimeError("refusing to create a gated repo")
        api.create_repo(
            repo_id=item.repo_id,
            repo_type=item.repo_type,
            private=item.private,
            exist_ok=True,
        )
        card_path_note = "README.md"
        api.upload_file(
            path_or_fileobj=item.card_markdown.encode("utf-8"),
            path_in_repo=card_path_note,
            repo_id=item.repo_id,
            repo_type=item.repo_type,
        )
        if item.local_dir:
            api.upload_folder(
                folder_path=item.local_dir,
                repo_id=item.repo_id,
                repo_type=item.repo_type,
            )
    collection = api.create_collection(
        title=plan.collection_title,
        namespace=plan.namespace,
        description=collection_description(),
        private=plan.collection_private,
        exist_ok=True,
    )
    slug = collection.slug
    for item in plan.items:
        api.add_collection_item(
            collection_slug=slug,
            item_id=item.repo_id,
            item_type=item.repo_type,
            exists_ok=True,
        )


HubFactory = Callable[[str], Any]


def publish(
    plan: PublishPlan,
    *,
    token: str | None,
    hub_factory: HubFactory | None = None,
    cards_dir: Path | None = None,
) -> str:
    if cards_dir is not None:
        write_cards(plan, cards_dir)
    text = render_plan(plan)
    print(text)
    if not plan.execute:
        return text
    if not token:
        raise RuntimeError("HF_TOKEN is not set. Nothing was uploaded.")
    factory = hub_factory or _default_hub
    execute_plan(plan, factory(token))
    return text


def _default_hub(token: str) -> Any:
    from huggingface_hub import HfApi

    return HfApi(token=token)


def _dir(path: Path | None) -> str | None:
    if path is None:
        return None
    return str(path)

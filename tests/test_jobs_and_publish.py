import json

import pytest

from schemasmith.hf.cards import adapter_card, collection_description, dataset_card
from schemasmith.hf.hardware import ceiling_usd
from schemasmith.hf.jobs import build_plan, hourly_from_unit, launch_plan, render_plan
from schemasmith.hf.publish import build_publish_plan, execute_plan, publish


def test_job_preview_uses_the_small_gpu_and_hides_the_token(monkeypatch):
    monkeypatch.setenv("HF_TOKEN", "test-token-not-a-real-key")
    plan = build_plan("sft", namespace="example-user", timeout_seconds=3600)
    text = render_plan(plan)
    assert plan.flavor == "l4x1"
    assert plan.ceiling_usd == pytest.approx(0.80)
    assert "test-token-not-a-real-key" not in text
    assert "HF_TOKEN" not in plan.command[-1]
    assert "private" in text
    assert ceiling_usd("t4-medium", 1800) == pytest.approx(0.30)
    assert hourly_from_unit(0.0067, "minute") == pytest.approx(0.402)


def test_launch_refreshes_the_rate_then_calls_run_job():
    plan = build_plan("eval", namespace="example-user", timeout_seconds=1800)
    seen = {}

    def run_job(**kwargs):
        seen.update(kwargs)
        return {"id": "job-1"}

    def lookup():
        return {"t4-small": 0.5, "t4-medium": 0.6, "l4x1": 0.8}

    launch_plan(plan, token="secret-token", run_job=run_job, hardware_lookup=lookup)
    assert seen["secrets"] == {"HF_TOKEN": "secret-token"}
    assert seen["token"] == "secret-token"
    assert seen["namespace"] == "example-user"
    assert seen["flavor"] == "l4x1"
    assert seen["timeout"] == 1800
    assert seen["env"]["SCHEMASMITH_PUSH"] == "1"
    assert "secret-token" not in json.dumps(seen["command"])


def test_launch_refuses_a_price_list_without_the_flavor():
    plan = build_plan("dpo", namespace="example-user")
    with pytest.raises(RuntimeError, match="Not launching"):
        launch_plan(
            plan,
            token="secret-token",
            run_job=lambda **_kwargs: None,
            hardware_lookup=lambda: {},
        )


def test_publish_dry_run_is_private_and_makes_no_hub_call(tmp_path):
    plan = build_publish_plan(namespace="example-user", public=False, execute=False)
    called = {"n": 0}

    def factory(_token):
        called["n"] += 1
        raise AssertionError("dry run must not build a client")

    text = publish(plan, token=None, hub_factory=factory, cards_dir=tmp_path / "cards")
    assert called["n"] == 0
    assert "dry run" in text
    assert "private=True" in text
    card = (tmp_path / "cards" / "schemasmith-tiny-aya-sft.md").read_text(encoding="utf-8")
    assert "CC-BY-NC-4.0" in card
    assert "TBD" not in card
    assert len(collection_description()) <= 150
    assert "MIT" in dataset_card()
    assert "CC-BY-NC-4.0" in adapter_card("dpo")


def test_public_flag_flips_visibility_only_when_asked():
    private_plan = build_publish_plan(namespace="example-user")
    public_plan = build_publish_plan(namespace="example-user", public=True, execute=True)
    assert all(item.private for item in private_plan.items)
    assert private_plan.collection_private is True
    assert all(item.private is False for item in public_plan.items)
    assert all(item.gated is False for item in public_plan.items)
    assert public_plan.collection_private is False
    assert "CC-BY-NC-4.0" in public_plan.items[1].card_markdown
    assert "Tiny Aya Global" in public_plan.items[1].card_markdown


def test_execute_passes_private_and_does_not_default_the_collection():
    plan = build_publish_plan(namespace="example-user", execute=True)
    calls = []

    class Api:
        def create_repo(self, **kwargs):
            calls.append(("repo", kwargs))

        def upload_file(self, **kwargs):
            calls.append(("file", kwargs["repo_id"]))

        def upload_folder(self, **kwargs):
            calls.append(("folder", kwargs))

        def create_collection(self, **kwargs):
            calls.append(("collection", kwargs))

            class Collection:
                slug = "example-user/schemasmith"

            return Collection()

        def add_collection_item(self, **kwargs):
            calls.append(("item", kwargs["item_id"], kwargs["item_type"]))

    execute_plan(plan, Api())
    collection = [item for item in calls if item[0] == "collection"][0][1]
    assert collection["private"] is True
    assert collection["namespace"] == "example-user"
    repos = [item[1] for item in calls if item[0] == "repo"]
    assert all(item["private"] is True for item in repos)


def test_job_installs_the_private_wheel_not_github():
    plan = build_plan("sft", namespace="example-user", n_tickets=240)
    shell = plan.command[-1]
    assert "github.com" not in shell
    assert "git+" not in shell
    assert "example-user/schemasmith-src" in shell
    assert "pytorch/pytorch:2.13.0" in plan.image
    assert plan.env["PIP_BREAK_SYSTEM_PACKAGES"] == "1"
    assert plan.env["SCHEMASMITH_N"] == "240"
    assert plan.secret_names == ("HF_TOKEN",)


def test_upload_source_wheel_creates_a_private_repo(tmp_path):
    from schemasmith.hf.jobs import WHEEL, upload_source_wheel

    calls = []

    class FakeApi:
        def create_repo(self, **kwargs):
            calls.append(("create", kwargs))

        def upload_file(self, **kwargs):
            calls.append(("upload", kwargs))

    wheel = tmp_path / WHEEL
    wheel.write_bytes(b"x")
    repo = upload_source_wheel("example-user", str(wheel), "tok", api=FakeApi())
    assert repo == "example-user/schemasmith-src"
    assert calls[0][1]["private"] is True
    assert calls[1][1]["path_in_repo"] == WHEEL

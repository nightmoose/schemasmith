import json

from schemasmith.contract.validator import amount_in_text, extract_json_object, validate_instance
from schemasmith.data.generator import generate_tickets
from schemasmith.data.preferences import (
    FAILURE_MODES,
    INVENTED_REFUND,
    build_preference_pairs,
    chat_format_dpo_dir,
    render_chat_pair,
    write_dpo_dataset,
)


def test_pairs_cover_failure_modes_and_keep_chosen_valid():
    records = generate_tickets(10, seed=4)
    pairs = build_preference_pairs(records, seed=4)
    again = build_preference_pairs(records, seed=4)
    assert [row["failure_mode"] for row in pairs] == [row["failure_mode"] for row in again]
    assert set(row["failure_mode"] for row in pairs) == set(FAILURE_MODES)
    for record, pair in zip(records, pairs, strict=True):
        chosen = json.loads(pair["chosen"])
        ok, errors = validate_instance(chosen)
        assert ok, errors
        assert pair["chosen"] != pair["rejected"]
        assert record.text in pair["prompt"]
        if pair["failure_mode"] == "broken_json":
            parsed, err = extract_json_object(pair["rejected"])
            assert parsed is None and err
        elif pair["failure_mode"] == "missing_field":
            parsed, _err = extract_json_object(pair["rejected"])
            assert "language" not in parsed
            assert not validate_instance(parsed)[0]
        elif pair["failure_mode"] == "invented_field":
            parsed, _err = extract_json_object(pair["rejected"])
            assert "courier_note" in parsed
            assert not validate_instance(parsed)[0]
        elif pair["failure_mode"] == "wrong_language":
            parsed, _err = extract_json_object(pair["rejected"])
            assert parsed["language"] != record.language
        elif pair["failure_mode"] == "invented_refund":
            parsed, _err = extract_json_object(pair["rejected"])
            assert parsed["refund_amount_usd"] == INVENTED_REFUND
            assert not amount_in_text(INVENTED_REFUND, record.text)


def test_dpo_export_holds_out_validation(tmp_path):
    summary = write_dpo_dataset(generate_tickets(16, seed=2), tmp_path / "dpo", seed=2)
    train_path = tmp_path / "dpo" / "train" / "dpo.jsonl"
    val_path = tmp_path / "dpo" / "validation" / "dpo.jsonl"
    train = train_path.read_text(encoding="utf-8").strip().splitlines()
    validation = val_path.read_text(encoding="utf-8").strip().splitlines()
    assert summary["split_counts"]["train"] == len(train)
    assert summary["split_counts"]["validation"] == len(validation) >= 1
    row = json.loads(train[0])
    assert set(row) == {"prompt", "chosen", "rejected"}
    card = (tmp_path / "dpo" / "README.md").read_text(encoding="utf-8")
    assert "config_name: dpo" in card


class _FakeChatTokenizer:
    bos_token = "<BOS>"
    eos_token = "<EOT>"

    def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=False):
        text = "<BOS>"
        for message in messages:
            if message["role"] == "user":
                text += f"<U>{message['content']}<EOT>"
            else:
                text += f"<A><R>{message['content']}</R><EOT>"
        if add_generation_prompt:
            text += "<A><R>"
        return text


def test_render_chat_pair_uses_the_template_without_double_specials():
    row = {"prompt": "P", "chosen": "{}", "rejected": "{"}
    out = render_chat_pair(row, _FakeChatTokenizer())
    assert out == {"prompt": "<U>P<EOT><A><R>", "chosen": "{}</R>", "rejected": "{</R>"}


def test_chat_format_dpo_dir_rewrites_both_splits(tmp_path):
    write_dpo_dataset(generate_tickets(16, seed=2), tmp_path / "dpo", seed=2)
    count = chat_format_dpo_dir(tmp_path / "dpo", _FakeChatTokenizer(), tmp_path / "chat")
    assert count == 16
    assert (tmp_path / "chat" / "README.md").exists()
    first = json.loads((tmp_path / "chat" / "train" / "dpo.jsonl").read_text().splitlines()[0])
    assert first["prompt"].startswith("<U>") and first["prompt"].endswith("<A><R>")
    assert first["chosen"].endswith("</R>")

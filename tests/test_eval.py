import json

from schemasmith.data.generator import generate_tickets
from schemasmith.eval.auditkit_bridge import auditkit_available, run_auditkit
from schemasmith.eval.harness import evaluate_records
from schemasmith.training import MODEL_ID, lora_sft_kwargs


def test_echo_gold_scores_perfect_on_the_contract():
    records = generate_tickets(8, seed=9)
    card = evaluate_records(
        records,
        generate=lambda texts: [json.dumps(record.gold) for record in records],
    )
    assert card.parse_rate == 1
    assert card.schema_valid_rate == 1
    assert card.field_exact_match == 1
    assert card.unsupported_value_rate == 0
    assert card.n_non_english == 7
    assert card.non_english_contract_valid_rate == 1


def test_garbage_outputs_score_zero():
    records = generate_tickets(4, seed=9)
    card = evaluate_records(records, outputs=["nope"] * 4)
    assert card.parse_rate == 0
    assert card.schema_valid_rate == 0
    assert card.field_exact_match == 0
    assert card.unsupported_value_rate is None


def test_auditkit_bridge_skips_when_missing():
    records = generate_tickets(2, seed=1)
    if auditkit_available():
        result = run_auditkit(records, [json.dumps(row.gold) for row in records])
        assert result is not None
        assert result["headline"]["json_schema_valid"] == 1
    else:
        assert run_auditkit(records, ["{}", "{}"]) is None


def test_lora_kwargs_stay_on_the_trl_backend():
    kwargs = lora_sft_kwargs(dataset_dir="data/out", output_dir="out/lora")
    assert kwargs["model_name"] == MODEL_ID
    assert kwargs["backend"] == "trl"
    assert kwargs["subset"] == "sft_alpaca"
    assert kwargs["use_peft"] is True
    assert kwargs["quantization"]["load_in_4bit"] is True
    assert "unsloth" not in json.dumps(kwargs).lower()
    assert kwargs["max_steps"] == 84  # 219 rows, batch 8, 3 epochs


def test_generation_stops_on_chat_turn_end_tokens():
    from schemasmith.eval.generate import stop_token_ids

    class Tok:
        eos_token_id = 6
        unk_token_id = 4
        ids = {"<|END_RESPONSE|>": 261001, "<|END_OF_TURN_TOKEN|>": 6}

        def convert_tokens_to_ids(self, token):
            return self.ids.get(token, self.unk_token_id)

    assert stop_token_ids(Tok(), 3) == [3, 6, 261001]
    assert stop_token_ids(Tok(), [3, 7]) == [3, 6, 7, 261001]


def test_dpo_is_one_gentle_epoch():
    from schemasmith.training import lora_dpo_kwargs

    kwargs = lora_dpo_kwargs(dataset_dir="d", output_dir="o", model_name="m")
    assert kwargs["num_epochs"] == 1
    assert kwargs["max_steps"] == 24  # 192 pairs, batch 8, 1 epoch
    assert kwargs["learning_rate"] == 2e-5

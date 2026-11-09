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
    assert kwargs["max_steps"] == 69  # 180 rows, batch 8, 3 epochs

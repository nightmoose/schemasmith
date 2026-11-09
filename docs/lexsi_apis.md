# Lexsi tool calls

Pinned versions are in `requirements-lexsi.txt`. None of these packages are copied into this repository.

## Confirmed by running the package

CuratorKIT **0.1.23** and AuditKIT **1.1.6** were installed in a throwaway directory and called. They are not imported when `schemasmith` imports.

### CuratorKIT

Executed `Curator(CuratorConfig(...)).run()` on local alpaca JSONL, with no `llm_model` and no `generation_task`.

- `format="alpaca"`, `dedup="exact"`, `clean=True`, `schema_gate=True`, `min_tokens=3` ran.
- `export_formats=["alpaca", "messages"]` wrote `sft_alpaca.jsonl` and `sft_messages.jsonl`.
- `output_split={"train": 0.67, "validation": 0.33}` created `train/` and `validation/` and a dataset-card `README.md` whose `sft_messages` config points at those files.
- With no `output_split`, the same files were written at the output root and the README listed a single `train` split.
- Exact dedup reduced 3 identical-pair rows to 2. `rejected.jsonl` stayed empty for that drop. The drop showed up in `stage_counts["ExactDeduplicator"]`.
- `preprocessing_fn` that returned `None` rejected the row. `rejection_reason` was `preprocessing_fn_returned_none` and `rejected_count` on the reader stage was 1.
- `result.passed` and `result.rejected` are lists, not integers. Counts in this repo use `len`.
- Messages export joins `instruction` and `input` with a blank line into the user turn. The local exporter matches that join.
- `clean_fields=["instruction", "input"]` on an 8-row seed-1 run left both the ticket text and the gold JSON equal to the input records. `tests/test_lexsi_optional.py` runs the same call on 24 rows plus planted rejects and checks that every exported gold object still validates.

### AuditKIT

Executed `auditkit.evaluate` (1.1.6) on two samples with a callable model.

- The argument is `scorers`. A README snippet that passes `metrics=` does not match this signature.
- `ak.IsJson`, `ak.JsonSchemaValid`, and `ak.FieldF1` are registered. On the two-row fixture the headline was `is_json` 0.5, `json_schema_valid` 0.5, `field_f1` 0.5, because the second string was not JSON.
- `JsonSchemaValid` accepts a schema dict or a path. `strip_fences` defaults to false. `backend="auto"` used the minimal backend when `jsonschema` was not installed. This project passes `backend="jsonschema"` and `strip_fences=True`.
- `IsJson.score` calls `json.loads` on the raw string. It does not strip fences.
- `result.headline` is a dict. `result.summary()` is a short text report. `result.fingerprint` is a hex string.

`ak.evaluate(..., "hf:CohereLabs/tiny-aya-global", scorers=[...])` is the documented way to score that model. It was not executed. It needs the AuditKIT transformers extra and a GPU.

## Confirmed by reading AlignTune 0.1.32

The 0.1.32 sdist (not a training run) shows:

- `LICENSE` starts with `MIT License`.
- PyPI `requires_dist` includes `curatorkit[connectors]>=0.1.15` as a core dependency.
- `create_sft_trainer(model_name, dataset_name, backend="trl", ..., use_peft=True, lora_r, lora_alpha, lora_dropout, lora_target_modules, quantization, bf16, gradient_checkpointing, attn_implementation, subset, max_steps, num_epochs, seed, eval_strategy, load_best_model_at_end, save_strategy, save_steps, logging_steps, report_to, warmup_ratio, group_by_length)` threads those names into the SFT config. `use_peft` defaults to false, so the notebook sets it true.
- `examples/sft_trl_1.py` passes `backend="trl"`, `use_peft=True`, `lora_target_modules="all-linear"`, and a 4-bit `quantization` dict with `load_in_4bit`, `bnb_4bit_use_double_quant`, `bnb_4bit_quant_type="nf4"`, and `bnb_4bit_compute_dtype="bfloat16"`.
- `dataset_builder.py` loads a Hugging Face dataset with `subset` or `config`. It does not pass `config_name` through to `load_dataset`. CuratorKIT's README example uses `config_name="sft_messages"`. This repo passes `subset="sft_alpaca"` instead.
- `AlpacaAdapter` detects `instruction` and `output`, and folds `input` into the prompt. A table whose only column is `messages` is not one of the detected formats (`custom` then raises unless a mapping is supplied). CuratorKIT still writes `sft_messages.jsonl` for later use.
- The notebook must not select the Unsloth backend. AlignTune vendors Unsloth under its own license, separate from the MIT code we call.

`create_sft_trainer(...).train()` was not executed. No adapter was produced.

## Assumed, not executed

- `datasets.load_dataset(curatorkit_output_dir, "sft_alpaca")` succeeds on the README CuratorKIT writes. The README shape was inspected. `load_dataset` was not called.
- AlignTune's TRL trainer accepts `eval_strategy="no"` together with `load_best_model_at_end=False`, and `attn_implementation="sdpa"`, on a T4 with the 4-bit dict above. The factory stores those fields. A GPU run was not done.
- When both `max_steps` and `num_epochs` are set, the run stops at `max_steps`. That is how the example script is written. It was not timed.
- Tiny Aya's chat template works with the strings AlpacaAdapter builds. The model card shows `apply_chat_template` for this model id. This repo has not loaded the weights.
- A free T4 has enough memory for the 4-bit LoRA settings in `lora_sft_kwargs`. That is a sizing choice, not a measurement.

## Not used

No paid model API. No generator LLM inside CuratorKIT. No Unsloth backend. No AuditKIT `hf:` run.

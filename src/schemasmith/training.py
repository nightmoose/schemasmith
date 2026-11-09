"""LoRA SFT and DPO settings for AlignTune's TRL backend.

This module does not import AlignTune. The notebook passes the dict to
``create_sft_trainer``. Keyword names were read from AlignTune 0.1.32
``create_sft_trainer`` (sdist). Both calls were run on HF Jobs (L4).

``subset`` is ``sft_alpaca`` because that release's dataset adapter detects
``instruction`` plus ``output``. A messages-only table is not detected.
CuratorKIT's README example passes ``config_name="sft_messages"``;
``dataset_builder.py`` in 0.1.32 reads ``subset`` or ``config``, not
``config_name``.

``backend`` is ``trl``. The Unsloth backend is not used.
"""

from __future__ import annotations

import math
from typing import Any

# Core submission model. The other six Cohere hackathon models are not used.
MODEL_ID = "CohereLabs/tiny-aya-global"
CORE_MODEL_ID = MODEL_ID

# Default data size for GPU runs. 320 rows gives a 219 / 37 / 64 split with
# held-out templates in test.
DEFAULT_N = 320
SFT_EPOCHS = 3
DPO_EPOCHS = 1
BATCH_SIZE = 1
GRAD_ACCUM = 8


def steps_for(train_rows: int, epochs: int) -> int:
    """Optimizer steps for ``epochs`` passes over ``train_rows`` rows."""
    if train_rows < 1:
        raise ValueError("train_rows must be at least 1")
    return math.ceil(train_rows / (BATCH_SIZE * GRAD_ACCUM)) * epochs


def lora_sft_kwargs(*, dataset_dir: str, output_dir: str, train_rows: int = 219) -> dict[str, Any]:
    """Arguments for ``aligntune.core.backend_factory.create_sft_trainer``.

    AlignTune turns the alpaca rows into a user / assistant ``messages`` pair
    (user turn is instruction, blank line, ticket) and TRL tokenizes it with
    the model chat template. That is the same format the compare prompts with.
    Steps are about ``SFT_EPOCHS`` passes over ``train_rows``.
    """
    steps = steps_for(train_rows, SFT_EPOCHS)
    return {
        "model_name": MODEL_ID,
        "dataset_name": dataset_dir,
        "subset": "sft_alpaca",
        "backend": "trl",
        "output_dir": output_dir,
        "num_epochs": SFT_EPOCHS,
        "max_steps": steps,
        "batch_size": BATCH_SIZE,
        "gradient_accumulation_steps": GRAD_ACCUM,
        "learning_rate": 2e-4,
        "max_seq_length": 768,
        "use_peft": True,
        "lora_r": 16,
        "lora_alpha": 32,
        "lora_dropout": 0.05,
        "lora_target_modules": "all-linear",
        "quantization": {
            "load_in_4bit": True,
            "bnb_4bit_use_double_quant": True,
            "bnb_4bit_quant_type": "nf4",
            "bnb_4bit_compute_dtype": "bfloat16",
        },
        "bf16": True,
        "gradient_checkpointing": True,
        "attn_implementation": "sdpa",
        "seed": 42,
        "eval_strategy": "no",
        "load_best_model_at_end": False,
        "save_strategy": "steps",
        "save_steps": steps,
        "logging_steps": 5,
        "report_to": "none",
        "warmup_ratio": 0.03,
        "group_by_length": False,
    }


def lora_dpo_kwargs(
    *,
    dataset_dir: str,
    output_dir: str,
    model_name: str,
    train_rows: int = 192,
) -> dict[str, Any]:
    """Arguments for ``aligntune.core.backend_factory.create_rl_trainer``.

    ``model_name`` should be the base model, or a directory of weights that
    already include the SFT update. AlignTune 0.1.32 then applies a new LoRA.
    It does not take a separate adapter-resume argument. ``config_name`` is
    passed because the RL data loader forwards it to ``load_dataset``.
    ``attn_implementation`` and ``bf16`` are not read by ``create_rl_trainer``
    in that sdist, so they are omitted.

    Steps are about ``DPO_EPOCHS`` passes over ``train_rows`` pairs. Prompts
    and responses should already carry the chat template; see
    ``schemasmith.data.preferences.chat_format_dpo_dir``.
    """
    steps = steps_for(train_rows, DPO_EPOCHS)
    return {
        "model_name": model_name,
        "dataset_name": dataset_dir,
        "config_name": "dpo",
        "algorithm": "dpo",
        "backend": "trl",
        "output_dir": output_dir,
        "num_epochs": DPO_EPOCHS,
        "max_steps": steps,
        "batch_size": BATCH_SIZE,
        "gradient_accumulation_steps": GRAD_ACCUM,
        # 5e-5 for 2 epochs pushed the policy off valid JSON on the
        # held-out compare. One epoch at 2e-5 keeps it closer to SFT.
        "learning_rate": 2e-5,
        "beta": 0.1,
        "max_seq_length": 768,
        "use_peft": True,
        "lora_r": 16,
        "lora_alpha": 32,
        "lora_dropout": 0.05,
        "lora_target_modules": "all-linear",
        "quantization": {
            "load_in_4bit": True,
            "bnb_4bit_use_double_quant": True,
            "bnb_4bit_quant_type": "nf4",
            "bnb_4bit_compute_dtype": "bfloat16",
        },
        "gradient_checkpointing": True,
        "seed": 42,
        "eval_strategy": "no",
        "load_best_model_at_end": False,
        "save_strategy": "steps",
        "save_steps": steps,
        "logging_steps": 5,
        "report_to": "none",
        "warmup_ratio": 0.03,
        "group_by_length": False,
    }

"""LoRA SFT settings for AlignTune's TRL backend.

This module does not import AlignTune. The notebook passes the dict to
``create_sft_trainer``. Keyword names were read from AlignTune 0.1.32
``create_sft_trainer`` (sdist). The training call itself has not been run.

``subset`` is ``sft_alpaca`` because that release's dataset adapter detects
``instruction`` plus ``output``. A messages-only table is not detected.
CuratorKIT's README example passes ``config_name="sft_messages"``;
``dataset_builder.py`` in 0.1.32 reads ``subset`` or ``config``, not
``config_name``.

``backend`` is ``trl``. The Unsloth backend is not used.
"""

from __future__ import annotations

from typing import Any

MODEL_ID = "CohereLabs/tiny-aya-global"


def lora_sft_kwargs(*, dataset_dir: str, output_dir: str) -> dict[str, Any]:
    """Arguments for ``aligntune.core.backend_factory.create_sft_trainer``."""
    return {
        "model_name": MODEL_ID,
        "dataset_name": dataset_dir,
        "subset": "sft_alpaca",
        "backend": "trl",
        "output_dir": output_dir,
        "num_epochs": 1,
        "max_steps": 40,
        "batch_size": 1,
        "gradient_accumulation_steps": 8,
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
        "save_steps": 40,
        "logging_steps": 5,
        "report_to": "none",
        "warmup_ratio": 0.03,
        "group_by_length": False,
    }

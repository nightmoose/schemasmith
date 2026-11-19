"""SageMaker real-time handler for SchemaSmith (Tiny Aya + LoRA SFT, then DPO).

Loaded by the AWS PyTorch inference DLC. model.tar.gz holds sft-adapter/,
dpo-adapter/ and code/ (this file + requirements.txt). The gated base model
is pulled from the Hub at startup with a read token the execution role gets
from Secrets Manager (AmazonSageMaker-lexsi-hf); the token is never logged.

Request:  {"prompt": "<user turn: instruction + ticket>", "max_new_tokens": 256}
Response: {"output": "<model text>", "new_tokens": N, "gen_seconds": S}
Generation matches the eval: greedy, chat template, stop on config EOS and
the chat turn-end tokens.
"""

from __future__ import annotations

import json
import os
import time

import torch

MODEL_ID = "CohereLabs/tiny-aya-global"
TURN_END_TOKENS = ("<|END_RESPONSE|>", "<|END_OF_TURN_TOKEN|>")


def _hf_token() -> str | None:
    try:
        import boto3

        sm = boto3.client("secretsmanager", region_name=os.environ.get("AWS_REGION", "us-east-1"))
        return json.loads(sm.get_secret_value(SecretId="AmazonSageMaker-lexsi-hf")["SecretString"])[
            "HF_TOKEN"
        ]
    except Exception as exc:  # noqa: BLE001
        print("hf token lookup failed:", type(exc).__name__)
        return None


def _stops(tok, config_eos) -> list[int]:
    ids: set[int] = set()
    if isinstance(config_eos, int):
        ids.add(config_eos)
    elif config_eos:
        ids.update(int(i) for i in config_eos)
    if tok.eos_token_id is not None:
        ids.add(int(tok.eos_token_id))
    for t in TURN_END_TOKENS:
        i = tok.convert_tokens_to_ids(t)
        if isinstance(i, int) and i != tok.unk_token_id:
            ids.add(i)
    return sorted(ids)


def model_fn(model_dir, context=None):
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    token = _hf_token()
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    # bf16 on Ampere+ GPUs and on CPU; fp16 on older GPUs (T4) without native bf16.
    dtype = torch.bfloat16
    if dev == "cuda" and torch.cuda.get_device_capability()[0] < 8:
        dtype = torch.float16
    t0 = time.perf_counter()
    tok = AutoTokenizer.from_pretrained(MODEL_ID, token=token)
    model = AutoModelForCausalLM.from_pretrained(MODEL_ID, dtype=dtype, device_map=dev, token=token)
    model = PeftModel.from_pretrained(
        model, os.path.join(model_dir, "sft-adapter")
    ).merge_and_unload()
    model = PeftModel.from_pretrained(
        model, os.path.join(model_dir, "dpo-adapter")
    ).merge_and_unload()
    model.eval()
    print(f"model ready on {dev} {dtype} in {time.perf_counter() - t0:.1f}s")
    return {
        "tok": tok,
        "model": model,
        "dev": dev,
        "stops": _stops(tok, model.generation_config.eos_token_id),
    }


def input_fn(body, content_type="application/json"):
    if isinstance(body, (bytes, bytearray)):
        body = body.decode("utf-8")
    return json.loads(body)


def predict_fn(data, m):
    tok, model = m["tok"], m["model"]
    text = tok.apply_chat_template(
        [{"role": "user", "content": data["prompt"]}], tokenize=False, add_generation_prompt=True
    )
    enc = tok(text, return_tensors="pt", add_special_tokens=False).to(m["dev"])
    t = time.perf_counter()
    with torch.no_grad():
        gen = model.generate(
            **enc,
            max_new_tokens=int(data.get("max_new_tokens", 256)),
            do_sample=False,
            eos_token_id=m["stops"],
            pad_token_id=tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id,
        )
    if m["dev"] == "cuda":
        torch.cuda.synchronize()
    new = gen[0, enc["input_ids"].shape[1] :]
    return {
        "output": tok.decode(new, skip_special_tokens=True),
        "new_tokens": int(new.shape[0]),
        "gen_seconds": round(time.perf_counter() - t, 4),
    }


def output_fn(pred, accept="application/json"):
    return json.dumps(pred, ensure_ascii=False)

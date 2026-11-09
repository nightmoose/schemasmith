"""GPU generation for the held-out compare.

Importing this module does not import torch. ``require_cuda_backend`` and
``generate_texts`` raise before writing a results file when CUDA is absent.
"""

from __future__ import annotations

from collections.abc import Sequence


def require_cuda_backend() -> None:
    try:
        import torch
    except ImportError as exc:
        raise RuntimeError(
            "compare needs torch and a CUDA GPU on the machine that runs it. Nothing was written."
        ) from exc
    if not torch.cuda.is_available():
        raise RuntimeError("compare needs a CUDA GPU. Nothing was written.")


def generate_texts(
    prompts: Sequence[str],
    *,
    model_id: str,
    adapter: str | None = None,
    max_new_tokens: int = 256,
) -> list[str]:
    """Generate one completion per prompt. Loads one model, then frees it."""
    require_cuda_backend()
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )
    if adapter:
        from peft import PeftModel

        model = PeftModel.from_pretrained(model, adapter)
    model.eval()
    device = next(model.parameters()).device
    outputs: list[str] = []
    for prompt in prompts:
        messages = [{"role": "user", "content": prompt}]
        input_ids = tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_tensors="pt",
        ).to(device)
        with torch.no_grad():
            generated = model.generate(
                input_ids,
                max_new_tokens=max_new_tokens,
                do_sample=False,
            )
        new_tokens = generated[0, input_ids.shape[-1] :]
        outputs.append(tokenizer.decode(new_tokens, skip_special_tokens=True))
    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return outputs

from pathlib import Path

MODEL_PATH = Path(
    "runs/verge_m2_ministral3_14b_20260824/"
    "reproducibility/model_snapshot.txt"
).read_text(
    encoding="utf-8"
).strip()

_MODEL = None
_TOKENIZER = None


def load_llm():
    global _MODEL, _TOKENIZER

    if _MODEL is not None:
        return _TOKENIZER, _MODEL

    import torch

    from transformers import (
        AutoTokenizer,
        Mistral3ForConditionalGeneration,
    )

    if not torch.cuda.is_available():
        raise RuntimeError(
            "Ministral M2 requires a GPU."
        )

    print("Loading M2 model:", MODEL_PATH)
    print("GPU:", torch.cuda.get_device_name(0))

    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_PATH,
        local_files_only=True,
        fix_mistral_regex=True,
    )

    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    tokenizer.truncation_side = "right"

    model = Mistral3ForConditionalGeneration.from_pretrained(
        MODEL_PATH,
        dtype=torch.bfloat16,
        device_map={"": 0},
        local_files_only=True,
        low_cpu_mem_usage=True,
    )

    model.eval()

    _TOKENIZER = tokenizer
    _MODEL = model

    print("Ministral-3-14B ready.")

    return tokenizer, model


def generate_text(
    prompt: str,
    max_new_tokens: int = 512,
) -> str:

    import torch

    tokenizer, model = load_llm()

    messages = [
        {
            "role": "user",
            "content": str(prompt),
        }
    ]

    formatted = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )

    prompt_ids = tokenizer(
        formatted,
        add_special_tokens=True,
    )["input_ids"]

    context_candidates = []

    config = getattr(
        model,
        "config",
        None,
    )

    for obj in (
        config,
        getattr(config, "text_config", None),
    ):
        value = getattr(
            obj,
            "max_position_embeddings",
            None,
        )

        if (
            isinstance(value, int)
            and 0 < value < 10**9
        ):
            context_candidates.append(value)

    tokenizer_context = getattr(
        tokenizer,
        "model_max_length",
        None,
    )

    if (
        isinstance(tokenizer_context, int)
        and 0 < tokenizer_context < 10**9
    ):
        context_candidates.append(
            tokenizer_context
        )

    context_limit = (
        min(context_candidates)
        if context_candidates
        else 131072
    )

    max_prompt_tokens = (
        context_limit - int(max_new_tokens)
    )

    if len(prompt_ids) > max_prompt_tokens:
        raise RuntimeError(
            "Full Verifier/Refiner prompt exceeds "
            "Ministral context window: "
            f"{len(prompt_ids)} > "
            f"{max_prompt_tokens}"
        )

    inputs = tokenizer(
        formatted,
        return_tensors="pt",
        truncation=False,
    ).to("cuda")

    used = int(
        inputs["input_ids"].shape[1]
    )

    with torch.inference_mode():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )

    generated = outputs[0][used:]

    return tokenizer.decode(
        generated,
        skip_special_tokens=True,
    ).strip()

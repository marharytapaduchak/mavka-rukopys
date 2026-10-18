"""Qwen loading, conversations, and response generation."""

from pathlib import Path
import importlib.metadata
import time

from .config import TYPES
from .utils import atomic_json, file_digest, read_json


def load_qwen(root, adapter=None, quantized=True, max_pixels=512 * 512):
    """Load Qwen, its processor, an optional adapter, and the inference protocol."""
    import torch
    import importlib.metadata
    from huggingface_hub import model_info
    from transformers import AutoProcessor, BitsAndBytesConfig, Qwen3VLForConditionalGeneration
    from peft import PeftModel
    root = Path(root)
    project = read_json(root / "project.json")

    if not torch.cuda.is_available():
        raise RuntimeError("Enable a CUDA GPU runtime")

    model_id = project["model_id"]

    # Resolve once per project, then reuse the exact revision for all notebooks.
    if "model_revision" not in project:
        project["model_revision"] = model_info(model_id).sha
        atomic_json(root / "project.json", project)

    dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16

    kwargs = {
        "revision": project["model_revision"],
        "dtype": dtype,
        "device_map": {"": 0},
        "attn_implementation": "sdpa",
    }

    if quantized:
        kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=dtype,
        )

    model = Qwen3VLForConditionalGeneration.from_pretrained(
        model_id,
        **kwargs,
    )
    processor = AutoProcessor.from_pretrained(
        model_id,
        revision=project["model_revision"],
        min_pixels=32 * 32 * 32,
        max_pixels=max_pixels,
    )

    if adapter:
        cfg = read_json(Path(adapter) / "run_config.json")

        if cfg["model_revision"] != project["model_revision"]:
            raise ValueError("Adapter model revision mismatch")

        if cfg["protocol"]["max_pixels"] != max_pixels:
            raise ValueError(
                "Use the adapter training pixel budget or explicitly "
                "implement a resolution ablation"
            )

        processor = AutoProcessor.from_pretrained(adapter)
        model = PeftModel.from_pretrained(model, adapter)

    model.eval()

    protocol = {
        "model_id": model_id,
        "model_revision": project["model_revision"],
        "quantized": quantized,
        "dtype": str(dtype),
        "max_pixels": max_pixels,
        "adapter_sha256": (
            file_digest(Path(adapter) / "adapter_model.safetensors") if adapter else None
        ),
        "packages": {
            x: importlib.metadata.version(x)
            for x in (
                "torch",
                "transformers",
                "peft",
                "bitsandbytes",
                "pillow",
            )
        },
    }

    return model, processor, protocol


def conversation(crop, prompt, answer=None):
    """Build a multimodal conversation with an optional assistant answer."""
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": crop},
                {"type": "text", "text": prompt},
            ],
        }
    ]

    if answer is not None:
        messages.append(
            {
                "role": "assistant",
                "content": [
                    {"type": "text", "text": answer},
                ],
            }
        )

    return messages


def generate(model, processor, crop, prompt, max_tokens=512):
    """Generate a deterministic response and record timing and token usage."""
    import torch
    batch = processor.apply_chat_template(
        conversation(crop, prompt),
        tokenize=True,
        add_generation_prompt=True,
        return_dict=True,
        return_tensors="pt",
    )
    batch.pop("token_type_ids", None)
    batch = batch.to(model.device)

    torch.cuda.synchronize()
    start = time.perf_counter()

    with torch.inference_mode():
        output = model.generate(
            **batch,
            max_new_tokens=max_tokens,
            do_sample=False,
            use_cache=True,
        )

    torch.cuda.synchronize()

    ids = output[0, batch["input_ids"].shape[1] :]
    prediction = processor.decode(
        ids,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    ).strip()

    return {
        "prediction": prediction,
        "seconds": time.perf_counter() - start,
        "generated_tokens": ids.numel(),
        "hit_token_limit": ids.numel() >= max_tokens,
    }


def parse_type(text):
    """Normalize a predicted label and return None if it is invalid."""
    label = text.strip().lower()
    return label if label in TYPES else None

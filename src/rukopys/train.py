"""Multitask examples and resumable LoRA training."""

from pathlib import Path
import random

from .config import CLASS_PROMPT, PROMPTS
from .models import conversation
from .utils import atomic_json, digest, file_digest, read_json


def training_examples(rows):
    """Create classification examples and transcription examples for scorable regions."""
    examples = []

    for row in rows:
        examples.append(
            {
                **row,
                "task": "classification",
                "prompt": CLASS_PROMPT,
                "answer": row["type"],
            }
        )

        if row["scorable"]:
            examples.append(
                {
                    **row,
                    "task": "transcription",
                    "prompt": PROMPTS[row["type"]],
                    "answer": row["reference"],
                }
            )

    return examples


def supervised_batch(processor, example, device):
    """Prepare one training batch with loss restricted to assistant answer tokens."""
    import torch
    from PIL import Image
    with Image.open(example["crop_path"]) as im:
        crop = im.convert("RGB")

    prefix = processor.apply_chat_template(
        conversation(crop, example["prompt"]),
        tokenize=True,
        add_generation_prompt=True,
        return_dict=True,
        return_tensors="pt",
    )
    full = processor.apply_chat_template(
        conversation(
            crop,
            example["prompt"],
            example["answer"],
        ),
        tokenize=True,
        add_generation_prompt=False,
        return_dict=True,
        return_tensors="pt",
    )

    n = prefix["input_ids"].shape[1]

    if not torch.equal(
        prefix["input_ids"],
        full["input_ids"][:, :n],
    ):
        raise ValueError("Chat-template prefix mismatch")

    labels = full["input_ids"].clone()
    labels[:, :n] = -100
    labels[full["attention_mask"] == 0] = -100

    if not (labels != -100).any():
        raise ValueError("No supervised answer tokens")

    full["labels"] = labels
    full.pop("token_type_ids", None)

    return full.to(device)


def train(
    root,
    model,
    processor,
    protocol,
    run_name="gold_multitask",
    epochs=1,
    lr=1e-4,
    rank=8,
    accumulation=4,
    resume=None,
    save_every=50,
    max_regions=None,
):
    """Train a LoRA adapter with gradient accumulation and resumable checkpoints."""
    import torch
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training, set_peft_model_state_dict
    from safetensors.torch import load_file
    from transformers import set_seed
    if epochs < 1 or accumulation < 1 or save_every < 1:
        raise ValueError("Invalid training settings")

    root = Path(root)
    rows = read_json(root / "data/train.json")
    val = read_json(root / "data/validation.json")

    if {r["page_id"] for r in rows} & {r["page_id"] for r in val}:
        raise ValueError("Train/validation page overlap")

    if max_regions:
        rows = sorted(
            rows,
            key=lambda r: digest([42, r["id"]]),
        )[:max_regions]

    for row in rows:
        if file_digest(row["crop_path"]) != row["crop_sha256"]:
            raise ValueError("Training crop bytes changed")

    examples = training_examples(rows)

    if not examples:
        raise ValueError("No training examples")

    seed = read_json(root / "project.json")["seed"]
    set_seed(seed)

    config = {
        "protocol": protocol,
        "model_revision": protocol["model_revision"],
        "examples_sha256": digest(examples),
        "seed": seed,
        "epochs": epochs,
        "lr": lr,
        "rank": rank,
        "alpha": rank * 2,
        "dropout": 0.05,
        "accumulation": accumulation,
        "prompts": PROMPTS,
        "class_prompt": CLASS_PROMPT,
        "task_weighting": (
            "one classification example per region and "
            "one transcription example per scorable region"
        ),
    }

    run = root / "runs" / run_name

    if (run / "config.json").exists() and not resume:
        raise ValueError("Run exists: explicitly resume or choose a new run_name")

    run.mkdir(parents=True, exist_ok=True)

    if protocol["adapter_sha256"]:
        raise ValueError("Training must start from the base model")

    if protocol["quantized"]:
        model = prepare_model_for_kbit_training(
            model,
            use_gradient_checkpointing=True,
            gradient_checkpointing_kwargs={
                "use_reentrant": False,
            },
        )
    else:
        model.requires_grad_(False)
        model.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs={
                "use_reentrant": False,
            }
        )
        model.enable_input_require_grads()

    targets = [
        n
        for n, _ in model.named_modules()
        if "language_model" in n
        and n.rsplit(".", 1)[-1]
        in (
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
        )
    ]

    if not targets:
        raise ValueError("No language attention modules found")

    model = get_peft_model(
        model,
        LoraConfig(
            r=rank,
            lora_alpha=rank * 2,
            lora_dropout=0.05,
            target_modules=targets,
            bias="none",
            task_type="CAUSAL_LM",
        ),
    )

    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(
        params,
        lr=lr,
        weight_decay=0,
    )

    fp16 = protocol["dtype"] == "torch.float16"
    dtype = torch.float16 if fp16 else torch.bfloat16

    scaler = torch.amp.GradScaler(
        "cuda",
        enabled=fp16,
        init_scale=1024,
    )

    orders = []

    for epoch in range(epochs):
        order = list(range(len(examples)))
        random.Random(seed + epoch).shuffle(order)
        orders.append(order)

    epoch = position = updates = 0
    history = []

    if resume:
        checkpoint = Path(resume)

        if not (checkpoint / "COMPLETE").exists():
            raise ValueError("Incomplete checkpoint")

        if read_json(checkpoint / "run_config.json") != config:
            raise ValueError("Resume configuration mismatch")

        set_peft_model_state_dict(
            model,
            load_file(str(checkpoint / "adapter_model.safetensors")),
        )

        state = torch.load(
            checkpoint / "training_state.pt",
            map_location="cpu",
            weights_only=False,
        )

        optimizer.load_state_dict(state["optimizer"])
        scaler.load_state_dict(state["scaler"])

        epoch = state["epoch"]
        position = state["position"]
        updates = state["updates"]
        history = state["history"]

        random.setstate(state["python_rng"])
        torch.set_rng_state(state["torch_rng"])
        torch.cuda.set_rng_state_all(state["cuda_rng"])

    atomic_json(run / "config.json", config)

    def save():
        """Save the adapter, processor, optimizer, and RNG states as a checkpoint."""
        dest = run / f"checkpoint_{updates:06d}"
        temp = run / f".checkpoint_{updates:06d}.tmp"

        if dest.exists():
            return dest

        temp.mkdir(exist_ok=True)

        model.save_pretrained(temp)
        processor.save_pretrained(temp)
        atomic_json(temp / "run_config.json", config)

        torch.save(
            {
                "optimizer": optimizer.state_dict(),
                "scaler": scaler.state_dict(),
                "epoch": epoch,
                "position": position,
                "updates": updates,
                "history": history,
                "python_rng": random.getstate(),
                "torch_rng": torch.get_rng_state(),
                "cuda_rng": torch.cuda.get_rng_state_all(),
            },
            temp / "training_state.pt",
        )

        (temp / "COMPLETE").write_text("OK")
        temp.rename(dest)

        atomic_json(
            run / "latest.json",
            {"checkpoint": str(dest)},
        )

        return dest

    model.train()
    model.config.use_cache = False

    while epoch < epochs:
        order = orders[epoch]

        while position < len(order):
            indices = order[position : position + accumulation]

            for attempt in range(4):
                optimizer.zero_grad(set_to_none=True)
                losses = []

                for index in indices:
                    batch = supervised_batch(
                        processor,
                        examples[index],
                        model.device,
                    )

                    with torch.autocast("cuda", dtype=dtype):
                        out = model(**batch, use_cache=False)
                        loss = out.loss

                    if not torch.isfinite(loss):
                        raise RuntimeError("Non-finite forward loss; " "progress has not advanced")

                    scaler.scale(loss / len(indices)).backward()

                    losses.append(loss.item())
                    del batch, out, loss

                scaler.unscale_(optimizer)

                finite = all(
                    torch.isfinite(p.grad).all().item() for p in params if p.grad is not None
                )

                if not finite:
                    if not fp16:
                        raise RuntimeError("Non-finite BF16 gradients")

                    scaler.step(optimizer)
                    scaler.update()
                    continue

                torch.nn.utils.clip_grad_norm_(
                    params,
                    1.0,
                    error_if_nonfinite=True,
                )

                scaler.step(optimizer)
                scaler.update()
                break

            else:
                raise RuntimeError("Repeated gradient overflow; " "resume from latest checkpoint")

            optimizer.zero_grad(set_to_none=True)
            position += len(indices)
            updates += 1

            history.append(
                {
                    "epoch": epoch + 1,
                    "position": position,
                    "update": updates,
                    "loss": sum(losses) / len(losses),
                }
            )

            if updates % 10 == 0:
                print(history[-1])

            if updates % save_every == 0:
                save()

        epoch += 1
        position = 0
        save()

    model.eval()

    return model, str(save())

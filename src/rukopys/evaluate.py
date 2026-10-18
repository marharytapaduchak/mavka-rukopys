"""Validation of classification and transcription."""

from PIL import Image
from pathlib import Path

from .config import CLASS_PROMPT, PROMPTS, TYPES
from .metrics import cer_summary
from .models import generate, parse_type
from .utils import atomic_json, digest, file_digest, read_json


def evaluate(
    root,
    model,
    processor,
    protocol,
    output,
    prompts=None,
    max_tokens=512,
):
    """Evaluate classification and oracle/routed transcription on validation crops."""
    rows = read_json(Path(root) / "data/validation.json")
    prompts = prompts or PROMPTS

    config = {
        "manifest": digest(rows),
        "protocol": protocol,
        "prompts": prompts,
        "class_prompt": CLASS_PROMPT,
        "max_tokens": max_tokens,
    }

    saved = read_json(output) if Path(output).exists() else {"config": config, "results": []}

    if saved["config"] != config:
        raise ValueError("Existing evaluation has a different configuration")

    results = saved["results"]

    if [r["id"] for r in results] != [r["id"] for r in rows[: len(results)]]:
        raise ValueError("Resume prefix differs")

    model.eval()

    for row in rows[len(results) :]:
        if file_digest(row["crop_path"]) != row["crop_sha256"]:
            raise ValueError("Crop bytes changed")

        with Image.open(row["crop_path"]) as im:
            crop = im.convert("RGB")

        classification = generate(
            model,
            processor,
            crop,
            CLASS_PROMPT,
            32,
        )
        predicted_type = parse_type(classification["prediction"])

        # Oracle-type branch isolates transcription;
        # predicted-type branch measures routing errors.
        oracle = (
            generate(
                model,
                processor,
                crop,
                prompts[row["type"]],
                max_tokens,
            )
            if row["scorable"]
            else {
                "prediction": "",
                "seconds": 0,
                "hit_token_limit": False,
            }
        )

        if predicted_type in ("image", "graph"):
            routed = {
                "prediction": "",
                "seconds": 0,
                "hit_token_limit": False,
            }
        elif predicted_type in prompts:
            routed = (
                oracle
                if predicted_type == row["type"] and row["scorable"]
                else generate(
                    model,
                    processor,
                    crop,
                    prompts[predicted_type],
                    max_tokens,
                )
            )
        else:
            routed = {
                "prediction": "",
                "seconds": 0,
                "hit_token_limit": False,
            }

        results.append(
            {
                **row,
                "prediction": oracle["prediction"],
                "oracle": oracle,
                "routed": routed,
                "classification": classification,
                "predicted_type": predicted_type,
            }
        )

        atomic_json(
            output,
            {
                "config": config,
                "complete": len(results) == len(rows),
                "results": results,
            },
        )

        print(
            f"{len(results)}/{len(rows)} | " f"{row['source']} | {row['type']} -> {predicted_type}"
        )

    summary = {
        "oracle": cer_summary([r for r in results if r["scorable"]]),
        "routed": cer_summary(
            [
                {
                    **r,
                    "prediction": r["routed"]["prediction"],
                }
                for r in results
                if r["scorable"]
            ]
        ),
        "classification_accuracy": (
            sum(r["type"] == r["predicted_type"] for r in results) / len(results)
            if results
            else None
        ),
        "invalid_labels": sum(r["predicted_type"] is None for r in results),
        "by_type": {
            t: cer_summary([r for r in results if r["type"] == t and r["scorable"]]) for t in TYPES
        },
        "by_source": {
            s: cer_summary([r for r in results if r["source"] == s and r["scorable"]])
            for s in sorted({r["source"] for r in results})
        },
    }

    atomic_json(
        output,
        {
            "config": config,
            "complete": True,
            "results": results,
            "summary": summary,
        },
    )

    return summary

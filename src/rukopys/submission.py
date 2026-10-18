"""Fixed-policy inference; no test labels or manual test-specific repairs."""

from pathlib import Path

from PIL import Image

from .config import TYPES
from .models import generate, parse_type
from .utils import atomic_json, digest, file_digest, read_json, valid_bbox


def transcribe_frozen(pages, model, processor, protocol, output, policy):
    """Transcribe detected regions using a fixed policy and resumable page outputs."""
    names = [page["image"] for page in pages]

    if not names or len(names) != len(set(names)):
        raise ValueError("Empty or duplicate page list")

    if policy["unresolved_text"] != "empty" or policy["invalid_type"] != "stop":
        raise ValueError("Unsupported export policy")

    config = {
        "schema": "frozen_submission_v1",
        "detections_sha256": digest(pages),
        "image_sha256": {page["image"]: file_digest(page["image_path"]) for page in pages},
        "protocol": protocol,
        "policy": policy,
    }

    output = Path(output)
    saved = read_json(output) if output.exists() else {"config": config, "pages": []}

    if saved["config"] != config:
        raise ValueError("Frozen inference configuration mismatch; retain existing output")

    done = saved["pages"]

    if [page["image"] for page in done] != names[: len(done)]:
        raise ValueError("Resume order mismatch")

    if len(done) > len(pages):
        raise ValueError("Too many saved pages")

    model.eval()

    for page in pages[len(done) :]:
        regions = []
        diagnostics = []

        with Image.open(page["image_path"]) as image:
            image = image.convert("RGB")

            for index, region in enumerate(page["regions"]):
                bbox = valid_bbox(region["bbox"], *image.size)

                with image.crop(tuple(bbox)) as crop:
                    classification = generate(
                        model,
                        processor,
                        crop,
                        policy["class_prompt"],
                        32,
                    )
                    kind = parse_type(classification["prediction"])

                    if kind not in TYPES or classification["hit_token_limit"]:
                        raise ValueError(
                            "Invalid classification; " "inference stopped without guessing"
                        )

                    prediction = (
                        generate(
                            model,
                            processor,
                            crop,
                            policy["prompts"][kind],
                            policy["max_tokens"],
                        )
                        if kind in policy["prompts"]
                        else {
                            "prediction": "",
                            "hit_token_limit": False,
                            "seconds": 0,
                        }
                    )

                abstain = bool(prediction["hit_token_limit"])

                regions.append(
                    {
                        "bbox": bbox,
                        "type": kind,
                        "text": "" if abstain else prediction["prediction"],
                    }
                )
                diagnostics.append(
                    {
                        "region_index": index,
                        "classification": classification,
                        "transcription": prediction,
                        "abstained": abstain,
                    }
                )

        done.append(
            {
                "image": page["image"],
                "regions": regions,
                "diagnostics": diagnostics,
            }
        )

        atomic_json(
            output,
            {
                "config": config,
                "complete": len(done) == len(pages),
                "pages": done,
            },
        )

        print(f"{len(done)}/{len(pages)} pages saved", flush=True)

    atomic_json(
        output,
        {
            "config": config,
            "complete": True,
            "pages": done,
        },
    )

    return done

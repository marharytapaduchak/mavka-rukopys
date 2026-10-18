"""Page detection, transcription, and CSV export."""

from PIL import Image
from doclayout_yolo import YOLOv10
from pathlib import Path
import csv
import json

from .config import CLASS_PROMPT, PROMPTS, TYPES
from .models import generate, parse_type
from .utils import atomic_json, digest, file_digest, read_json, valid_bbox


def detect_pages(
    pages,
    weights,
    output,
    confidence=0.25,
    image_size=1024,
):
    """Detect document regions with YOLO and save resumable page-level results."""
    config = {
        "weights_sha256": file_digest(weights),
        "pages_sha256": digest(pages),
        "confidence": confidence,
        "image_size": image_size,
    }

    saved = read_json(output) if Path(output).exists() else {"config": config, "pages": []}

    if saved["config"] != config:
        raise ValueError("Detection configuration mismatch")

    done = saved["pages"]

    if [p["image"] for p in done] != [p["image"] for p in pages[: len(done)]]:
        raise ValueError("Detection resume order mismatch")

    model = YOLOv10(str(weights))

    for page in pages[len(done) :]:
        result = model.predict(
            page["image_path"],
            conf=confidence,
            imgsz=image_size,
            device="cuda:0",
            verbose=False,
        )[0]

        h, w = result.orig_shape

        if (w, h) != (page["width"], page["height"]):
            raise ValueError("Unexpected original image dimensions")

        regions = []

        for box, score, cls in zip(
            result.boxes.xyxy.cpu().tolist(),
            result.boxes.conf.cpu().tolist(),
            result.boxes.cls.cpu().tolist(),
        ):
            box = [
                max(0, min(w, box[0])),
                max(0, min(h, box[1])),
                max(0, min(w, box[2])),
                max(0, min(h, box[3])),
            ]

            valid_bbox(box, w, h)
            label = result.names[int(cls)]

            regions.append(
                {
                    "bbox": box,
                    "confidence": score,
                    "type": label if label in TYPES else None,
                }
            )

        done.append(
            {
                **page,
                "regions": regions,
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

    return done


def transcribe_pages(
    pages,
    model,
    processor,
    protocol,
    output,
    max_tokens=1024,
):
    """Classify and transcribe detected regions with resumable page-level output."""
    if len({p["image"] for p in pages}) != len(pages):
        raise ValueError("Duplicate filenames")

    config = {
        "detections_sha256": digest(pages),
        "image_sha256": {p["image"]: file_digest(p["image_path"]) for p in pages},
        "protocol": protocol,
        "prompts": PROMPTS,
        "class_prompt": CLASS_PROMPT,
        "max_tokens": max_tokens,
    }

    saved = read_json(output) if Path(output).exists() else {"config": config, "pages": []}

    if saved["config"] != config:
        raise ValueError("Inference resume configuration mismatch")

    done = saved["pages"]

    if [p["image"] for p in done] != [p["image"] for p in pages[: len(done)]]:
        raise ValueError("Inference resume order mismatch")

    model.eval()

    for page in pages[len(done) :]:
        with Image.open(page["image_path"]) as im:
            image = im.convert("RGB")

        regions = []
        diagnostics = []

        for index, region in enumerate(page["regions"]):
            bbox = valid_bbox(region["bbox"], *image.size)
            crop = image.crop(tuple(bbox))

            # Always classify with Qwen so behavior matches the validation protocol.
            cls = generate(
                model,
                processor,
                crop,
                CLASS_PROMPT,
                32,
            )
            kind = parse_type(cls["prediction"])

            if kind is None:
                raise ValueError(
                    f"Invalid classification on {page['image']} "
                    f"region {index}: {cls['prediction']!r}; "
                    "no guessed label exported"
                )

            pred = (
                generate(
                    model,
                    processor,
                    crop,
                    PROMPTS[kind],
                    max_tokens,
                )
                if kind in PROMPTS
                else {
                    "prediction": "",
                    "hit_token_limit": False,
                    "seconds": 0,
                }
            )

            regions.append(
                {
                    "bbox": bbox,
                    "type": kind,
                    "text": pred["prediction"],
                }
            )
            diagnostics.append(
                {
                    "region_index": index,
                    "classification": cls,
                    "transcription": pred,
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

        print(f"{len(done)}/{len(pages)} | " f"{page['image']} | {len(regions)} regions")

    return done


def write_submission(
    pages,
    expected_images,
    path,
    allow_truncated=False,
):
    """Validate page predictions and export them in the submission CSV format."""
    names = [p["image"] for p in pages]

    if len(set(names)) != len(names) or set(names) != set(expected_images):
        raise ValueError("Expected exactly one record for every image")

    for page in pages:
        if not allow_truncated and any(
            d["transcription"]["hit_token_limit"] for d in page.get("diagnostics", [])
        ):
            raise ValueError("Truncated generation: increase budget " "and use a new output file")

        for r in page["regions"]:
            if r["type"] not in TYPES or not isinstance(r["text"], str):
                raise ValueError("Invalid region")

            if r["type"] in ("image", "graph") and r["text"]:
                raise ValueError("Non-text region must have empty text")

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    by_name = {p["image"]: p for p in pages}

    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["image", "regions"],
        )
        writer.writeheader()

        for name in expected_images:
            writer.writerow(
                {
                    "image": name,
                    "regions": json.dumps(
                        by_name[name]["regions"],
                        ensure_ascii=False,
                    ),
                }
            )

    return str(path)

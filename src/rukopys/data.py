"""Dataset splits and resumable crop preparation."""

from pathlib import Path
import json
import math
import time

from .config import MODEL_ID, TYPES
from .utils import atomic_json, digest, file_digest, read_json, valid_bbox


def make_split(records, revision, seed=42, fraction=0.2):
    """Create a deterministic, source-stratified split with disjoint pages."""
    ids = [r["file_name"] for r in records]

    if len(set(ids)) != len(ids):
        raise ValueError("Duplicate page IDs")

    train, validation = [], []

    for source in sorted({r["source"] for r in records}):
        group = sorted(
            [r for r in records if r["source"] == source],
            key=lambda r: digest([seed, r["file_name"]]),
        )

        n = (
            min(
                len(group) - 1,
                max(1, math.ceil(len(group) * fraction)),
            )
            if len(group) > 1
            else 0
        )

        validation.extend(r["file_name"] for r in group[:n])
        train.extend(r["file_name"] for r in group[n:])

    return {
        "dataset_revision": revision,
        "seed": seed,
        "train": sorted(train),
        "validation": sorted(validation),
        "limitations": [
            "Page-disjoint; writer independence and dictation text "
            "independence are not guaranteed."
        ],
    }


def prepare_data(root, seed=42, existing_split=None, revision=None):
    """Prepare validated crops and manifests with resumable per-page receipts."""
    from huggingface_hub import dataset_info, hf_hub_download
    from PIL import Image, ImageChops
    root = Path(root)
    data = root / "data"
    data.mkdir(parents=True, exist_ok=True)

    repo = "UkrainianCatholicUniversity/rukopys"
    state_path = data / "preparation_state.json"

    state = read_json(state_path) if state_path.exists() else {}
    previous_project = read_json(root / "project.json") if (root / "project.json").exists() else {}

    started = time.perf_counter()

    def download(filename, selected_revision):
        """Retrieve a dataset file from cache or download it with retries."""
        try:
            return hf_hub_download(
                repo,
                filename,
                repo_type="dataset",
                revision=selected_revision,
                local_files_only=True,
            )
        except Exception:
            print(f"  Downloading: {filename}", flush=True)

        for attempt in range(1, 4):
            try:
                return hf_hub_download(
                    repo,
                    filename,
                    repo_type="dataset",
                    revision=selected_revision,
                    etag_timeout=15,
                )
            except Exception as exc:
                print(
                    f"  Download attempt {attempt}/3 failed: " f"{type(exc).__name__}: {exc}",
                    flush=True,
                )

                if attempt == 3:
                    raise

        raise RuntimeError("Unreachable")

    revision = revision or state.get("dataset_revision") or previous_project.get("dataset_revision")

    print("Preparing metadata (cached files are reused).", flush=True)

    if not revision:
        # Recover the revision of the already cached main metadata from the old run.
        try:
            cached = hf_hub_download(
                repo,
                "train/metadata.jsonl",
                repo_type="dataset",
                local_files_only=True,
            )

            parts = Path(cached).parts
            revision = parts[parts.index("snapshots") + 1]

            print(
                f"Recovered cached dataset revision: {revision}",
                flush=True,
            )
        except Exception:
            revision = dataset_info(repo).sha

    metadata = download("train/metadata.jsonl", revision)

    records = [
        json.loads(line)
        for line in Path(metadata).read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]

    split_path = Path(existing_split) if existing_split else data / "split.json"
    split = read_json(split_path) if split_path.exists() else make_split(records, revision, seed)

    train = set(split["train"])
    val = set(split["validation"])
    all_ids = {r["file_name"] for r in records}

    if train & val or train | val != all_ids:
        raise ValueError("Split must partition all metadata pages exactly")

    if split.get("dataset_revision", revision) != revision:
        raise ValueError("Split revision mismatch")

    config = {
        "schema": 2,
        "dataset_revision": revision,
        "metadata_sha256": file_digest(metadata),
        "split_sha256": digest(split),
        "root": str(root.absolute()),
    }

    if state and state.get("config") != config:
        raise ValueError(
            "Preparation settings changed. "
            "Keep the same root, dataset revision and split; "
            "do not overwrite existing progress."
        )

    atomic_json(data / "split.json", split)
    atomic_json(
        state_path,
        {
            "dataset_revision": revision,
            "config": config,
            "complete": False,
            "processed_pages": 0,
            "total_pages": len(records),
        },
    )

    receipts = data / "preparation_pages"
    receipts.mkdir(exist_ok=True)

    outputs = {"train": [], "validation": []}
    rejected = []
    pages = []

    created = reused = resumed = 0

    print(
        f"Total pages: {len(records)}; "
        f"annotated regions: {sum(len(p.get('regions', [])) for p in records)}",
        flush=True,
    )
    print(
        "Existing PNGs are checked against their source crop before reuse. "
        "Completed-page receipts allow later resumes without downloading that page.",
        flush=True,
    )

    for page_number, page in enumerate(records, 1):
        partition = "train" if page["file_name"] in train else "validation"

        receipt = receipts / (digest(page["file_name"]) + ".json")
        page_signature = digest(
            {
                "config": config,
                "page": page,
                "partition": partition,
            }
        )

        cached_page = read_json(receipt) if receipt.exists() else None

        if (
            cached_page
            and cached_page.get("signature") == page_signature
            and Path(cached_page["page"]["image_path"]).exists()
            and all(Path(r["crop_path"]).exists() for r in cached_page["rows"])
        ):
            # Metadata comes from an atomically committed page receipt.
            outputs[partition].extend(cached_page["rows"])
            rejected.extend(cached_page["rejected"])
            pages.append(cached_page["page"])
            resumed += 1

        else:
            print(
                f"[{page_number}/{len(records)}] "
                f"{page['file_name']} | {partition} | "
                f"regions={len(page.get('regions', []))}",
                flush=True,
            )

            image_path = download("train/" + page["file_name"], revision)
            page_rows = []
            page_rejected = []

            with Image.open(image_path) as im:
                image = im.convert("RGB")
                width, height = image.size

                page_info = {
                    "image": page["file_name"],
                    "image_path": str(image_path),
                    "split": partition,
                    "source": page["source"],
                    "width": width,
                    "height": height,
                }

                for index, region in enumerate(page.get("regions", [])):
                    try:
                        kind = region["type"]

                        if kind not in TYPES:
                            raise ValueError("Unknown type")

                        bbox = valid_bbox(region["bbox"], width, height)
                        text = region.get("text") or ""

                        if not isinstance(text, str):
                            raise ValueError("Non-string text")

                    except (ValueError, KeyError) as exc:
                        page_rejected.append(
                            {
                                "page": page["file_name"],
                                "region": index,
                                "reason": str(exc),
                            }
                        )
                        continue

                    uid = digest([page["file_name"], index])[:24]
                    crop_path = data / "crops" / partition / (uid + ".png")
                    crop_path.parent.mkdir(parents=True, exist_ok=True)

                    expected = image.crop(tuple(bbox))
                    matches = False

                    if crop_path.exists():
                        try:
                            with Image.open(crop_path) as old:
                                old.load()
                                matches = (
                                    old.mode == "RGB"
                                    and old.size == expected.size
                                    and ImageChops.difference(old, expected).getbbox() is None
                                )
                        except (OSError, ValueError):
                            matches = False

                        # Never overwrite an old file whose content cannot be confirmed.
                        if not matches:
                            raise ValueError(
                                "Existing crop differs from source or is damaged: "
                                f"{crop_path}. It has been preserved. "
                                "Move this specific file aside, then retry."
                            )

                    if matches:
                        reused += 1
                    else:
                        temporary = crop_path.with_suffix(".png.tmp")
                        expected.save(
                            temporary,
                            format="PNG",
                            compress_level=1,
                        )
                        temporary.replace(crop_path)
                        created += 1

                    scorable = (
                        kind not in ("image", "graph")
                        and region.get("language") == "uk"
                        and region.get("legibility") == "legible"
                        and bool(text.strip())
                    )

                    page_rows.append(
                        {
                            "id": uid,
                            "page_id": page["file_name"],
                            "region_index": index,
                            "source": page["source"],
                            "bbox": bbox,
                            "type": kind,
                            "reference": text,
                            "scorable": scorable,
                            "crop_path": str(crop_path),
                            "crop_sha256": file_digest(crop_path),
                        }
                    )

                    expected.close()

                image.close()

            atomic_json(
                receipt,
                {
                    "signature": page_signature,
                    "page": page_info,
                    "rows": page_rows,
                    "rejected": page_rejected,
                },
            )

            outputs[partition].extend(page_rows)
            rejected.extend(page_rejected)
            pages.append(page_info)

        region_count = sum(len(rows) for rows in outputs.values())

        atomic_json(
            state_path,
            {
                "dataset_revision": revision,
                "config": config,
                "complete": False,
                "processed_pages": page_number,
                "total_pages": len(records),
                "regions": region_count,
                "last_page": page["file_name"],
            },
        )

        if page_number % 10 == 0 or page_number == len(records):
            elapsed = (time.perf_counter() - started) / 60

            print(
                f"Progress: {page_number}/{len(records)} pages | "
                f"{region_count} valid crops | "
                f"reused PNG={reused}, new PNG={created}, "
                f"resumed pages={resumed} | {elapsed:.1f} min",
                flush=True,
            )

    for name, rows in outputs.items():
        atomic_json(data / (name + ".json"), rows)

    atomic_json(data / "pages.json", pages)
    atomic_json(data / "rejected.json", rejected)

    project = {
        **previous_project,
        "dataset_id": repo,
        "dataset_revision": revision,
        "model_id": MODEL_ID,
        "seed": split.get("seed", seed),
        "train_manifest_sha256": digest(outputs["train"]),
        "validation_manifest_sha256": digest(outputs["validation"]),
    }

    atomic_json(root / "project.json", project)
    atomic_json(
        state_path,
        {
            "dataset_revision": revision,
            "config": config,
            "complete": True,
            "processed_pages": len(records),
            "total_pages": len(records),
            "regions": sum(len(rows) for rows in outputs.values()),
        },
    )

    result = {name: len(rows) for name, rows in outputs.items()} | {
        "rejected": len(rejected),
        "reused_png": reused,
        "new_png": created,
        "resumed_pages": resumed,
    }

    print("Preparation completed; final manifests saved.", flush=True)

    return result

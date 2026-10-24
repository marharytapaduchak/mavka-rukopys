#!/usr/bin/env python
"""Download RUKOPYS training metadata from Hugging Face."""

import argparse
import hashlib
from pathlib import Path

from huggingface_hub import hf_hub_download

DATASET_ID = "UkrainianCatholicUniversity/rukopys"
METADATA_FILE = "train/metadata.jsonl"

DEFAULT_OUTPUT = Path("data/raw/metadata.jsonl")


def download_metadata(output_path, revision=None):
    """Download metadata without downloading document images."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if output_path.exists():
        print(f"Metadata already exists: {output_path}")
        return output_path

    print(f"Downloading metadata from {DATASET_ID}")

    cached_path = hf_hub_download(
        repo_id=DATASET_ID,
        filename=METADATA_FILE,
        repo_type="dataset",
        revision=revision,
    )

    data = Path(cached_path).read_bytes()
    output_path.write_bytes(data)

    checksum = hashlib.sha256(data).hexdigest()

    print(f"Saved metadata to: {output_path}")
    print(f"SHA256: {checksum}")
    print(f"Size: {len(data) / (1024 * 1024):.2f} MB")

    return output_path


def main():
    parser = argparse.ArgumentParser(description="Download RUKOPYS training metadata.")

    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help="Local metadata destination.",
    )

    parser.add_argument(
        "--revision",
        type=str,
        default=None,
        help="Optional Hugging Face dataset revision.",
    )

    args = parser.parse_args()

    download_metadata(output_path=args.output, revision=args.revision)


if __name__ == "__main__":
    main()


# RUKOPYS — Utility Scripts

This directory contains utility scripts for the
MAVKA — RUKOPYS document understanding project.

## Available Scripts

| Script | Purpose |
|---|---|
| `download_data.py` | Download RUKOPYS training metadata from Hugging Face without downloading document images. |

## Downloading Dataset Metadata

The `download_data.py` script retrieves the annotation
metadata from the [RUKOPYS dataset](https://huggingface.co/datasets/UkrainianCatholicUniversity/rukopys).

**Dataset:** `UkrainianCatholicUniversity/rukopys`

**Source file:** `train/metadata.jsonl`

**Default destination:** `data/raw/metadata.jsonl`

The metadata contains document information and region-level
annotations used for data exploration and preparation.

### Usage

Run the script from the repository root:

```bash
python scripts/download_data.py
```

By default, the downloaded metadata is saved to:

```text
data/raw/metadata.jsonl
```

### Optional Arguments

Specify a custom output path:

```bash
python scripts/download_data.py --output data/raw/custom_metadata.jsonl
```

Specify a Hugging Face dataset revision:

```bash
python scripts/download_data.py --revision <revision>
```

Using a fixed revision helps ensure reproducibility
across experiments.

## Script Behavior

The script:

1. Downloads only the training metadata.
2. Creates the output directory if necessary.
3. Skips downloading if the destination file already exists.
4. Calculates and displays the SHA-256 checksum of newly downloaded metadata.
5. Reports the file size and saved location.

**Important:** Existing files are not automatically
revalidated against the Hugging Face dataset revision.

If the metadata already exists at the destination,
the script skips downloading and does not recompute its checksum.

## Difference from Data Preparation

The metadata downloader is a lightweight utility and
does not perform the complete preprocessing pipeline.

Full data preparation is implemented in:

- [`src/rukopys/data.py`](../src/rukopys/data.py)
- [`notebooks/01_prepare_data.ipynb`](../notebooks/01_prepare_data.ipynb)

The full pipeline handles:

- Dataset image retrieval.
- Train/validation splitting.
- Bounding box validation.
- Region crop extraction.
- Training and validation manifest generation.

## Exploratory Data Analysis

Dataset exploration is available in:

[`notebooks/07_rukopys_eda.ipynb`](../notebooks/07_rukopys_eda.ipynb)

The EDA notebook can download metadata directly
from Hugging Face and does not require running
`download_data.py` beforehand.

## Reproducibility

For reproducible experiments:

- Use a fixed Hugging Face dataset revision.
- Keep training and validation partitions consistent.
- Record dataset versions and checksums.
- Avoid mixing metadata from different dataset revisions.

Large datasets, generated crops, and model checkpoints
should not be committed directly to the repository.

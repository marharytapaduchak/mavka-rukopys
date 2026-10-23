
# RUKOPYS — Dataset Card

## Name and source

**RUKOPYS** is a dataset of Ukrainian handwritten documents published by the Ukrainian Catholic University.

The dataset is used in the **Handwritten to Data** Kaggle competition, which focuses on converting handwritten document images into structured digital information.

- **Dataset:** [UkrainianCatholicUniversity/rukopys](https://huggingface.co/datasets/UkrainianCatholicUniversity/rukopys)
- **Competition:** [Handwritten to Data](https://www.kaggle.com/competitions/handwritten-to-data/overview)
- **Dataset hosting:** Hugging Face

## Problem and motivation

Recognizing Ukrainian handwriting is challenging because documents vary in writing style, layout, image quality, and content.

Traditional OCR systems may struggle with cursive handwriting, historical spelling, mathematical expressions, and complex document layouts.

Our project aims to transform handwritten documents into structured data through three tasks:

1. **Region detection** — locate meaningful regions using bounding boxes.
2. **Region classification** — identify the content type of each region.
3. **Transcription** — recognize and extract textual content.

## Dataset structure

The dataset contains document images and region-level annotations.

The training metadata is provided in `train/metadata.jsonl`.

Each document record contains:

| Field | Description |
|---|---|
| `file_name` | Document image filename |
| `source` | Source or category of the document |
| `regions` | List of annotated regions |

Each annotated region contains:

| Field | Description |
|---|---|
| `bbox` | Bounding box coordinates |
| `type` | Region content type |
| `text` | Ground-truth transcription, where available |
| `language` | Language annotation, where available |
| `legibility` | Handwriting legibility annotation, where available |

### Region types

The project uses seven region categories:

| Region type | Expected output |
|---|---|
| `handwritten` | Transcribed handwritten text |
| `printed` | Transcribed printed text |
| `annotation` | Transcribed annotations |
| `formula` | Mathematical expressions in LaTeX |
| `table` | Structured rows and cells |
| `image` | Bounding box and type, without transcription |
| `graph` | Bounding box and type, without transcription |

## Data preparation

The preparation pipeline is implemented in:

- [`notebooks/01_prepare_data.ipynb`](../notebooks/01_prepare_data.ipynb)
- [`src/rukopys/data.py`](../src/rukopys/data.py)

The pipeline performs:

1. Downloading metadata and document images from Hugging Face.
2. Creating a reproducible train/validation split.
3. Validating bounding boxes and region annotations.
4. Extracting individual image crops.
5. Generating metadata manifests for model training and evaluation.

### Train/validation split

The dataset is divided into training and validation partitions.

- **Training:** approximately 80%
- **Validation:** approximately 20%
- **Random seed:** 42
- **Strategy:** source-stratified, page-disjoint split

The split prevents the same page from appearing in both partitions.

However, it does not guarantee that the same handwriting author or dictated text is absent from both partitions.

## Data quality and validation

The preparation pipeline checks:

- Duplicate page identifiers.
- Valid region categories.
- Bounding box coordinates.
- Transcription field types.
- Train/validation overlap.
- Consistency of generated image crops.

Invalid annotations are recorded separately for inspection.

Non-text regions such as images and graphs do not require transcription.


## Exploratory Data Analysis

Exploratory Data Analysis (EDA) was conducted to examine the RUKOPYS dataset, identify potential challenges, and support further model development.

The analysis is available in [`07_rukopys_eda.ipynb`](../notebooks/07_rukopys_eda.ipynb).

### Dataset Overview

The labeled training dataset contains:

| Statistic | Value |
|---|---:|
| Document pages | 1,330 |
| Annotated regions | 25,651 |
| Document sources | 4 |
| Region categories | 7 |
| Regions with non-empty transcription | 25,376 |
| Median regions per page | 20 |
| Median transcription length | 30 characters |

### Document Source Distribution

| Source | Pages |
|---|---:|
| School | 682 |
| Dictation | 359 |
| University | 162 |
| Archive | 127 |

School documents represent the largest proportion of the dataset (51.3%), while archival documents account for only 9.5%.

### Region Type Distribution

| Region Type | Count | Percentage |
|---|---:|---:|
| Handwritten | 21,577 | 84.12% |
| Formula | 2,892 | 11.27% |
| Annotation | 554 | 2.16% |
| Printed | 308 | 1.20% |
| Table | 142 | 0.55% |
| Image | 119 | 0.46% |
| Graph | 59 | 0.23% |

The dataset exhibits significant class imbalance, with handwritten regions dominating the annotations. Less frequent classes, particularly tables and graphs, may be more challenging to recognize reliably.

### Data Quality and Splitting

All 25,651 bounding boxes passed coordinate and image-boundary validation.

A deterministic, source-stratified train/validation split was reconstructed using the project's splitting logic:

- **Training:** 1,062 pages
- **Validation:** 268 pages
- **Random seed:** 42
- **Page overlap:** None

The reconstructed split should be checked against the split used in the model training experiments.

### Key Findings and Implications

1. **Class imbalance:** Model performance should be evaluated across individual region types, not only using overall accuracy.
2. **Document diversity:** Different document sources and handwriting styles may affect generalization. Source-specific evaluation is recommended.
3. **Transcription complexity:** Variation in transcription length should be considered when selecting generation parameters.
4. **Annotation quality:** Bounding box validation did not reveal invalid coordinates.
5. **Evaluation:** CER and Page CER are important for assessing transcription quality, alongside detection and classification metrics.

These findings will guide further fine-tuning, error analysis, and evaluation of the document understanding pipeline.

## Known limitations and challenges

Several dataset characteristics may affect model performance:

- **Handwriting variability:** writing styles differ across documents and authors.
- **Historical spelling:** older documents may contain uncommon vocabulary and spelling conventions.
- **Image quality:** some pages may contain noise, blur, or low contrast.
- **Complex layouts:** tables, formulas, and mixed-content pages require specialized handling.
- **Class imbalance:** The dataset is highly imbalanced, with handwritten regions accounting for 84.12% of all annotations. Rare categories such as tables (0.55%) and graphs (0.23%) may require additional attention during evaluation.
- **Generalization:** performance may vary between document sources and handwriting styles.

## Evaluation

The competition evaluates different aspects of document understanding.

| Metric | Purpose |
|---|---|
| Detection F1 | Measures bounding box detection quality |
| Classification Accuracy | Measures region classification correctness |
| Character Error Rate (CER) | Measures transcription errors |
| Page CER | Measures transcription quality at the page level |

Character Error Rate is calculated as:

`CER = (Substitutions + Deletions + Insertions) / Reference characters`

Lower CER indicates more accurate transcription.

The official competition evaluation protocol is used for final scoring.

## Dataset storage and reproducibility

Large dataset files and generated image crops are not stored directly in Git.

Instead, they are retrieved from Hugging Face and processed locally.

The preparation code records the dataset revision and uses deterministic split generation to support reproducibility.

Generated artifacts include:

- `split.json`
- `train.json`
- `validation.json`
- `pages.json`
- `rejected.json`
- Cropped region images

## License and usage

Dataset redistribution and usage must follow the conditions specified on the [RUKOPYS Hugging Face page](https://huggingface.co/datasets/UkrainianCatholicUniversity/rukopys).

## References

- [RUKOPYS Dataset — Hugging Face](https://huggingface.co/datasets/UkrainianCatholicUniversity/rukopys)
- [Handwritten to Data — Kaggle](https://www.kaggle.com/competitions/handwritten-to-data/overview)


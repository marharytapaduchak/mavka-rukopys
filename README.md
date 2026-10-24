# Handwritten to data

Predicts which of 3 grape cultivars a wine came from, using 13 chemical measurements — a small, complete, fully-reproducible example project built from [`ucu-ai-course/project-template`](https://github.com/ucu-ai-course/project-template).

![lint](https://github.com/ucu-ai-course/project-template/actions/workflows/lint.yml/badge.svg)
![license](https://img.shields.io/badge/license-MIT-blue.svg)

> [!note]
> This repo is **both** a GitHub Template (click "Use this template" to get a clean, working starting point for your own semester project) **and** a fully filled-in example (this exact repo trains a real model and produces real numbers — nothing here is a placeholder). See §"Repository structure" for what to replace with your own project's content.

## Demo



## Team

| Name | GitHub | Role |
|---|---|---|
| Daryna Shevchuk | [@dasha-pn](https://github.com/dasha-pn) | - |
| Marharyta Paduchak | [@marharytapaduchak](https://github.com/marharytapaduchak) | - |
| Viktoriya Kibyeryeva | [@vi-kibereva](https://github.com/vi-kibereva) | - |


## Problem and motivation

Ukrainian handwritten documents contain valuable information, but much of it remains difficult to access and process automatically. Unlike printed text, handwriting varies significantly in style, legibility, spelling, and document structure. Historical documents, mathematical expressions, tables, and mixed-content pages make recognition even more challenging.

Traditional OCR systems often struggle with handwritten Ukrainian text and complex page layouts. Recognizing individual words is not enough: a complete document understanding system must also locate meaningful regions, identify their content types, and preserve the original text.

The **MAVKA — RUKOPYS** project addresses this challenge as part of the [Handwritten to Data](https://www.kaggle.com/competitions/handwritten-to-data/) Kaggle competition.

The goal is to transform images of Ukrainian documents into structured digital representations through three tasks:

1. **Region detection:** Identify document regions and predict their bounding boxes.
2. **Region classification:** Classify each region as handwritten text, printed text, annotation, formula, table, image, or graph.
3. **Transcription:** Extract textual content while preserving original spelling, punctuation, and mathematical notation.

The project combines document layout detection with vision-language models to build an end-to-end pipeline for Ukrainian handwritten document understanding.


## Data

The project uses the [RUKOPYS dataset](https://huggingface.co/datasets/UkrainianCatholicUniversity/rukopys), published by the Ukrainian Catholic University and used in the *Handwritten to Data* competition.

The dataset contains document images and region-level annotations, including bounding boxes, content types, and ground-truth transcriptions where available.

### Dataset overview

Exploratory Data Analysis (EDA) of the labeled training dataset identified:

| Statistic | Value |
|---|---:|
| Document pages | 1,330 |
| Annotated regions | 25,651 |
| Document sources | 4 |
| Region categories | 7 |
| Regions with non-empty transcription | 25,376 |
| Median regions per page | 20 |
| Median transcription length | 30 characters |

The dataset includes four document sources:

| Source | Pages |
|---|---:|
| School | 682 |
| Dictation | 359 |
| University | 162 |
| Archive | 127 |

### Region distribution

| Region type | Count | Percentage |
|---|---:|---:|
| Handwritten | 21,577 | 84.12% |
| Formula | 2,892 | 11.27% |
| Annotation | 554 | 2.16% |
| Printed | 308 | 1.20% |
| Table | 142 | 0.55% |
| Image | 119 | 0.46% |
| Graph | 59 | 0.23% |

Handwritten text accounts for **84.12%** of all annotated regions, while categories such as tables and graphs are significantly underrepresented. This class imbalance motivates per-class evaluation rather than relying only on overall accuracy.

### Data preparation and validation

The dataset preparation pipeline supports:

- Loading document images and annotations from Hugging Face.
- Validating bounding boxes and annotation fields.
- Creating reproducible training and validation partitions.
- Extracting region crops for classification and transcription.

EDA confirmed that all **25,651 bounding boxes** passed coordinate and image-boundary validation.

A deterministic, source-stratified, page-disjoint split was reconstructed with random seed 42:

- **Training:** 1,062 pages (approximately 80%).
- **Validation:** 268 pages (approximately 20%).

No pages overlap between the reconstructed partitions. This split still needs to be verified against the actual training artifacts.

### Key challenges

The EDA highlighted several factors relevant to model development:

- **Class imbalance:** Rare region categories may be harder to recognize reliably.
- **Document diversity:** School, dictation, university, and archival documents differ in content and presentation.
- **Handwriting variability:** Writing styles, legibility, and historical spelling complicate transcription.
- **Complex layouts:** Formulas, tables, and mixed-content pages require more than plain-text OCR.

These findings inform model fine-tuning, evaluation, and error analysis.

For the complete dataset description, preparation procedure, and EDA results, see:

- [Dataset documentation](data/README.md)
- [EDA notebook](notebooks/07_rukopys_eda.ipynb)


## Approach



## Results



## Quick start

```bash
git clone <repo-url> && cd wine-origin-classifier
make setup                          # creates .venv, installs deps, installs pre-commit hooks
make reproduce                      # data -> train -> evaluate; should print accuracy=0.9722 macro_f1=0.9710
source .venv/bin/activate           # everything past this point assumes the venv is active
streamlit run app/streamlit_app.py  # opens the demo at localhost:8501
```

> [!important]
> `make setup` uses `python3.13` by default (`PYTHON=python3.13` in the `Makefile`) — if that's not on your `PATH`, either install it or run `make setup PYTHON=python3.11` (3.11+ works fine, this project has no 3.13-only dependency).
>
> `setup` and `reproduce` are the only two `Makefile` targets — deliberately, as a small example of the pattern rather than a wrapper around everything. Every other command in this README (training, the demo, linting, tests) is run directly; see "Development" below.

## Reproducibility

- **Seed**: 

## Development



## AI usage


## Contribution statement

| Member | Contribution | Responsible for |
|---|---|---|
| Daryna Shevchuk | 33.3% | - |
| Marharyta Paduchak | 33.3% | - |
| Viktoriya Kibyeryeva | 33.3% | - |

All members agree with the above distribution.

## License and citation

| | License | Notes |
|---|---|---|
| Code (this repo) | MIT — see [`LICENSE`](LICENSE) | Free to reuse, including commercially |
| Data | See [`data/README.md`](data/README.md) | **Not MIT** — the dataset's own terms apply, independent of this repo's code license |
| Trained model | Inherits the data's terms + this repo's code license | See [`models/README.md`](models/README.md) |
| Text / poster / slides | CC BY 4.0 (recommended) | Attribute this repo if reused |

> [!important] Why this matters: MIT on the repo does not cover the data
> The single most common licensing mistake in student projects: assuming an MIT `LICENSE` file at the repo root means *everything* in the repo — including any data you collected or redistributed — is under that license. It doesn't. Code and data are licensed independently; check both.

Citing this project: see [`CITATION.cff`](CITATION.cff) (GitHub renders a "Cite this repository" button from it automatically). Citing the underlying dataset: see [`data/README.md`](data/README.md).

## Acknowledgments

Built from [`ucu-ai-course/project-template`](https://github.com/ucu-ai-course/project-template). Thanks to the course instructors and our project mentor for feedback throughout the semester.

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
| Daryna Shevchuk | [@dasha-pn](https://github.com/dasha-pn) | Stage 2: Qwen classification and transcription, QLoRA fine-tuning, and detector integration |
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

### Step 1: Region Detection

#### Model
We use [DocLayout-YOLO](https://github.com/opendatalab/DocLayout-YOLO) to localize regions on each page before  text recognition. The detected regions are passed to Qwen for reading and interpretation.

#### Training
- **Initialization:** `doclayout_yolo_docstructbench_imgsz1024.pt` from the Hugging Face repo
  [`juliozhao/DocLayout-YOLO-DocStructBench`](https://huggingface.co/juliozhao/DocLayout-YOLO-DocStructBench).
- **Fine-tuning:** trained on the challenge dataset for 50 epochs at image size 1024, with batch size of 2 and AdamW as optimizer.
- **Split:** the list of page filenames was shuffled randomly (seed = 42) and the first 80% were used for training (1064 images) and the remaining 20% for validation (266 images). The split is at page level, so pages from the same writer, document or archival fund may appear in both sets.

Fine-tuning adapts the model from its pretrained domain which is mostly printed documents to the layout and handwriting conventions of RYKOPUS dataset.

#### Results
Evaluated on the held-out validation set (a predicted box counts as correct at IoU ≥ 0.5):

| Metric | Value |
|---|---|
| F1 | 0.889 |
| Precision | 0.890 |
| Recall | 0.888 |
| Confidence threshold | 0.35 |

The confidence threshold of 0.35 was chosen by sweeping the threshold and selecting the value that maximizes F1. Precision and recall are almost perfectly balanced at this point, so the detector neither over-predicts nor misses regions systematically.

#### Limitations
- Around 11% of regions are still missed or mislocalized.
- Errors at this stage propagate downstream since Qwen can't work with missed or partially covered region.

### Step 2: Region Classification and Transcription

#### Model and motivation

We use [Qwen3-VL-4B-Instruct](https://huggingface.co/Qwen/Qwen3-VL-4B-Instruct)
to classify and transcribe crops extracted from document regions.

Our research question is whether a single vision-language model can be adapted
to the different content types in Ukrainian documents. We use the same model
with task-specific prompts rather than separate recognizers for text, formulas,
and tables.

This choice does not imply that Qwen outperforms specialized OCR models.
We plan a controlled comparison on handwritten, printed, and annotation regions.
If a specialized recognizer performs better, we will integrate it into Stage 2
while retaining Qwen for classification, formulas, and tables.

#### Inference pipeline

Each detected region is processed in two steps:

1. **Classification:** Qwen predicts one of seven labels:
   `handwritten`, `printed`, `annotation`, `formula`, `table`, `image`, or `graph`.
2. **Type-specific transcription:** the predicted label determines the prompt
   used for the next model call.

| Region type | Requested output |
|---|---|
| Handwritten, printed, annotation | Text in reading order, preserving original spelling, punctuation, and incomplete words |
| Formula | LaTeX preserving mathematical symbols and structure |
| Table | Cell contents separated by `|`, with rows separated by newlines |
| Image, graph | No transcription; bounding box and predicted type are retained with empty text |

Text prompts explicitly instruct the model not to correct, translate, explain,
or complete the source text. These instructions define the intended behavior;
their reliability is assessed through evaluation.

Each output region contains its bounding box, predicted type, and transcription.
The integrated pipeline exports page-level predictions in the competition's
submission format.

#### Baseline

Our Stage 2 baseline is Qwen3-VL-4B-Instruct without a project-trained adapter,
loaded with 4-bit quantization.

This is our local experimental baseline. It should not be interpreted as a
reproduction of the competition baseline score cited in the proposal.

#### QLoRA pilot training

We implemented supervised multitask fine-tuning using QLoRA to reduce training
memory requirements on a local NVIDIA GTX 1080 Ti with 11 GB VRAM.

The base model is loaded with 4-bit NF4 quantization and remains frozen.
LoRA adapters are trained in the language model's attention projections:
`q_proj`, `k_proj`, `v_proj`, and `o_proj`. The visual encoder is not fine-tuned.

Each training region contributes one classification example. Regions eligible
for transcription additionally contribute a transcription example with the
reference text as the target.

Training uses token-level cross-entropy loss restricted to assistant-answer
tokens. Image and instruction tokens are excluded from the loss.

| Setting | Pilot value |
|---|---:|
| Training regions | 64 |
| Training examples | 123 |
| Epochs | 1 |
| Optimizer updates | 31 |
| Base quantization | 4-bit NF4 |
| LoRA rank | 8 |
| LoRA alpha | 16 |
| LoRA dropout | 0.05 |
| Optimizer | AdamW |
| Learning rate | 0.0001 |
| Weight decay | 0 |
| Micro-batch size | 1 |
| Gradient accumulation | 4 |
| Gradient clipping norm | 1.0 |

These settings describe a smoke test of the training pipeline, not a completed
hyperparameter search or full-dataset fine-tuning experiment.

#### Validation

Stage 2 uses a deterministic, source-stratified, page-disjoint holdout split:

- **Training:** 1,062 pages and 20,569 regions.
- **Validation:** 268 pages and 5,082 regions.

Of the validation regions, 4,955 meet the pipeline's transcription-scoring
criteria. We use one fixed split; k-fold cross-validation has not been performed.

Evaluation uses ground-truth bounding boxes to isolate Stage 2 from detection
errors. We distinguish two transcription settings:

- **Oracle type:** the reference region type selects the transcription prompt.
- **Routed:** Qwen's predicted type selects the transcription prompt.

We report classification accuracy and corpus raw character error rate (CER).
These local CER values do not implement the competition's complete normalization,
region-matching, and PageCER protocol and are not directly comparable to its
composite score.

#### Pilot results

| Validation metric | Base Qwen | Qwen + pilot QLoRA |
|---|---:|---:|
| Classification accuracy ↑ | 91.91% | 93.35% |
| Routed corpus raw CER ↓ | 74.55% | 86.84% |
| Handwritten oracle corpus raw CER ↓ | 70.72% | 86.44% |
| Formula oracle corpus raw CER ↓ | 115.47% | 64.51% |

The pilot adapter improved classification and formula transcription but
worsened overall and handwritten transcription. Observed failures included
repetitions and text unrelated to the crop.

We associate the degradation with the small training sample and initial
hyperparameter choices; controlled follow-up experiments are needed to isolate
their effects. The base model was selected for the current integrated inference
run because it achieved lower routed validation CER.

#### Detector integration and submission

The integration notebook combines the trained DocLayout-YOLO detector with the
validation-selected Qwen configuration.

Model selection, prompts, generation budget, detector settings, and failure
handling are recorded before transcription. Compatible saved predictions are
reused, and interrupted inference resumes from completed pages.

The current submission run uses:

- The supplied detector weights, `best.pt`.
- Base Qwen3-VL-4B-Instruct without the pilot adapter.
- Detector confidence threshold **0.25** and image size **1024**.
- A maximum of **512 generated tokens** per call.
- A fixed abstention rule: transcription responses reaching the token limit
  are exported as empty text while retaining the bounding box and predicted type.
- An invalid classification stops the run rather than receiving a guessed label.

The completed run covers **385 test pages** and contains **218 token-limit
abstentions**. Its uploaded submission received a Kaggle score of **0.57763**.

This score belongs to the complete detector-to-Qwen pipeline, not to the
QLoRA adapter or Stage 2 alone.

#### Limitations and next steps

- The current adapter was trained on only 64 regions.
- Hyperparameter tuning and larger-scale training remain unfinished.
- The detector and Stage 2 use different training/validation splits; a common
  held-out split is needed for integrated local evaluation.
- Test predictions were inspected during earlier debugging. The revised frozen
  workflow prevents further per-image tuning but does not restore test
  independence.
- The empty-text abstention rule was inherited from debugging and has not yet
  been independently validated as a quality improvement.
- Page-disjoint splitting does not guarantee writer or source-text independence.

Next steps are to verify training examples, scale up QLoRA training, compare
configurations on validation, complete competition-compatible local evaluation,
and compare Qwen with a specialized text recognizer.

#### Notebooks

- [Data preparation](notebooks/01_prepare_data.ipynb)
- [Qwen baseline](notebooks/02_qwen_baseline.ipynb)
- [QLoRA training](notebooks/03_train_qwen.ipynb)
- [Validation and error analysis](notebooks/04_evaluate_qwen.ipynb)
- [Prompt experiments](notebooks/05_experiments.ipynb)
- [Detector integration and submission](notebooks/06_detector_integration.ipynb)

Reusable implementation code is maintained in `src/rukopys/`.

## Results

The current integrated pipeline uses the trained DocLayout-YOLO detector and
base Qwen3-VL-4B-Instruct, selected over the pilot adapter using validation CER.

| Result | Value |
|---|---:|
| Integrated Kaggle submission score | 0.57763 |
| Test pages processed | 385 |
| Token-limit responses exported as empty text | 218 |

The QLoRA pilot improved classification but worsened overall transcription.
Detailed Stage 2 validation results and evaluation limitations are reported above.

## Quick start

```bash
git clone [<repo-url>](https://github.com/marharytapaduchak/mavka-rukopys) && cd mavka-rukopys
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
| Daryna Shevchuk | 33.3% | Stage 2 implementation: Qwen classification and transcription, task-specific prompts, QLoRA training pipeline and pilot, baseline/adapter evaluation, prompt experiments, and detector-to-submission integration |
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

- Region detection builds on [DocLayout-YOLO](https://github.com/opendatalab/DocLayout-YOLO)
  (Zhao et al., 2024, arXiv:2410.12628). We use the publicly released DocStructBench
  checkpoint as initialization.

- DocLayout-YOLO is built on YOLOv10 and the Ultralytics codebase.

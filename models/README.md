
# RUKOPYS — Models and Checkpoints

This directory documents the models used in the
MAVKA — RUKOPYS document understanding pipeline.

The project combines document layout detection with
region classification and transcription.

## Model Architecture

### Stage 1 — Document Region Detection

**Model:** DocLayout-YOLO (YOLOv10-based)

**Purpose:**
- Detect meaningful document regions.
- Predict bounding boxes.
- Extract regions for subsequent processing.

The detection experiments are available in:

[`notebooks/00_region_detection.ipynb`](../notebooks/00_region_detection.ipynb)

### Stage 2 — Region Classification and Transcription

**Model:** Qwen3-VL-4B-Instruct

**Purpose:**
- Classify detected regions.
- Transcribe handwritten and printed text.
- Recognize annotations.
- Represent mathematical formulas using LaTeX.
- Extract tables in a structured text format.

Images and graphs are classified without transcription.

The Qwen experiments are available in:

- [`02_qwen_baseline.ipynb`](../notebooks/02_qwen_baseline.ipynb)
- [`03_train_qwen.ipynb`](../notebooks/03_train_qwen.ipynb)
- [`04_evaluate_qwen.ipynb`](../notebooks/04_evaluate_qwen.ipynb)
- [`05_experiments.ipynb`](../notebooks/05_experiments.ipynb)

## Fine-Tuning

The project uses QLoRA for parameter-efficient
fine-tuning of Qwen3-VL.

Fine-tuning aims to improve region classification
and transcription accuracy on Ukrainian handwritten documents.

Training configurations and evaluation results
are documented in the corresponding notebooks.

## Model Checkpoints

Model weights and checkpoints are not stored directly
in this GitHub repository.

These artifacts may include:

| Artifact | Purpose |
|---|---|
| YOLO checkpoint | Trained document region detector |
| Qwen base model | Base vision-language model |
| QLoRA adapter | Fine-tuned transcription and classification adapter |
| Processor configuration | Model input preprocessing |

Checkpoint locations depend on the experiment configuration.

No public checkpoint download location is claimed
unless explicitly documented.

## Evaluation

The project evaluates:

- Detection F1 and IoU.
- Region classification accuracy.
- Character Error Rate (CER).
- Page-level Character Error Rate (Page CER).
- Overall Kaggle competition score.

Baseline and fine-tuned models should be compared
using consistent validation data and evaluation procedures.

## Reproducibility

Each experiment should record:

- Model identifier and version.
- Dataset revision and validation split.
- Training configuration and random seed.
- Checkpoint or adapter identifier.
- Evaluation protocol and metrics.

Large model artifacts should be stored separately
from the Git repository.

## Future Improvements

Planned experiments include:

1. Larger-scale QLoRA fine-tuning.
2. Hyperparameter optimization.
3. Silver-data experiments.
4. Postprocessing and OCR error correction.
5. Evaluation of alternative OCR models if necessary.

## References

- [Qwen3-VL-4B-Instruct](https://huggingface.co/Qwen/Qwen3-VL-4B-Instruct)
- [RUKOPYS Dataset](https://huggingface.co/datasets/UkrainianCatholicUniversity/rukopys)
- [Handwritten to Data Competition](https://www.kaggle.com/competitions/handwritten-to-data)

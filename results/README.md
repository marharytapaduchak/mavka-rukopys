
# RUKOPYS — Experimental Results

This directory is intended for storing evaluation results,
experiment summaries, and visualizations produced by the
MAVKA — RUKOPYS project.

## Evaluation Metrics

The project evaluates three main tasks:

| Task | Metrics |
|---|---|
| Region Detection | Precision, Recall, F1, IoU |
| Region Classification | Accuracy |
| Text Recognition | CER, Page CER |

## Experiment Results

Experiments are implemented in the notebooks located in
[`notebooks/`](../notebooks/).

Results should include:

- Model configuration and checkpoint information.
- Dataset split and random seed.
- Evaluation metrics.
- Comparison with baseline models.
- Error analysis and representative examples.

## Visualizations

Figures may include:

- Dataset statistics and class distributions.
- Detection predictions and bounding boxes.
- Training and validation curves.
- OCR error analysis.
- Model comparison plots.

## Reproducibility

Each experiment should document its configuration, dataset
version, and evaluation procedure.

Only verified RUKOPYS results should be reported here.
Large datasets and model checkpoints should be stored separately.

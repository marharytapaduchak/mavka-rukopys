
# RUKOPYS — Demo and Results Visualization

This document describes how to prepare visual demonstrations
of the MAVKA — RUKOPYS pipeline.

## Current Demonstration

The project currently uses Jupyter notebooks to demonstrate:

- Document region detection with DocLayout-YOLO.
- Region classification and transcription with Qwen3-VL.
- Model evaluation and error analysis.
- End-to-end inference and Kaggle submission generation.

See the [`notebooks/`](../notebooks/) directory.

## Recommended Visualizations

For presentations and project documentation, useful examples include:

1. Original document image.
2. Detected bounding boxes with region types.
3. Transcribed text for selected regions.
4. Comparison between predicted and reference transcription.
5. Evaluation metrics and representative errors.

## Future Demo

An interactive application may be developed to allow users
to upload documents and inspect predictions directly.

Any screenshots or demonstrations should be generated from
actual model outputs and clearly labeled.

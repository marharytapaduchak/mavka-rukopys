
# RUKOPYS — Demo Application

This directory is reserved for a future interactive demonstration
of the MAVKA — RUKOPYS document understanding pipeline.

## Current Status

An interactive application has not yet been implemented.

The current pipeline is developed and evaluated through
the notebooks in [`notebooks/`](../notebooks/).

## Planned Functionality

A future demo may allow users to:

1. Upload an image of a handwritten Ukrainian document.
2. Detect document regions using DocLayout-YOLO.
3. Classify and transcribe regions using Qwen3-VL.
4. Visualize detected bounding boxes.
5. View and export structured transcription results.

## Implementation

The reusable inference logic is available in
[`src/rukopys/inference.py`](../src/rukopys/inference.py).

The application framework and interface have not been selected yet.

# Notebook conventions

| Notebook | Purpose |
|---|---|
| `00_region_detection.ipynb` | Train and evaluate the DocLayout-YOLO detector; save `best.pt`. |
| `01_prepare_data.ipynb` | Prepare crops, manifests, and a reproducible train/validation split. |
| `02_qwen_baseline.ipynb` | Evaluate the base Qwen model on validation regions. |
| `03_train_qwen.ipynb` | Train a multitask QLoRA adapter for classification and transcription. |
| `04_evaluate_qwen.ipynb` | Evaluate the adapter, compare it with the baseline, and review errors. |
| `05_experiments.ipynb` | Compare transcription prompts on the same validation data. |
| `06_detector_integration.ipynb` | Combine detection and transcription, then export a Kaggle submission using a fixed inference policy. |

1. **Keep reusable logic in `src/rukopys/`.** Notebooks document experiments and import shared functions from the package. `src/core.py` remains a compatibility wrapper for older imports.

2. **Start with the goal and inputs.** Write conclusions after inspecting the results. Use English and a consistent team voice without starting every sentence with “we”.

3. **Place imports and configuration near the beginning.** Discover project paths relative to the working directory. Install dependencies separately from normal execution.

4. **Preserve saved artifacts.** Data, checkpoints, evaluations, and predictions are stored under `src/work/`. Reuse compatible completed results and resume interrupted work where supported. Keep the original directory location because saved artifacts may contain absolute paths.

5. **Separate different experiments.** Changes to models, training settings, prompts, or inference policies require a distinct run name or output directory. Do not overwrite previous results or bypass compatibility checks.

6. **Run `Restart & Run All` before committing.** Compatible cached results may be reused; verification does not require repeating completed GPU computations.

7. **Commit useful notebook outputs.** Retain concise metrics, tables, and representative figures. Store large datasets, predictions, and model weights separately. Check that repository hooks preserve the intended outputs.

8. **Coordinate notebook editing.** Assign one editor per notebook at a time to avoid difficult JSON merge conflicts.

9. **Separate training, validation, and test roles.** Train on the training partition and select configurations using validation. Notebook 06 fixes its policy before processing test images, without test annotations or manual test-specific repairs. Disclose earlier test inspection when reporting results.

10. **Describe results accurately.** Distinguish smoke tests from full training, cached results from new computations, and CSV validity from OCR quality. Claim Colab compatibility only after implementing and testing it.

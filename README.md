# Reliability-Aware Feedback-Controlled Fusion for HSI-LiDAR Classification

Clean research code for strict spatially disjoint HSI-LiDAR classification with
feedback-controlled adaptive fusion operator prioritization.

## Files

- `feedback_fusion_control.py` — PID feedback controller, dataset characterization, and hybrid operator scoring.
- `controlled_fusion_integration.py` — fusion operators, selectable fusion model, and controlled trainer.
- `run_strict_split_multirun.py` — strict spatially disjoint multi-run training and evaluation entry point.
- `requirements.txt` — minimal Python dependencies.

## Expected external file

The runner expects your dataset loader to be available as:

```text
dataset_loader_spatial_disjoint.py
```

and to expose:

```python
load_dataset(...)
```

returning train, validation, and test loaders plus dataset metadata.

## Example

```bash
python run_strict_split_multirun.py --dataset muufl --epochs 200 --device cuda --num_runs 10
python run_strict_split_multirun.py --dataset houston2013 --epochs 200 --device cuda --num_runs 10
python run_strict_split_multirun.py --dataset augsburg --epochs 200 --device cuda --num_runs 10
```

## Outputs

Each run saves:

- `best.pth`
- `summary.json`
- `confusion_matrix_val.csv`, `confusion_matrix_test.csv`
- `per_class_metrics_val.csv`, `per_class_metrics_test.csv`

The multi-run folder saves:

- `multi_run_results.csv`
- `multi_run_summary.json`
- `paper_table_mean_std.csv`

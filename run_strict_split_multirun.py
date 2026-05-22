"""
Strict spatially disjoint multi-run trainer for RAFC-Fusion.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import random
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import (
    accuracy_score,
    cohen_kappa_score,
    confusion_matrix,
    precision_recall_fscore_support,
)

try:
    from scipy.stats import ttest_rel
except Exception:  # pragma: no cover
    ttest_rel = None

from controlled_fusion_integration import ControlledFusionModel, ControlledTrainer
from dataset_loader_spatial_disjoint import load_dataset
from feedback_fusion_control import FusionMode


DATASET_DEFAULTS = {
    "houston2013": {"patch_size": 7, "batch_size": 32},
    "muufl": {"patch_size": 7, "batch_size": 32},
    "augsburg": {"patch_size": 7, "batch_size": 32},
}


def set_all_seeds(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if torch.backends.cudnn.is_available():
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def ensure_dir(path: str) -> str:
    os.makedirs(path, exist_ok=True)
    return path


def save_json(path: str, data: Dict[str, Any]) -> None:
    with open(path, "w", encoding="utf-8") as file:
        json.dump(make_json_safe(data), file, indent=2)


def write_csv(path: str, headers: List[str], rows: List[List[Any]]) -> None:
    with open(path, "w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(headers)
        writer.writerows(rows)


def make_json_safe(obj: Any) -> Any:
    if isinstance(obj, torch.Tensor):
        return None
    if isinstance(obj, FusionMode):
        return obj.value
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.float16, np.float32, np.float64)):
        return float(obj)
    if isinstance(obj, (np.int16, np.int32, np.int64)):
        return int(obj)
    if isinstance(obj, dict):
        return {str(key): make_json_safe(value) for key, value in obj.items() if value is not None}
    if isinstance(obj, list):
        return [make_json_safe(value) for value in obj]
    return obj


def build_loaders(
    dataset: str,
    data_root: Optional[str],
    patch_size: int,
    batch_size: int,
    n_train_per_class: int,
    n_val_per_class: int,
    seed: int,
    num_workers: int = 0,
):
    train_loader, val_loader, test_loader, info = load_dataset(
        dataset_name=dataset,
        batch_size=batch_size,
        patch_size=patch_size,
        num_workers=num_workers,
        data_root=data_root,
        n_train_per_class=n_train_per_class,
        n_val_per_class=n_val_per_class,
        seed=seed,
        use_spatial_buffer=False,
        val_from_train_blocks=True,
        force_rebuild_split=False,
        split_cache_dir="official_splits",
    )

    meta = {
        "hsi_channels": info["hsi_channels"],
        "lidar_channels": info["lidar_channels"],
        "num_classes": info["num_classes"],
        "num_train_samples": info["num_train_samples"],
        "num_val_samples": info["num_val_samples"],
        "num_test_samples": info["num_test_samples"],
        "n_train_per_class": info["n_train_per_class"],
        "n_val_per_class": info["n_val_per_class"],
        "seed": info["seed"],
    }
    return train_loader, val_loader, test_loader, meta


def evaluate_model_on_loader(
    model: torch.nn.Module,
    loader,
    device: str,
    num_classes: int,
    out_dir: str,
    split_name: str,
) -> Dict[str, float]:
    model.eval()
    y_pred, y_true = [], []

    with torch.no_grad():
        for batch in loader:
            hsi = batch["hsi"].to(device)
            lidar = batch["lidar"].to(device)
            labels = batch["label"].to(device)
            preds = model(hsi, lidar).argmax(dim=1)
            y_pred.append(preds.cpu().numpy())
            y_true.append(labels.cpu().numpy())

    y_pred = np.concatenate(y_pred)
    y_true = np.concatenate(y_true)

    overall_acc = accuracy_score(y_true, y_pred)
    kappa = cohen_kappa_score(y_true, y_pred)
    cm = confusion_matrix(y_true, y_pred, labels=np.arange(num_classes))
    per_class_acc = cm.diagonal() / cm.sum(axis=1).clip(min=1)

    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, labels=np.arange(num_classes), zero_division=0
    )
    macro_p, macro_r, macro_f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="macro", zero_division=0
    )
    weighted_p, weighted_r, weighted_f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="weighted", zero_division=0
    )

    ensure_dir(out_dir)
    pd.DataFrame(cm).to_csv(os.path.join(out_dir, f"confusion_matrix_{split_name}.csv"), index=False)
    pd.DataFrame(
        {
            "class": np.arange(num_classes),
            "accuracy": per_class_acc,
            "precision": precision,
            "recall": recall,
            "f1": f1,
        }
    ).to_csv(os.path.join(out_dir, f"per_class_metrics_{split_name}.csv"), index=False)

    return {
        "overall_acc": float(overall_acc),
        "aa": float(per_class_acc.mean()),
        "kappa": float(kappa),
        "macro_precision": float(macro_p),
        "macro_recall": float(macro_r),
        "macro_f1": float(macro_f1),
        "weighted_precision": float(weighted_p),
        "weighted_recall": float(weighted_r),
        "weighted_f1": float(weighted_f1),
    }


def extract_best_acc(result: Any) -> float:
    if isinstance(result, dict):
        return float(result.get("best_acc", 0.0))
    return float(result)


def train_one_run(
    dataset: str,
    device: str,
    epochs: int,
    patch_size: int,
    lr: float,
    data_root: Optional[str],
    n_train_per_class: int,
    n_val_per_class: int,
    seed: int,
    out_dir: str,
    fusion_mode: Optional[str] = None,
    baseline_checkpoint: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    set_all_seeds(seed)

    default = DATASET_DEFAULTS[dataset]
    train_loader, val_loader, test_loader, meta = build_loaders(
        dataset=dataset,
        data_root=data_root,
        patch_size=patch_size or default["patch_size"],
        batch_size=default["batch_size"],
        n_train_per_class=n_train_per_class,
        n_val_per_class=n_val_per_class,
        seed=seed,
    )

    model = ControlledFusionModel(
        hsi_channels=meta["hsi_channels"],
        lidar_channels=meta["lidar_channels"],
        num_classes=meta["num_classes"],
        embed_dim=256,
    )
    trainer = ControlledTrainer(model, train_loader, val_loader, device=device, lr=lr, epochs=epochs)

    ensure_dir(out_dir)
    fusion_log: List[Dict] = []
    accuracy_log: List[Dict] = []
    reusable_baseline_checkpoint = None

    if fusion_mode is not None:
        mode = FusionMode(fusion_mode)
        result = trainer.train_with_mode(mode, num_epochs=epochs)
        val_acc = extract_best_acc(result)
        trainer.model.set_fusion_mode(mode)
        label = f"mode_{mode.value}"
        reusable_baseline_checkpoint = {
            "state_dict": {key: value.cpu().clone() for key, value in trainer.model.state_dict().items()},
            "mode": mode,
            "val_acc": val_acc,
        }
    else:
        sample_batch = next(iter(train_loader))
        if baseline_checkpoint is not None:
            trainer.model.load_state_dict({k: v.to(device) for k, v in baseline_checkpoint["state_dict"].items()})
            trainer.model.set_fusion_mode(baseline_checkpoint["mode"])
            baseline_acc = float(baseline_checkpoint["val_acc"])
        else:
            baseline_result = trainer.train_with_mode(FusionMode.BASELINE_GHOST, num_epochs=epochs)
            baseline_acc = extract_best_acc(baseline_result)

        val_acc, fusion_log, accuracy_log = trainer.train_with_control(
            dataset_name=dataset,
            baseline_accuracy=baseline_acc,
            sample_batch=sample_batch,
        )
        label = "proposed"

    if trainer.best_model_state is not None:
        trainer.model.load_state_dict({k: v.to(device) for k, v in trainer.best_model_state.items()})
        trainer.model.set_fusion_mode(trainer.best_model_mode or FusionMode.BASELINE_GHOST)

    best_path = os.path.join(out_dir, "best.pth")
    torch.save(trainer.model.state_dict(), best_path)

    val_metrics = evaluate_model_on_loader(trainer.model, val_loader, device, meta["num_classes"], out_dir, "val")
    test_metrics = evaluate_model_on_loader(trainer.model, test_loader, device, meta["num_classes"], out_dir, "test")

    summary = {
        "dataset": dataset,
        "label": label,
        "split_protocol": {
            "train_per_class": meta["n_train_per_class"],
            "val_per_class": meta["n_val_per_class"],
            "num_train_samples": meta["num_train_samples"],
            "num_val_samples": meta["num_val_samples"],
            "num_test_samples": meta["num_test_samples"],
            "seed": meta["seed"],
        },
        "final_val_acc": float(val_acc),
        "val_overall_acc": val_metrics["overall_acc"],
        "val_aa": val_metrics["aa"],
        "val_kappa": val_metrics["kappa"],
        "test_overall_acc": test_metrics["overall_acc"],
        "test_aa": test_metrics["aa"],
        "test_kappa": test_metrics["kappa"],
        "test_macro_f1": test_metrics["macro_f1"],
        "best_pth": best_path,
        "active_mode": trainer.model.active_mode.value,
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "fusion_log": fusion_log,
        "accuracy_log": accuracy_log,
        "reusable_baseline_checkpoint": reusable_baseline_checkpoint,
    }

    save_json(
        os.path.join(out_dir, "summary.json"),
        {key: value for key, value in summary.items() if key != "reusable_baseline_checkpoint"},
    )
    return summary


def mean_std(values: List[float]) -> Tuple[float, float]:
    array = np.asarray(values, dtype=np.float64)
    return float(array.mean()), float(array.std(ddof=1 if len(array) > 1 else 0))


def paired_ttest(a: List[float], b: List[float]) -> Dict[str, Any]:
    if ttest_rel is None:
        return {"available": False, "reason": "scipy is not installed"}
    if len(a) != len(b) or len(a) < 2:
        return {"available": False, "reason": "paired arrays must have the same length and n >= 2"}
    statistic, p_value = ttest_rel(np.asarray(a), np.asarray(b))
    return {
        "available": True,
        "t_stat": float(statistic),
        "p_value": float(p_value),
        "significant_p_lt_0_05": bool(p_value < 0.05),
    }


def aggregate_runs(results: List[Dict[str, Any]], label: str) -> Dict[str, Any]:
    keys = ["test_overall_acc", "test_aa", "test_kappa", "test_macro_f1", "val_overall_acc"]
    summary = {"label": label, "num_runs": len(results)}
    for key in keys:
        mean, std = mean_std([item[key] for item in results])
        summary[f"{key}_mean"] = mean
        summary[f"{key}_std"] = std
        summary[f"raw_{key}"] = [item[key] for item in results]
    return summary


def format_pct(mean: float, std: float) -> str:
    return f"{mean * 100:.2f} ± {std * 100:.2f}"


def run_multi(
    dataset: str,
    device: str,
    epochs: int,
    patch_size: int,
    lr: float,
    data_root: Optional[str],
    n_train_per_class: int,
    n_val_per_class: int,
    base_seed: int,
    num_runs: int,
    baseline_mode: str,
) -> Dict[str, Any]:
    root = ensure_dir(os.path.join("results", dataset))
    multi_dir = ensure_dir(os.path.join(root, f"multi_run_{datetime.now().strftime('%Y%m%d_%H%M%S')}"))

    baseline_results = []
    proposed_results = []

    for run_index in range(num_runs):
        seed = base_seed + run_index

        baseline_out = ensure_dir(os.path.join(multi_dir, f"run_{run_index + 1:02d}_baseline"))
        baseline_result = train_one_run(
            dataset=dataset,
            device=device,
            epochs=epochs,
            patch_size=patch_size,
            lr=lr,
            data_root=data_root,
            n_train_per_class=n_train_per_class,
            n_val_per_class=n_val_per_class,
            seed=seed,
            out_dir=baseline_out,
            fusion_mode=baseline_mode,
        )
        baseline_results.append(baseline_result)

        proposed_out = ensure_dir(os.path.join(multi_dir, f"run_{run_index + 1:02d}_proposed"))
        proposed_result = train_one_run(
            dataset=dataset,
            device=device,
            epochs=epochs,
            patch_size=patch_size,
            lr=lr,
            data_root=data_root,
            n_train_per_class=n_train_per_class,
            n_val_per_class=n_val_per_class,
            seed=seed,
            out_dir=proposed_out,
            fusion_mode=None,
            baseline_checkpoint=baseline_result["reusable_baseline_checkpoint"],
        )
        proposed_results.append(proposed_result)

    baseline_summary = aggregate_runs(baseline_results, "baseline_ghost")
    proposed_summary = aggregate_runs(proposed_results, "proposed")
    oa_ttest = paired_ttest(
        proposed_summary["raw_test_overall_acc"],
        baseline_summary["raw_test_overall_acc"],
    )

    rows = []
    for index, (baseline, proposed) in enumerate(zip(baseline_results, proposed_results), start=1):
        rows.append(
            [
                index,
                baseline["split_protocol"]["seed"],
                baseline["test_overall_acc"],
                proposed["test_overall_acc"],
                proposed["test_overall_acc"] - baseline["test_overall_acc"],
                baseline["test_aa"],
                proposed["test_aa"],
                baseline["test_kappa"],
                proposed["test_kappa"],
                proposed["active_mode"],
            ]
        )

    write_csv(
        os.path.join(multi_dir, "multi_run_results.csv"),
        [
            "run",
            "seed",
            "baseline_oa",
            "proposed_oa",
            "oa_gain",
            "baseline_aa",
            "proposed_aa",
            "baseline_kappa",
            "proposed_kappa",
            "proposed_active_mode",
        ],
        rows,
    )

    summary = {
        "dataset": dataset,
        "num_runs": num_runs,
        "split_protocol": {
            "train_per_class": n_train_per_class,
            "val_per_class": n_val_per_class,
            "base_seed": base_seed,
        },
        "baseline_summary": baseline_summary,
        "proposed_summary": proposed_summary,
        "paired_t_test_oa": oa_ttest,
        "result_dir": multi_dir,
    }
    save_json(os.path.join(multi_dir, "multi_run_summary.json"), summary)

    paper_table = pd.DataFrame(
        [
            {
                "method": "Baseline-Ghost",
                "OA_mean_std_pct": format_pct(
                    baseline_summary["test_overall_acc_mean"],
                    baseline_summary["test_overall_acc_std"],
                ),
                "AA_mean_std_pct": format_pct(
                    baseline_summary["test_aa_mean"],
                    baseline_summary["test_aa_std"],
                ),
                "Kappa_mean_std": f"{baseline_summary['test_kappa_mean']:.4f} ± {baseline_summary['test_kappa_std']:.4f}",
            },
            {
                "method": "Proposed",
                "OA_mean_std_pct": format_pct(
                    proposed_summary["test_overall_acc_mean"],
                    proposed_summary["test_overall_acc_std"],
                ),
                "AA_mean_std_pct": format_pct(
                    proposed_summary["test_aa_mean"],
                    proposed_summary["test_aa_std"],
                ),
                "Kappa_mean_std": f"{proposed_summary['test_kappa_mean']:.4f} ± {proposed_summary['test_kappa_std']:.4f}",
            },
        ]
    )
    paper_table.to_csv(os.path.join(multi_dir, "paper_table_mean_std.csv"), index=False)
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser("Strict spatially disjoint HSI-LiDAR training")
    parser.add_argument("--dataset", choices=list(DATASET_DEFAULTS), required=True)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--patch_size", type=int, default=7)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--data_root", default=None)
    parser.add_argument("--n_train_per_class", type=int, default=7)
    parser.add_argument("--n_val_per_class", type=int, default=7)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num_runs", type=int, default=10)
    parser.add_argument("--baseline_mode", default="baseline_ghost", choices=["baseline", "baseline_ghost"])
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summary = run_multi(
        dataset=args.dataset,
        device=args.device,
        epochs=args.epochs,
        patch_size=args.patch_size,
        lr=args.lr,
        data_root=args.data_root,
        n_train_per_class=args.n_train_per_class,
        n_val_per_class=args.n_val_per_class,
        base_seed=args.seed,
        num_runs=args.num_runs,
        baseline_mode=args.baseline_mode,
    )
    print(f"Saved results to: {summary['result_dir']}")


if __name__ == "__main__":
    main()

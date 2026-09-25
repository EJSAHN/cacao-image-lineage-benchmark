#!/usr/bin/env python3
"""Shared utilities for the end-to-end leakage sensitivity analysis."""
from __future__ import annotations

import hashlib
import json
import math
import os
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import pandas as pd

TASK_LABEL_ORDERS: dict[str, list[str]] = {
    "cacao_coarse_three_class": ["healthy", "black_pod", "frosty_pod"],
    "cocoamonilia_four_stage": [
        "healthy",
        "m1_hump",
        "m2_spot",
        "m3_sporulation",
    ],
}

TASK_DISPLAY_NAMES = {
    "cacao_coarse_three_class": "Cacao three-class",
    "cocoamonilia_four_stage": "CocoaMonilia four-stage",
}

ARCHITECTURES = ["resnet18", "efficientnet_b0"]
EVALUATION_DESIGNS = ["PATH_RANDOM_5FOLD", "VERIFIED_SCENE_BLOCK_5FOLD"]
REPEAT_SEEDS = [20260731, 20261740, 20262749, 20263758, 20264767]


@dataclass(frozen=True)
class ProjectPaths:
    root: Path
    big: Path

    @property
    def stage3b_prepared(self) -> Path:
        return self.root / "results" / "stage3b_prepared"

    @property
    def stage3b_results(self) -> Path:
        return self.root / "results" / "stage3b"

    @property
    def prepared(self) -> Path:
        return self.root / "results" / "stage3d_prepared"

    @property
    def cache_manifest(self) -> Path:
        return self.prepared / "stage3d_image_cache_manifest.tsv"

    @property
    def cache_chunks(self) -> Path:
        return self.prepared / "cache_chunks"

    @property
    def cache_dir(self) -> Path:
        return self.big / "training_cache" / "stage3d_shortside256_v1"

    @property
    def cache_status(self) -> Path:
        return self.root / "results" / "stage3d_cache_tasks"

    @property
    def run_results(self) -> Path:
        return self.root / "results" / "stage3d_run_tasks"

    @property
    def aggregate(self) -> Path:
        return self.root / "results" / "stage3d"

    @property
    def figures(self) -> Path:
        return self.root / "results" / "figures" / "stage3d"

    @property
    def model_cache(self) -> Path:
        return self.big / "model_cache" / "stage2c"


def sha256_file(path: str | Path, chunk_size: int = 16 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        while True:
            chunk = handle.read(chunk_size)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def stable_hash(text: str, length: int = 20) -> str:
    return hashlib.sha1(text.encode("utf-8")).hexdigest()[:length]


def write_tsv(df: pd.DataFrame, path: str | Path, *, compress: bool | None = None) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if compress is None:
        compress = str(path).endswith(".gz")
    df.to_csv(
        path,
        sep="\t",
        index=False,
        na_rep="",
        compression="gzip" if compress else None,
    )


def read_tsv(path: str | Path, **kwargs) -> pd.DataFrame:
    return pd.read_csv(
        path,
        sep="\t",
        dtype=str,
        keep_default_na=False,
        na_values=[],
        low_memory=False,
        **kwargs,
    )


def json_dump(payload: object, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")


def class_weight_vector(labels: Sequence[str], label_order: Sequence[str]) -> np.ndarray:
    labels_arr = np.asarray(labels, dtype=object)
    counts = np.asarray([(labels_arr == label).sum() for label in label_order], dtype=float)
    if np.any(counts <= 0):
        raise RuntimeError(f"Training split lacks one or more labels: counts={counts.tolist()}")
    weights = len(labels_arr) / (len(label_order) * counts)
    return weights.astype(np.float32)


def confusion_matrix_numpy(
    y_true: Sequence[str], y_pred: Sequence[str], labels: Sequence[str]
) -> np.ndarray:
    lookup = {label: idx for idx, label in enumerate(labels)}
    matrix = np.zeros((len(labels), len(labels)), dtype=np.int64)
    for truth, pred in zip(y_true, y_pred):
        if truth not in lookup or pred not in lookup:
            raise ValueError(f"Unknown label in confusion matrix: {truth!r}/{pred!r}")
        matrix[lookup[truth], lookup[pred]] += 1
    return matrix


def expected_calibration_error(
    y_true: Sequence[str], probabilities: np.ndarray, labels: Sequence[str], n_bins: int = 15
) -> float:
    y_true_arr = np.asarray(y_true, dtype=object)
    probs = np.asarray(probabilities, dtype=np.float64)
    confidence = probs.max(axis=1)
    prediction = np.asarray(labels, dtype=object)[np.argmax(probs, axis=1)]
    correct = (prediction == y_true_arr).astype(float)
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    for idx in range(n_bins):
        if idx == n_bins - 1:
            mask = (confidence >= edges[idx]) & (confidence <= edges[idx + 1])
        else:
            mask = (confidence >= edges[idx]) & (confidence < edges[idx + 1])
        if not np.any(mask):
            continue
        ece += float(mask.mean()) * abs(float(correct[mask].mean()) - float(confidence[mask].mean()))
    return float(ece)


def multiclass_brier_score(
    y_true: Sequence[str], probabilities: np.ndarray, labels: Sequence[str]
) -> float:
    lookup = {label: idx for idx, label in enumerate(labels)}
    target = np.zeros_like(probabilities, dtype=np.float64)
    for row, label in enumerate(y_true):
        target[row, lookup[label]] = 1.0
    return float(np.mean(np.sum((np.asarray(probabilities, dtype=float) - target) ** 2, axis=1)))


def multiclass_metrics(
    y_true: Sequence[str],
    y_pred: Sequence[str],
    probabilities: np.ndarray,
    labels: Sequence[str],
) -> tuple[dict[str, float], pd.DataFrame, pd.DataFrame]:
    y_true_arr = np.asarray(y_true, dtype=object)
    y_pred_arr = np.asarray(y_pred, dtype=object)
    probs = np.asarray(probabilities, dtype=np.float64)
    if len(y_true_arr) == 0:
        raise ValueError("Cannot score an empty prediction set")
    matrix = confusion_matrix_numpy(y_true_arr, y_pred_arr, labels)
    per_class_rows: list[dict[str, object]] = []
    recalls: list[float] = []
    f1s: list[float] = []
    supports: list[int] = []
    for idx, label in enumerate(labels):
        tp = int(matrix[idx, idx])
        fn = int(matrix[idx, :].sum() - tp)
        fp = int(matrix[:, idx].sum() - tp)
        support = int(matrix[idx, :].sum())
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2.0 * precision * recall / (precision + recall) if precision + recall else 0.0
        per_class_rows.append(
            {
                "label": label,
                "support": support,
                "precision": precision,
                "recall": recall,
                "f1": f1,
            }
        )
        recalls.append(recall)
        f1s.append(f1)
        supports.append(support)
    accuracy = float(np.mean(y_true_arr == y_pred_arr))
    balanced_accuracy = float(np.mean(recalls))
    macro_f1 = float(np.mean(f1s))
    weighted_f1 = float(np.average(f1s, weights=np.asarray(supports, dtype=float)))
    clipped = np.clip(probs, 1e-12, 1.0)
    lookup = {label: idx for idx, label in enumerate(labels)}
    truth_indices = np.asarray([lookup[label] for label in y_true_arr], dtype=int)
    log_loss = float(-np.mean(np.log(clipped[np.arange(len(truth_indices)), truth_indices])))
    metrics = {
        "n": int(len(y_true_arr)),
        "accuracy": accuracy,
        "balanced_accuracy": balanced_accuracy,
        "macro_f1": macro_f1,
        "weighted_f1": weighted_f1,
        "log_loss": log_loss,
        "brier_score": multiclass_brier_score(y_true_arr, probs, labels),
        "ece_15bin": expected_calibration_error(y_true_arr, probs, labels, n_bins=15),
    }
    confusion_rows = []
    for i, true_label in enumerate(labels):
        for j, pred_label in enumerate(labels):
            confusion_rows.append(
                {
                    "true_label": true_label,
                    "predicted_label": pred_label,
                    "count": int(matrix[i, j]),
                }
            )
    return metrics, pd.DataFrame(per_class_rows), pd.DataFrame(confusion_rows)


def aggregate_lineage_predictions(
    predictions: pd.DataFrame, labels: Sequence[str]
) -> pd.DataFrame:
    prob_columns = [f"prob__{label}" for label in labels]
    rows: list[dict[str, object]] = []
    for lineage_id, group in predictions.groupby("final_strict_lineage_id", sort=True):
        unique_labels = sorted(group["true_label"].unique())
        if len(unique_labels) != 1:
            raise RuntimeError(
                f"Strict lineage {lineage_id} has inconsistent labels in end-to-end evaluation: {unique_labels}"
            )
        mean_probs = group[prob_columns].astype(float).mean(axis=0).to_numpy()
        pred_label = labels[int(np.argmax(mean_probs))]
        row: dict[str, object] = {
            "final_strict_lineage_id": lineage_id,
            "true_label": unique_labels[0],
            "predicted_label": pred_label,
            "n_physical_paths": len(group),
        }
        row.update({column: float(value) for column, value in zip(prob_columns, mean_probs)})
        rows.append(row)
    return pd.DataFrame(rows)


def seed_everything(seed: int) -> None:
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed % (2**32 - 1))
    try:
        import torch

        torch.manual_seed(seed)
        torch.use_deterministic_algorithms(True, warn_only=True)
    except ImportError:
        pass


def worker_seed(base_seed: int, worker_id: int) -> int:
    return int((base_seed + 1009 * (worker_id + 1)) % (2**32 - 1))


def t_interval(values: Sequence[float], confidence: float = 0.95) -> tuple[float, float, float]:
    from scipy.stats import t

    array = np.asarray(values, dtype=float)
    mean = float(np.mean(array))
    if len(array) < 2:
        return mean, float("nan"), float("nan")
    se = float(np.std(array, ddof=1) / math.sqrt(len(array)))
    critical = float(t.ppf((1.0 + confidence) / 2.0, df=len(array) - 1))
    return mean, mean - critical * se, mean + critical * se

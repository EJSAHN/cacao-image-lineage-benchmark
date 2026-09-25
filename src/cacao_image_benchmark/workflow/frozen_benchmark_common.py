#!/usr/bin/env python3
"""Shared utilities for frozen-feature benchmark leakage-aware image benchmarks."""
from __future__ import annotations

import hashlib
import json
import math
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Mapping, Sequence

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, log_loss

SEED = 20260731

TASK_LABEL_ORDERS: dict[str, list[str]] = {
    "cacao_coarse_three_class": ["healthy", "black_pod", "frosty_pod"],
    "cacao_causal_five_class": [
        "healthy",
        "black_pod",
        "frosty_pod",
        "mirid_damage",
        "pod_borer_damage",
    ],
    "cocoamonilia_four_stage": [
        "healthy",
        "m1_hump",
        "m2_spot",
        "m3_sporulation",
    ],
}

TASK_CLASS_WEIGHT: dict[str, str | None] = {
    "cacao_coarse_three_class": None,
    "cocoamonilia_four_stage": None,
    "cacao_causal_five_class": "balanced",
}

PRIMARY_FEATURE_SET = "efficientnet_b0"
PRIMARY_CLASSIFIER = "logistic"

CV_DESIGNS = {
    "PATH_RANDOM_5FOLD": "sample_id",
    "EXACT_COMPONENT_5FOLD": "exact_component_id",
    "STRICT_LINEAGE_5FOLD": "final_strict_lineage_id",
    "VERIFIED_SCENE_BLOCK_5FOLD": "final_split_block_id",
    "AMBIGUITY_SENS_BLOCK_5FOLD": "ambiguity_sensitive_block_id",
}

SOURCE_HOLDOUT_DESIGNS = {
    "SOURCE_HOLDOUT_NAIVE",
    "SOURCE_HOLDOUT_STRICT_SAFE",
    "SOURCE_HOLDOUT_SCENE_SAFE",
}

FIXED_DESIGNS = {
    "ORIGINAL_TRAIN_TO_VALIDATION",
    "ORIGINAL_TRAINVAL_TO_TEST",
}

METADATA_NUMERIC_COLUMNS = [
    "log_width",
    "log_height",
    "log_bytes",
    "aspect_ratio",
    "megapixels",
]

METADATA_CATEGORICAL_COLUMNS = [
    "image_format",
    "mode",
    "exif_make",
    "exif_model",
]


class UnionFind:
    def __init__(self, items: Iterable[str]):
        self.parent = {str(item): str(item) for item in items}
        self.rank = {str(item): 0 for item in items}

    def find(self, item: str) -> str:
        item = str(item)
        if item not in self.parent:
            self.parent[item] = item
            self.rank[item] = 0
        root = item
        while self.parent[root] != root:
            root = self.parent[root]
        while self.parent[item] != item:
            nxt = self.parent[item]
            self.parent[item] = root
            item = nxt
        return root

    def union(self, a: str, b: str) -> None:
        ra, rb = self.find(str(a)), self.find(str(b))
        if ra == rb:
            return
        # Deterministic tie-breaking makes IDs reproducible.
        if self.rank[ra] < self.rank[rb] or (
            self.rank[ra] == self.rank[rb] and ra > rb
        ):
            ra, rb = rb, ra
        self.parent[rb] = ra
        if self.rank[ra] == self.rank[rb]:
            self.rank[ra] += 1


def sha256_file(path: str | Path, chunk_size: int = 16 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        while True:
            chunk = handle.read(chunk_size)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def stable_id(prefix: str, values: Iterable[str], length: int = 16) -> str:
    payload = "|".join(sorted({str(v) for v in values if str(v)}))
    return f"{prefix}_{hashlib.sha1(payload.encode('utf-8')).hexdigest()[:length]}"


def write_tsv(df: pd.DataFrame, path: str | Path, *, compress: bool | None = None) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if compress is None:
        compress = str(path).endswith(".gz")
    compression = "gzip" if compress else None
    df.to_csv(path, sep="\t", index=False, na_rep="", compression=compression)


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


def safe_numeric(series: pd.Series, default: float = np.nan) -> pd.Series:
    out = pd.to_numeric(series, errors="coerce")
    if not math.isnan(default):
        out = out.fillna(default)
    return out


def normalize_text(series: pd.Series, missing: str = "missing") -> pd.Series:
    return (
        series.fillna("")
        .astype(str)
        .str.strip()
        .replace("", missing)
        .str.lower()
    )


def add_metadata_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    width = safe_numeric(out["width"]).clip(lower=1)
    height = safe_numeric(out["height"]).clip(lower=1)
    nbytes = safe_numeric(out["bytes"]).clip(lower=1)
    out["log_width"] = np.log(width)
    out["log_height"] = np.log(height)
    out["log_bytes"] = np.log(nbytes)
    out["aspect_ratio"] = width / height
    out["megapixels"] = width * height / 1_000_000.0
    for column in METADATA_CATEGORICAL_COLUMNS:
        if column not in out.columns:
            out[column] = "missing"
        out[column] = normalize_text(out[column])
    return out


def softmax(scores: np.ndarray) -> np.ndarray:
    scores = np.asarray(scores, dtype=np.float64)
    if scores.ndim == 1:
        scores = np.column_stack([-scores, scores])
    shifted = scores - np.nanmax(scores, axis=1, keepdims=True)
    exp = np.exp(shifted)
    denom = exp.sum(axis=1, keepdims=True)
    denom[denom == 0] = 1.0
    return exp / denom


def expected_calibration_error(
    y_true: Sequence[str],
    probabilities: np.ndarray,
    labels: Sequence[str],
    n_bins: int = 15,
) -> float:
    y_true_arr = np.asarray(y_true, dtype=object)
    probs = np.asarray(probabilities, dtype=np.float64)
    if probs.size == 0:
        return float("nan")
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


def multiclass_metrics(
    y_true: Sequence[str],
    y_pred: Sequence[str],
    labels: Sequence[str],
    probabilities: np.ndarray | None = None,
) -> dict[str, float]:
    y_true_arr = np.asarray(y_true, dtype=object)
    y_pred_arr = np.asarray(y_pred, dtype=object)
    cm = confusion_matrix(y_true_arr, y_pred_arr, labels=list(labels))
    row_sum = cm.sum(axis=1)
    recalls = np.divide(
        np.diag(cm),
        row_sum,
        out=np.zeros(len(labels), dtype=float),
        where=row_sum > 0,
    )
    metrics = {
        "n": float(len(y_true_arr)),
        "accuracy": float(accuracy_score(y_true_arr, y_pred_arr)),
        "balanced_accuracy": float(np.mean(recalls)),
        "macro_f1": float(
            f1_score(
                y_true_arr,
                y_pred_arr,
                labels=list(labels),
                average="macro",
                zero_division=0,
            )
        ),
        "weighted_f1": float(
            f1_score(
                y_true_arr,
                y_pred_arr,
                labels=list(labels),
                average="weighted",
                zero_division=0,
            )
        ),
    }
    if probabilities is not None:
        probs = np.asarray(probabilities, dtype=np.float64)
        try:
            label_to_index = {label: idx for idx, label in enumerate(labels)}
            target_index = np.asarray([label_to_index[str(value)] for value in y_true_arr], dtype=int)
            clipped = np.clip(probs[np.arange(len(target_index)), target_index], 1e-15, 1.0)
            metrics["log_loss"] = float(-np.mean(np.log(clipped)))
        except (KeyError, ValueError, IndexError):
            metrics["log_loss"] = float("nan")
        metrics["ece_15bin"] = expected_calibration_error(
            y_true_arr, probs, labels=list(labels), n_bins=15
        )
    else:
        metrics["log_loss"] = float("nan")
        metrics["ece_15bin"] = float("nan")
    return metrics


def per_class_metrics(
    y_true: Sequence[str],
    y_pred: Sequence[str],
    labels: Sequence[str],
) -> pd.DataFrame:
    cm = confusion_matrix(y_true, y_pred, labels=list(labels))
    rows: list[dict[str, object]] = []
    for idx, label in enumerate(labels):
        tp = float(cm[idx, idx])
        fn = float(cm[idx, :].sum() - tp)
        fp = float(cm[:, idx].sum() - tp)
        support = float(cm[idx, :].sum())
        recall = tp / (tp + fn) if tp + fn else 0.0
        precision = tp / (tp + fp) if tp + fp else 0.0
        f1 = (
            2.0 * precision * recall / (precision + recall)
            if precision + recall
            else 0.0
        )
        rows.append(
            {
                "label": label,
                "support": int(support),
                "precision": precision,
                "recall": recall,
                "f1": f1,
            }
        )
    return pd.DataFrame(rows)


def prediction_score_columns(labels: Sequence[str], classifier: str) -> list[str]:
    prefix = "prob" if classifier == "logistic" else "score"
    return [f"{prefix}__{label}" for label in labels]


def probabilities_from_prediction_frame(
    frame: pd.DataFrame, labels: Sequence[str], classifier: str
) -> np.ndarray | None:
    if classifier != "logistic":
        return None
    columns = [f"prob__{label}" for label in labels]
    return frame[columns].astype(float).to_numpy()


def decision_matrix_from_prediction_frame(
    frame: pd.DataFrame, labels: Sequence[str], classifier: str
) -> np.ndarray:
    prefix = "prob" if classifier == "logistic" else "score"
    columns = [f"{prefix}__{label}" for label in labels]
    return frame[columns].astype(float).to_numpy()


def aggregate_predictions_by_unit(
    frame: pd.DataFrame,
    labels: Sequence[str],
    classifier: str,
    unit_column: str = "analysis_unit_id",
) -> pd.DataFrame:
    score_columns = prediction_score_columns(labels, classifier)
    required = [unit_column, "y_true", *score_columns]
    missing = [column for column in required if column not in frame.columns]
    if missing:
        raise ValueError(f"Cannot aggregate predictions; missing columns: {missing}")
    work = frame.copy()
    for column in score_columns:
        work[column] = pd.to_numeric(work[column], errors="raise")
    label_counts = work.groupby(unit_column, sort=False)["y_true"].nunique()
    bad = label_counts[label_counts > 1]
    if not bad.empty:
        raise ValueError(
            f"{len(bad)} {unit_column} values contain multiple true labels; first={bad.index[0]}"
        )
    agg_spec: dict[str, str] = {column: "mean" for column in score_columns}
    passthrough = [
        "task_name",
        "config_id",
        "evaluation_design",
        "feature_set",
        "classifier",
        "heldout_source",
        "y_true",
        "final_strict_lineage_id",
        "final_split_block_id",
        "ambiguity_sensitive_block_id",
    ]
    for column in passthrough:
        if column in frame.columns and column not in agg_spec:
            agg_spec[column] = "first"
    aggregated = work.groupby(unit_column, sort=False, as_index=False).agg(agg_spec)
    scores = aggregated[score_columns].to_numpy(float)
    aggregated["y_pred"] = np.asarray(labels, dtype=object)[np.argmax(scores, axis=1)]
    aggregated["sample_id"] = aggregated[unit_column]
    aggregated["n_paths_aggregated"] = (
        work.groupby(unit_column, sort=False).size().reindex(aggregated[unit_column]).to_numpy()
    )
    return aggregated


def config_seed(config_id: str, extra: str = "") -> int:
    digest = hashlib.sha256(f"{SEED}|{config_id}|{extra}".encode("utf-8")).hexdigest()
    return int(digest[:8], 16)


def json_dump(data: object, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(data, handle, indent=2, sort_keys=True)
        handle.write("\n")


@dataclass(frozen=True)
class ProjectPaths:
    root: Path
    big: Path

    @property
    def stage2e(self) -> Path:
        return self.root / "results" / "stage2e"

    @property
    def stage2c_embeddings_meta(self) -> Path:
        return self.root / "results" / "stage2c_embeddings"

    @property
    def stage2c_matrix_dir(self) -> Path:
        return self.big / "embeddings" / "stage2c"

    @property
    def prepared(self) -> Path:
        return self.root / "results" / "stage3a_prepared"

    @property
    def task_results(self) -> Path:
        return self.root / "results" / "stage3a_tasks"

    @property
    def aggregate(self) -> Path:
        return self.root / "results" / "stage3a"

    @property
    def figures(self) -> Path:
        return self.root / "results" / "figures" / "stage3a"

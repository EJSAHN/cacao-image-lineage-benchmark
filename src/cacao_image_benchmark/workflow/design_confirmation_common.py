#!/usr/bin/env python3
"""Shared utilities for design-confirmation benchmarks."""
from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score

BASE_SEED = 20260731
REPEAT_SEEDS = [BASE_SEED + 1009 * i for i in range(20)]
ROBUST_REPEAT_SEEDS = REPEAT_SEEDS[:10]

TASK_LABEL_ORDERS: dict[str, list[str]] = {
    "cacao_coarse_three_class": ["healthy", "black_pod", "frosty_pod"],
    "cacao_causal_five_class": [
        "healthy", "black_pod", "frosty_pod", "mirid_damage", "pod_borer_damage"
    ],
    "cocoamonilia_four_stage": [
        "healthy", "m1_hump", "m2_spot", "m3_sporulation"
    ],
}

TASK_EXPECTED_UNITS = {
    "cacao_coarse_three_class": 8804,
    "cacao_causal_five_class": 9132,
    "cocoamonilia_four_stage": 1873,
}

TASK_CLASS_WEIGHT: dict[str, str | None] = {
    "cacao_coarse_three_class": None,
    "cacao_causal_five_class": "balanced",
    "cocoamonilia_four_stage": None,
}

CV_DESIGNS = {
    "PATH_RANDOM_5FOLD": "sample_id",
    "EXACT_COMPONENT_5FOLD": "exact_component_id",
    "STRICT_LINEAGE_5FOLD": "final_strict_lineage_id",
    "VERIFIED_SCENE_BLOCK_5FOLD": "final_split_block_id",
    "AMBIGUITY_SENS_BLOCK_5FOLD": "ambiguity_sensitive_block_id",
}

FIVECLASS_CV_DESIGNS = {
    "VERIFIED_SCENE_BLOCK_5FOLD": "final_split_block_id",
    "AMBIGUITY_SENS_BLOCK_5FOLD": "ambiguity_sensitive_block_id",
}

COCO_TEST_DESIGNS = [
    "COCO_TEST_NAIVE",
    "COCO_TEST_EXACT_SAFE",
    "COCO_TEST_STRICT_SAFE",
    "COCO_TEST_SCENE_SAFE",
    "COCO_TEST_AMBIGUITY_SAFE",
]

SOURCE_HOLDOUT_DESIGNS = [
    "SOURCE_HOLDOUT_NAIVE",
    "SOURCE_HOLDOUT_EXACT_SAFE",
    "SOURCE_HOLDOUT_STRICT_SAFE",
    "SOURCE_HOLDOUT_SCENE_SAFE",
    "SOURCE_HOLDOUT_AMBIGUITY_SAFE",
]

SOURCE_HOLDOUTS = [
    "fig_ghana_balanced",
    "fig_roboflow_mixed",
    "fig_spanish_yolov4",
]

TRAINING_MODES = ["ALL_PATHS", "LINEAGE_WEIGHTED", "LINEAGE_REPRESENTATIVE"]

METADATA_NUMERIC_COLUMNS = [
    "log_width", "log_height", "log_bytes", "aspect_ratio", "megapixels"
]
METADATA_CATEGORICAL_COLUMNS = ["image_format", "mode", "exif_make", "exif_model"]


class UnionFind:
    def __init__(self, items: Iterable[str]):
        self.parent = {str(x): str(x) for x in items}
        self.rank = {str(x): 0 for x in items}

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
        ra, rb = self.find(a), self.find(b)
        if ra == rb:
            return
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


def write_tsv(df: pd.DataFrame, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(
        path,
        sep="\t",
        index=False,
        na_rep="",
        compression="gzip" if str(path).endswith(".gz") else None,
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


def safe_numeric(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce")


def normalize_text(series: pd.Series, missing: str = "missing") -> pd.Series:
    return (
        series.fillna("").astype(str).str.strip().replace("", missing).str.lower()
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


def expected_calibration_error(
    y_true: Sequence[str], probabilities: np.ndarray, labels: Sequence[str], n_bins: int = 15
) -> float:
    y = np.asarray(y_true, dtype=object)
    probs = np.asarray(probabilities, dtype=float)
    confidence = probs.max(axis=1)
    pred = np.asarray(labels, dtype=object)[np.argmax(probs, axis=1)]
    correct = (pred == y).astype(float)
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    value = 0.0
    for idx in range(n_bins):
        mask = (confidence >= edges[idx]) & (
            confidence <= edges[idx + 1] if idx == n_bins - 1 else confidence < edges[idx + 1]
        )
        if np.any(mask):
            value += float(mask.mean()) * abs(float(correct[mask].mean()) - float(confidence[mask].mean()))
    return float(value)


def multiclass_brier_score(
    y_true: Sequence[str], probabilities: np.ndarray, labels: Sequence[str]
) -> float:
    label_to_idx = {label: idx for idx, label in enumerate(labels)}
    target = np.zeros_like(probabilities, dtype=float)
    for row, value in enumerate(y_true):
        target[row, label_to_idx[str(value)]] = 1.0
    return float(np.mean(np.sum((np.asarray(probabilities, float) - target) ** 2, axis=1)))


def multiclass_metrics(
    y_true: Sequence[str],
    y_pred: Sequence[str],
    labels: Sequence[str],
    probabilities: np.ndarray | None = None,
) -> dict[str, float]:
    y = np.asarray(y_true, dtype=object)
    pred = np.asarray(y_pred, dtype=object)
    cm = confusion_matrix(y, pred, labels=list(labels))
    row_sum = cm.sum(axis=1)
    recalls = np.divide(np.diag(cm), row_sum, out=np.zeros(len(labels), float), where=row_sum > 0)
    result = {
        "n": float(len(y)),
        "accuracy": float(accuracy_score(y, pred)),
        "balanced_accuracy": float(np.mean(recalls)),
        "macro_f1": float(f1_score(y, pred, labels=list(labels), average="macro", zero_division=0)),
        "weighted_f1": float(f1_score(y, pred, labels=list(labels), average="weighted", zero_division=0)),
        "log_loss": float("nan"),
        "ece_15bin": float("nan"),
        "multiclass_brier": float("nan"),
    }
    if probabilities is not None:
        probs = np.asarray(probabilities, dtype=float)
        label_to_idx = {label: idx for idx, label in enumerate(labels)}
        target_idx = np.asarray([label_to_idx[str(v)] for v in y], dtype=int)
        clipped = np.clip(probs[np.arange(len(y)), target_idx], 1e-15, 1.0)
        result["log_loss"] = float(-np.mean(np.log(clipped)))
        result["ece_15bin"] = expected_calibration_error(y, probs, labels, 15)
        result["multiclass_brier"] = multiclass_brier_score(y, probs, labels)
    return result


def per_class_metrics(y_true: Sequence[str], y_pred: Sequence[str], labels: Sequence[str]) -> pd.DataFrame:
    cm = confusion_matrix(y_true, y_pred, labels=list(labels))
    rows = []
    for idx, label in enumerate(labels):
        tp = float(cm[idx, idx])
        fn = float(cm[idx, :].sum() - tp)
        fp = float(cm[:, idx].sum() - tp)
        recall = tp / (tp + fn) if tp + fn else 0.0
        precision = tp / (tp + fp) if tp + fp else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        rows.append({
            "label": label,
            "support": int(cm[idx, :].sum()),
            "precision": precision,
            "recall": recall,
            "f1": f1,
        })
    return pd.DataFrame(rows)


def reliability_table(
    y_true: Sequence[str], probabilities: np.ndarray, labels: Sequence[str], n_bins: int = 10
) -> pd.DataFrame:
    y = np.asarray(y_true, dtype=object)
    probs = np.asarray(probabilities, dtype=float)
    conf = probs.max(axis=1)
    pred = np.asarray(labels, dtype=object)[np.argmax(probs, axis=1)]
    correct = (pred == y).astype(float)
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    rows = []
    for idx in range(n_bins):
        mask = (conf >= edges[idx]) & (
            conf <= edges[idx + 1] if idx == n_bins - 1 else conf < edges[idx + 1]
        )
        rows.append({
            "bin_index": idx + 1,
            "bin_low": edges[idx],
            "bin_high": edges[idx + 1],
            "n": int(mask.sum()),
            "mean_confidence": float(conf[mask].mean()) if np.any(mask) else float("nan"),
            "empirical_accuracy": float(correct[mask].mean()) if np.any(mask) else float("nan"),
            "calibration_gap": float(correct[mask].mean() - conf[mask].mean()) if np.any(mask) else float("nan"),
        })
    return pd.DataFrame(rows)


def aggregate_predictions_by_unit(
    frame: pd.DataFrame,
    labels: Sequence[str],
    unit_column: str = "analysis_unit_id",
) -> pd.DataFrame:
    score_columns = [f"prob__{label}" for label in labels]
    work = frame.copy()
    for column in score_columns:
        work[column] = pd.to_numeric(work[column], errors="raise")
    bad = work.groupby(unit_column)["y_true"].nunique()
    if (bad > 1).any():
        raise RuntimeError(f"Multiple labels within {unit_column}")
    agg = {column: "mean" for column in score_columns}
    for column in [
        "task_name", "config_id", "analysis_family", "evaluation_design", "feature_set",
        "training_mode", "repeat_index", "split_seed", "heldout_source", "y_true",
        "final_strict_lineage_id", "final_split_block_id", "ambiguity_sensitive_block_id",
    ]:
        if column in work.columns:
            agg[column] = "first"
    out = work.groupby(unit_column, sort=False, as_index=False).agg(agg)
    scores = out[score_columns].to_numpy(float)
    out["y_pred"] = np.asarray(labels, dtype=object)[np.argmax(scores, axis=1)]
    out["sample_id"] = out[unit_column]
    return out


def config_seed(config_id: str, extra: str = "") -> int:
    digest = hashlib.sha256(f"{BASE_SEED}|{config_id}|{extra}".encode()).hexdigest()
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
    def stage2c_meta(self) -> Path:
        return self.root / "results" / "stage2c_embeddings"

    @property
    def stage2c_matrix_dir(self) -> Path:
        return self.big / "embeddings" / "stage2c"

    @property
    def stage3a(self) -> Path:
        return self.root / "results" / "stage3a"

    @property
    def prepared(self) -> Path:
        return self.root / "results" / "stage3b_prepared"

    @property
    def task_results(self) -> Path:
        return self.root / "results" / "stage3b_tasks"

    @property
    def aggregate(self) -> Path:
        return self.root / "results" / "stage3b"

    @property
    def figures(self) -> Path:
        return self.root / "results" / "figures" / "stage3b"

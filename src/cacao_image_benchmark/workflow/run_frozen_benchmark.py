#!/usr/bin/env python3
"""Run one frozen-feature benchmark benchmark configuration."""
from __future__ import annotations

import argparse
import os
import time
import traceback
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.svm import LinearSVC
from threadpoolctl import threadpool_limits

from frozen_benchmark_common import (
    METADATA_CATEGORICAL_COLUMNS,
    METADATA_NUMERIC_COLUMNS,
    TASK_CLASS_WEIGHT,
    TASK_LABEL_ORDERS,
    ProjectPaths,
    config_seed,
    multiclass_metrics,
    prediction_score_columns,
    read_tsv,
    write_tsv,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True)
    parser.add_argument("--big", required=True)
    parser.add_argument("--array-index", type=int, required=True)
    parser.add_argument("--cpus", type=int, default=4)
    return parser.parse_args()


def class_counts(values: pd.Series, labels: list[str]) -> str:
    counts = values.value_counts()
    return "|".join(f"{label}:{int(counts.get(label, 0))}" for label in labels)


def build_estimator(
    feature_set: str,
    classifier: str,
    task_name: str,
    seed: int,
) -> Pipeline:
    class_weight = TASK_CLASS_WEIGHT[task_name]
    if classifier == "logistic":
        final_model = LogisticRegression(
            C=1.0,
            solver="lbfgs",
            max_iter=4000,
            class_weight=class_weight,
            random_state=seed,
        )
    elif classifier == "linear_svm":
        final_model = LinearSVC(
            C=1.0,
            class_weight=class_weight,
            dual="auto",
            max_iter=20000,
            random_state=seed,
        )
    else:
        raise ValueError(f"Unsupported classifier: {classifier}")

    if feature_set == "metadata":
        numeric = Pipeline(
            [
                ("impute", SimpleImputer(strategy="median")),
                ("scale", StandardScaler()),
            ]
        )
        categorical = Pipeline(
            [
                ("impute", SimpleImputer(strategy="most_frequent")),
                (
                    "onehot",
                    OneHotEncoder(handle_unknown="ignore", sparse_output=True),
                ),
            ]
        )
        transformer = ColumnTransformer(
            [
                ("numeric", numeric, METADATA_NUMERIC_COLUMNS),
                ("categorical", categorical, METADATA_CATEGORICAL_COLUMNS),
            ],
            remainder="drop",
        )
        return Pipeline([("features", transformer), ("classifier", final_model)])

    return Pipeline(
        [
            ("scale", StandardScaler()),
            ("classifier", final_model),
        ]
    )


def load_feature_matrix(
    samples: pd.DataFrame,
    feature_set: str,
    paths: ProjectPaths,
) -> pd.DataFrame | np.ndarray:
    if feature_set == "metadata":
        return samples[METADATA_NUMERIC_COLUMNS + METADATA_CATEGORICAL_COLUMNS].copy()
    rows = samples["embedding_row"].astype(int).to_numpy()
    if feature_set in {"resnet18", "concat"}:
        resnet = np.load(
            paths.stage2c_matrix_dir / "stage2c_resnet18_embeddings.npy",
            mmap_mode="r",
        )
        resnet_slice = np.asarray(resnet[rows], dtype=np.float32)
    if feature_set in {"efficientnet_b0", "concat"}:
        efficient = np.load(
            paths.stage2c_matrix_dir / "stage2c_efficientnet_b0_embeddings.npy",
            mmap_mode="r",
        )
        efficient_slice = np.asarray(efficient[rows], dtype=np.float32)
    if feature_set == "resnet18":
        return resnet_slice
    if feature_set == "efficientnet_b0":
        return efficient_slice
    if feature_set == "concat":
        return np.concatenate([resnet_slice, efficient_slice], axis=1)
    raise ValueError(f"Unsupported feature set: {feature_set}")


def subset_rows(matrix: pd.DataFrame | np.ndarray, indices: np.ndarray):
    if isinstance(matrix, pd.DataFrame):
        return matrix.iloc[indices]
    return matrix[indices]


def align_model_output(
    raw: np.ndarray,
    model_classes: list[str],
    labels: list[str],
) -> np.ndarray:
    raw = np.asarray(raw, dtype=float)
    if raw.ndim == 1:
        raw = np.column_stack([-raw, raw])
    aligned = np.full((raw.shape[0], len(labels)), np.nan, dtype=float)
    class_to_column = {label: idx for idx, label in enumerate(model_classes)}
    for idx, label in enumerate(labels):
        if label not in class_to_column:
            raise RuntimeError(f"Model output is missing class {label}; classes={model_classes}")
        aligned[:, idx] = raw[:, class_to_column[label]]
    return aligned


def split_iterator(
    config: pd.Series,
    samples: pd.DataFrame,
    paths: ProjectPaths,
):
    family = config["design_family"]
    design = config["evaluation_design"]
    heldout = config.get("heldout_source", "")
    key_to_index = {
        sample_id: idx for idx, sample_id in enumerate(samples["sample_id"].astype(str))
    }

    if family == "CV":
        assignments = read_tsv(paths.prepared / "stage3a_cv_assignments.tsv.gz")
        assignments = assignments[
            assignments["task_name"].eq(config["task_name"])
            & assignments["evaluation_design"].eq(design)
        ].copy()
        if len(assignments) != len(samples):
            raise RuntimeError(
                f"CV assignment count mismatch: assignments={len(assignments)} samples={len(samples)}"
            )
        assignment_map = assignments.set_index("sample_id")["fold"].astype(int).to_dict()
        folds = samples["sample_id"].map(assignment_map)
        if folds.isna().any():
            raise RuntimeError("CV assignments are missing sample IDs")
        folds = folds.astype(int).to_numpy()
        for fold in sorted(np.unique(folds)):
            yield int(fold), np.where(folds != fold)[0], np.where(folds == fold)[0]
        return

    if family == "FIXED":
        assignments = read_tsv(paths.prepared / "stage3a_fixed_assignments.tsv.gz")
        assignments = assignments[
            assignments["task_name"].eq(config["task_name"])
            & assignments["evaluation_design"].eq(design)
        ]
    elif family == "SOURCE_HOLDOUT":
        assignments = read_tsv(
            paths.prepared / "stage3a_source_holdout_assignments.tsv.gz"
        )
        assignments = assignments[
            assignments["task_name"].eq(config["task_name"])
            & assignments["evaluation_design"].eq(design)
            & assignments["heldout_source"].eq(heldout)
        ]
    else:
        raise ValueError(f"Unsupported design family: {family}")

    train_ids = assignments.loc[assignments["role"].eq("TRAIN"), "sample_id"]
    test_ids = assignments.loc[assignments["role"].eq("TEST"), "sample_id"]
    train_idx = np.array([key_to_index[sid] for sid in train_ids], dtype=int)
    test_idx = np.array([key_to_index[sid] for sid in test_ids], dtype=int)
    if len(train_idx) == 0 or len(test_idx) == 0:
        raise RuntimeError(f"Empty train/test split for {design}/{heldout}")
    yield 0, train_idx, test_idx


def main() -> int:
    args = parse_args()
    paths = ProjectPaths(Path(args.root), Path(args.big))
    paths.task_results.mkdir(parents=True, exist_ok=True)
    config_table = read_tsv(paths.prepared / "stage3a_config_table.tsv")
    match = config_table[config_table["array_index"].astype(int).eq(args.array_index)]
    if len(match) != 1:
        raise RuntimeError(f"Array index {args.array_index} maps to {len(match)} configs")
    config = match.iloc[0]
    config_id = config["config_id"]
    started = time.time()
    status_path = paths.task_results / f"stage3a_{config_id}_status.tsv"
    prediction_path = paths.task_results / f"stage3a_{config_id}_predictions.tsv.gz"
    fold_metric_path = paths.task_results / f"stage3a_{config_id}_fold_metrics.tsv"

    try:
        all_samples = read_tsv(paths.prepared / "stage3a_path_samples.tsv.gz")
        samples = all_samples[all_samples["task_name"].eq(config["task_name"])].copy()
        samples = samples.sort_values("sample_id").reset_index(drop=True)
        labels = TASK_LABEL_ORDERS[config["task_name"]]
        if set(samples["label"]) != set(labels):
            raise RuntimeError(
                f"Task label mismatch for {config['task_name']}: {sorted(samples['label'].unique())}"
            )
        X = load_feature_matrix(samples, config["feature_set"], paths)
        y = samples["label"].to_numpy(dtype=object)
        seed = config_seed(config_id)
        prediction_frames: list[pd.DataFrame] = []
        metric_rows: list[dict[str, object]] = []
        warning_messages: list[str] = []

        with threadpool_limits(limits=max(1, args.cpus)):
            for fold, train_idx, test_idx in split_iterator(config, samples, paths):
                train_labels = set(y[train_idx])
                test_labels = set(y[test_idx])
                if train_labels != set(labels) or test_labels != set(labels):
                    raise RuntimeError(
                        f"Fold {fold} lacks classes. train={train_labels}; test={test_labels}"
                    )
                estimator = build_estimator(
                    config["feature_set"],
                    config["classifier"],
                    config["task_name"],
                    seed + fold,
                )
                fold_start = time.time()
                with warnings.catch_warnings(record=True) as caught:
                    warnings.simplefilter("always")
                    estimator.fit(subset_rows(X, train_idx), y[train_idx])
                    for item in caught:
                        warning_messages.append(
                            f"fold={fold}:{item.category.__name__}:{str(item.message)}"
                        )
                fit_seconds = time.time() - fold_start
                predict_start = time.time()
                y_pred = estimator.predict(subset_rows(X, test_idx))
                model_classes = list(estimator.classes_)
                if config["classifier"] == "logistic":
                    raw_output = estimator.predict_proba(subset_rows(X, test_idx))
                    aligned = align_model_output(raw_output, model_classes, labels)
                    probabilities = aligned
                else:
                    raw_output = estimator.decision_function(subset_rows(X, test_idx))
                    aligned = align_model_output(raw_output, model_classes, labels)
                    probabilities = None
                predict_seconds = time.time() - predict_start

                fold_metrics = multiclass_metrics(
                    y[test_idx], y_pred, labels, probabilities=probabilities
                )
                metric_rows.append(
                    {
                        "config_id": config_id,
                        "task_name": config["task_name"],
                        "design_family": config["design_family"],
                        "evaluation_design": config["evaluation_design"],
                        "feature_set": config["feature_set"],
                        "classifier": config["classifier"],
                        "heldout_source": config.get("heldout_source", ""),
                        "fold": fold,
                        "train_n": len(train_idx),
                        "test_n": len(test_idx),
                        "train_class_counts": class_counts(samples.iloc[train_idx]["label"], labels),
                        "test_class_counts": class_counts(samples.iloc[test_idx]["label"], labels),
                        "fit_seconds": fit_seconds,
                        "predict_seconds": predict_seconds,
                        **fold_metrics,
                    }
                )

                prediction = samples.iloc[test_idx][
                    [
                        "task_name",
                        "sample_id",
                        "label",
                        "photo_path_id",
                        "analysis_unit_id",
                        "exact_component_id",
                        "final_strict_lineage_id",
                        "final_split_block_id",
                        "ambiguity_sensitive_block_id",
                        "source_dataset_id",
                        "source_archive",
                        "corrected_split",
                    ]
                ].copy()
                prediction = prediction.rename(columns={"label": "y_true"})
                prediction["config_id"] = config_id
                prediction["design_family"] = config["design_family"]
                prediction["evaluation_design"] = config["evaluation_design"]
                prediction["feature_set"] = config["feature_set"]
                prediction["classifier"] = config["classifier"]
                prediction["heldout_source"] = config.get("heldout_source", "")
                prediction["fold"] = fold
                prediction["y_pred"] = y_pred
                score_columns = prediction_score_columns(labels, config["classifier"])
                for idx, column in enumerate(score_columns):
                    prediction[column] = aligned[:, idx]
                prediction_frames.append(prediction)

        predictions = pd.concat(prediction_frames, ignore_index=True)
        fold_metrics_df = pd.DataFrame(metric_rows)
        write_tsv(predictions, prediction_path)
        write_tsv(fold_metrics_df, fold_metric_path)
        status = pd.DataFrame(
            [
                {
                    "array_index": args.array_index,
                    "config_id": config_id,
                    "task_name": config["task_name"],
                    "evaluation_design": config["evaluation_design"],
                    "feature_set": config["feature_set"],
                    "classifier": config["classifier"],
                    "heldout_source": config.get("heldout_source", ""),
                    "prediction_rows": len(predictions),
                    "folds_completed": len(metric_rows),
                    "warning_count": len(warning_messages),
                    "warning_messages": " || ".join(warning_messages[:20]),
                    "elapsed_seconds": time.time() - started,
                    "prediction_path": str(prediction_path),
                    "fold_metric_path": str(fold_metric_path),
                    "status": "COMPLETED",
                }
            ]
        )
        write_tsv(status, status_path)
        print(status.to_string(index=False))
        print(fold_metrics_df.to_string(index=False))
        return 0
    except Exception as exc:
        status = pd.DataFrame(
            [
                {
                    "array_index": args.array_index,
                    "config_id": config_id,
                    "task_name": config.get("task_name", ""),
                    "evaluation_design": config.get("evaluation_design", ""),
                    "feature_set": config.get("feature_set", ""),
                    "classifier": config.get("classifier", ""),
                    "heldout_source": config.get("heldout_source", ""),
                    "prediction_rows": 0,
                    "folds_completed": 0,
                    "warning_count": 0,
                    "warning_messages": "",
                    "elapsed_seconds": time.time() - started,
                    "prediction_path": str(prediction_path),
                    "fold_metric_path": str(fold_metric_path),
                    "status": "FAILED",
                    "error": f"{type(exc).__name__}: {exc}",
                }
            ]
        )
        write_tsv(status, status_path)
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    raise SystemExit(main())

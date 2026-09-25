#!/usr/bin/env python3
"""Run one matched-removal and shortcut controls visual-shortcut model configuration."""
from __future__ import annotations

import argparse
import sys
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
from threadpoolctl import threadpool_limits

from control_analysis_common import (
    METADATA_CATEGORICAL_COLUMNS,
    METADATA_NUMERIC_COLUMNS,
    TASK_CLASS_WEIGHT,
    TASK_LABEL_ORDERS,
    ProjectPaths,
    add_metadata_features,
    aggregate_predictions_by_unit,
    align_probabilities,
    config_seed,
    multiclass_metrics,
    read_tsv,
    write_tsv,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True, type=Path)
    parser.add_argument("--big", required=True, type=Path)
    parser.add_argument("--array-index", required=True, type=int)
    parser.add_argument("--cpus", type=int, default=4)
    return parser.parse_args()


def choose_representatives(samples: pd.DataFrame, indices: np.ndarray) -> np.ndarray:
    subset = samples.iloc[indices].copy()
    chosen: list[int] = []
    for _, group in subset.groupby("final_strict_lineage_id", sort=False):
        preferred = group[group["is_lineage_representative"].eq("YES")]
        if len(preferred):
            chosen.append(int(preferred.sort_values("sample_id").index[0]))
        else:
            chosen.append(int(group.sort_values("sample_id").index[0]))
    return np.asarray(chosen, dtype=int)


def load_matrix_features(samples: pd.DataFrame, feature_set: str, paths: ProjectPaths) -> np.ndarray:
    if feature_set == "normal_efficientnet":
        matrix = np.load(
            paths.stage2c_matrix_dir / "stage2c_efficientnet_b0_embeddings.npy",
            mmap_mode="r",
        )
        rows = samples["embedding_row"].astype(int).to_numpy()
        return np.asarray(matrix[rows], dtype=np.float32)
    feature_manifest = read_tsv(paths.prepared / "stage3c_feature_manifest.tsv")
    row_map = feature_manifest.set_index("photo_path_id")["visual_feature_row"].astype(int).to_dict()
    rows = samples["photo_path_id"].map(row_map)
    if rows.isna().any():
        raise RuntimeError("Visual feature manifest is missing sample paths")
    matrix = np.load(paths.feature_dir / f"stage3c_{feature_set}_features.npy", mmap_mode="r")
    return np.asarray(matrix[rows.astype(int).to_numpy()], dtype=np.float32)


def make_estimator(feature_set: str, task_name: str, seed: int) -> Pipeline:
    classifier = LogisticRegression(
        C=1.0,
        solver="lbfgs",
        max_iter=4000,
        class_weight=TASK_CLASS_WEIGHT[task_name],
        random_state=seed,
    )
    if feature_set == "metadata":
        preprocessor = ColumnTransformer([
            ("numeric", Pipeline([
                ("impute", SimpleImputer(strategy="median")),
                ("scale", StandardScaler()),
            ]), METADATA_NUMERIC_COLUMNS),
            ("categorical", Pipeline([
                ("impute", SimpleImputer(strategy="most_frequent")),
                ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=True)),
            ]), METADATA_CATEGORICAL_COLUMNS),
        ])
        return Pipeline([("preprocess", preprocessor), ("classifier", classifier)])
    return Pipeline([("scale", StandardScaler()), ("classifier", classifier)])


def split_iterator(config: pd.Series, samples: pd.DataFrame):
    assignment = read_tsv(config["assignment_path"])
    lookup = {sid: idx for idx, sid in enumerate(samples["sample_id"].astype(str))}
    if config["analysis_family"] == "REPEATED_SCENE_CV":
        fold_map = assignment.set_index("sample_id")["fold"].astype(int).to_dict()
        folds = samples["sample_id"].map(fold_map)
        if folds.isna().any():
            raise RuntimeError("Shortcut repeated assignment is missing sample IDs")
        values = folds.astype(int).to_numpy()
        for fold in range(5):
            raw_train = np.where(values != fold)[0]
            raw_test = np.where(values == fold)[0]
            # Training is lineage-balanced; evaluation remains path-weighted and is also
            # summarized after aggregation to strict image lineages.
            yield fold, choose_representatives(samples, raw_train), raw_test, len(raw_train)
        return
    train_ids = assignment.loc[assignment["role"].eq("TRAIN"), "sample_id"]
    test_ids = assignment.loc[assignment["role"].eq("TEST"), "sample_id"]
    raw_train = np.asarray([lookup[sid] for sid in train_ids], dtype=int)
    test_idx = np.asarray([lookup[sid] for sid in test_ids], dtype=int)
    yield 0, choose_representatives(samples, raw_train), test_idx, len(raw_train)


def main() -> int:
    args = parse_args()
    paths = ProjectPaths(args.root, args.big)
    paths.shortcut_tasks.mkdir(parents=True, exist_ok=True)
    configs = read_tsv(paths.prepared / "stage3c_shortcut_config_table.tsv")
    match = configs[pd.to_numeric(configs["array_index"]).eq(args.array_index)]
    if len(match) != 1:
        raise RuntimeError(f"Array index {args.array_index} maps to {len(match)} shortcut configs")
    config = match.iloc[0]
    config_id = str(config["config_id"])
    status_path = paths.shortcut_tasks / f"stage3c_{config_id}_status.tsv"
    metric_path = paths.shortcut_tasks / f"stage3c_{config_id}_fold_metrics.tsv"
    prediction_path = paths.shortcut_tasks / f"stage3c_{config_id}_predictions.tsv.gz"
    audit_path = paths.shortcut_tasks / f"stage3c_{config_id}_training_audit.tsv"
    started = time.time()
    try:
        all_samples = read_tsv(paths.stage3b_prepared / "stage3b_path_samples.tsv.gz")
        samples = all_samples[all_samples["task_name"].eq(config["task_name"])].copy()
        samples = samples.sort_values("sample_id").reset_index(drop=True)
        samples["embedding_row"] = pd.to_numeric(samples["embedding_row"], errors="raise").astype(int)
        samples = add_metadata_features(samples)
        labels = TASK_LABEL_ORDERS[str(config["task_name"])]
        if set(samples["label"]) != set(labels):
            raise RuntimeError("Shortcut task labels differ from contract")
        feature_set = str(config["feature_set"])
        X = samples if feature_set == "metadata" else load_matrix_features(samples, feature_set, paths)
        y = samples["label"].to_numpy(dtype=object)
        seed = int(config["split_seed"] or config_seed(config_id))
        predictions: list[pd.DataFrame] = []
        metrics_rows: list[dict[str, object]] = []
        audit_rows: list[dict[str, object]] = []
        warnings_seen: list[str] = []
        with threadpool_limits(limits=max(1, args.cpus)):
            for fold, train_idx, test_idx, raw_train_count in split_iterator(config, samples):
                if set(y[train_idx]) != set(labels) or set(y[test_idx]) != set(labels):
                    raise RuntimeError(f"Fold {fold} lacks one or more labels")
                estimator = make_estimator(feature_set, str(config["task_name"]), seed + fold)
                with warnings.catch_warnings(record=True) as caught:
                    warnings.simplefilter("always")
                    estimator.fit(X.iloc[train_idx] if feature_set == "metadata" else X[train_idx], y[train_idx])
                    warnings_seen.extend(f"fold={fold}:{w.category.__name__}:{w.message}" for w in caught)
                raw = estimator.predict_proba(X.iloc[test_idx] if feature_set == "metadata" else X[test_idx])
                probs = align_probabilities(raw, list(estimator.classes_), labels)
                y_pred = np.asarray(labels, dtype=object)[np.argmax(probs, axis=1)]

                pred = samples.iloc[test_idx][[
                    "task_name", "sample_id", "label", "photo_path_id", "analysis_unit_id",
                    "final_strict_lineage_id", "final_split_block_id", "source_dataset_id",
                    "source_archive", "corrected_split",
                ]].copy().rename(columns={"label": "y_true"})
                pred["config_id"] = config_id
                pred["analysis_family"] = config["analysis_family"]
                pred["evaluation_design"] = config["evaluation_design"]
                pred["feature_set"] = feature_set
                pred["repeat_index"] = config["repeat_index"]
                pred["split_seed"] = config["split_seed"]
                pred["heldout_source"] = config["heldout_source"]
                pred["fold"] = fold
                pred["y_pred"] = y_pred
                for index, label in enumerate(labels):
                    pred[f"prob__{label}"] = probs[:, index]

                path_values = multiclass_metrics(y[test_idx], y_pred, labels, probs)
                unit_predictions = aggregate_predictions_by_unit(pred, labels)
                unit_probs = unit_predictions[[f"prob__{label}" for label in labels]].to_numpy(float)
                unit_values = multiclass_metrics(
                    unit_predictions["y_true"], unit_predictions["y_pred"], labels, unit_probs
                )
                for weighting, values in [("PATH", path_values), ("STRICT_LINEAGE", unit_values)]:
                    metrics_rows.append({
                        "config_id": config_id,
                        "analysis_family": config["analysis_family"],
                        "task_name": config["task_name"],
                        "evaluation_design": config["evaluation_design"],
                        "feature_set": feature_set,
                        "repeat_index": config["repeat_index"],
                        "split_seed": config["split_seed"],
                        "heldout_source": config["heldout_source"],
                        "evaluation_weighting": weighting,
                        "fold": fold,
                        "train_n": len(train_idx),
                        "test_n": int(values["n"]),
                        **values,
                    })
                audit_rows.append({
                    "config_id": config_id,
                    "fold": fold,
                    "train_paths_before_representative_selection": raw_train_count,
                    "train_strict_lineages_used": len(train_idx),
                    "test_paths_used": len(test_idx),
                    "test_strict_lineages_used": int(unit_values["n"]),
                })
                predictions.append(pred)
        prediction_df = pd.concat(predictions, ignore_index=True)
        metric_df = pd.DataFrame(metrics_rows)
        audit_df = pd.DataFrame(audit_rows)
        write_tsv(prediction_df, prediction_path)
        write_tsv(metric_df, metric_path)
        write_tsv(audit_df, audit_path)
        status = pd.DataFrame([{
            "array_index": args.array_index,
            "config_id": config_id,
            "analysis_family": config["analysis_family"],
            "task_name": config["task_name"],
            "evaluation_design": config["evaluation_design"],
            "feature_set": feature_set,
            "repeat_index": config["repeat_index"],
            "heldout_source": config["heldout_source"],
            "prediction_rows": len(prediction_df),
            "folds_completed": int(metric_df["fold"].nunique()),
            "warning_count": len(warnings_seen),
            "warning_messages": " || ".join(warnings_seen[:20]),
            "elapsed_seconds": time.time() - started,
            "metric_path": str(metric_path),
            "prediction_path": str(prediction_path),
            "training_audit_path": str(audit_path),
            "status": "COMPLETED",
        }])
        write_tsv(status, status_path)
        print(status.to_string(index=False))
        print(metric_df.to_string(index=False))
        return 0
    except Exception as exc:
        write_tsv(pd.DataFrame([{
            "array_index": args.array_index,
            "config_id": config_id,
            "analysis_family": config.get("analysis_family", ""),
            "task_name": config.get("task_name", ""),
            "evaluation_design": config.get("evaluation_design", ""),
            "feature_set": config.get("feature_set", ""),
            "repeat_index": config.get("repeat_index", ""),
            "heldout_source": config.get("heldout_source", ""),
            "prediction_rows": 0,
            "folds_completed": 0,
            "warning_count": 0,
            "elapsed_seconds": time.time() - started,
            "metric_path": str(metric_path),
            "prediction_path": str(prediction_path),
            "training_audit_path": str(audit_path),
            "status": "FAILED",
            "message": f"{type(exc).__name__}: {exc}",
        }]), status_path)
        print(f"FATAL: {type(exc).__name__}: {exc}", file=sys.stderr)
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    raise SystemExit(main())

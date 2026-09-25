#!/usr/bin/env python3
"""Run one design-confirmation model configuration."""
from __future__ import annotations

import argparse
import time
import traceback
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits

from design_confirmation_common import (
    TASK_CLASS_WEIGHT,
    TASK_LABEL_ORDERS,
    ProjectPaths,
    config_seed,
    multiclass_metrics,
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


def load_features(samples: pd.DataFrame, feature_set: str, paths: ProjectPaths) -> np.ndarray:
    rows = samples["embedding_row"].astype(int).to_numpy()
    if feature_set in {"resnet18", "concat"}:
        matrix = np.load(paths.stage2c_matrix_dir / "stage2c_resnet18_embeddings.npy", mmap_mode="r")
        resnet = np.asarray(matrix[rows], dtype=np.float32)
    if feature_set in {"efficientnet_b0", "concat"}:
        matrix = np.load(paths.stage2c_matrix_dir / "stage2c_efficientnet_b0_embeddings.npy", mmap_mode="r")
        efficient = np.asarray(matrix[rows], dtype=np.float32)
    if feature_set == "resnet18":
        return resnet
    if feature_set == "efficientnet_b0":
        return efficient
    if feature_set == "concat":
        return np.concatenate([resnet, efficient], axis=1)
    raise ValueError(f"Unsupported feature_set: {feature_set}")


def align_probabilities(raw: np.ndarray, model_classes: list[str], labels: list[str]) -> np.ndarray:
    out = np.full((len(raw), len(labels)), np.nan, dtype=float)
    lookup = {str(label): idx for idx, label in enumerate(model_classes)}
    for idx, label in enumerate(labels):
        if label not in lookup:
            raise RuntimeError(f"Model output missing class {label}; classes={model_classes}")
        out[:, idx] = raw[:, lookup[label]]
    return out


def choose_training_rows(
    samples: pd.DataFrame, train_idx: np.ndarray, training_mode: str
) -> tuple[np.ndarray, np.ndarray | None, dict[str, object]]:
    train = samples.iloc[train_idx].copy()
    n_original = len(train)
    n_lineages = train["final_strict_lineage_id"].nunique()
    fallback_representatives = 0
    if training_mode == "ALL_PATHS":
        selected = train_idx
        weights = None
    elif training_mode == "LINEAGE_WEIGHTED":
        counts = train.groupby("final_strict_lineage_id")["sample_id"].transform("size").astype(float)
        # pandas 3 Copy-on-Write may expose a read-only NumPy view.
        # Request an owned writable array and avoid in-place scaling.
        weights = (1.0 / counts).to_numpy(dtype=float, copy=True)
        weights = weights * (len(weights) / weights.sum())
        selected = train_idx
    elif training_mode == "LINEAGE_REPRESENTATIVE":
        selected_rows = []
        for _, group in train.groupby("final_strict_lineage_id", sort=False):
            preferred = group[group["is_lineage_representative"].eq("YES")]
            if len(preferred) == 1:
                chosen_index = preferred.index[0]
            else:
                chosen_index = group.sort_values("sample_id").index[0]
                fallback_representatives += 1
            selected_rows.append(chosen_index)
        selected = np.asarray(selected_rows, dtype=int)
        weights = None
    else:
        raise ValueError(f"Unsupported training_mode: {training_mode}")

    selected_frame = samples.loc[selected]
    if selected_frame["final_strict_lineage_id"].nunique() != n_lineages:
        raise RuntimeError("Training selection lost one or more strict lineages")
    if training_mode == "LINEAGE_REPRESENTATIVE" and len(selected_frame) != n_lineages:
        raise RuntimeError("Representative training must contain exactly one row per lineage")
    if weights is None:
        effective_n = float(len(selected))
        weight_min = weight_max = weight_sum = float("nan")
    else:
        weight_sum = float(weights.sum())
        effective_n = float(weight_sum ** 2 / np.sum(weights ** 2))
        weight_min = float(weights.min())
        weight_max = float(weights.max())
    audit = {
        "training_mode": training_mode,
        "train_paths_before_mode": n_original,
        "train_strict_lineages": n_lineages,
        "train_rows_after_mode": len(selected),
        "representative_fallback_lineages": fallback_representatives,
        "sample_weight_sum": weight_sum,
        "sample_weight_min": weight_min,
        "sample_weight_max": weight_max,
        "sample_weight_effective_n": effective_n,
    }
    return selected, weights, audit


def split_iterator(config: pd.Series, samples: pd.DataFrame):
    assignment = read_tsv(config["assignment_path"])
    lookup = {sid: idx for idx, sid in enumerate(samples["sample_id"].astype(str))}
    family = config["analysis_family"]
    if family == "REPEATED_CV":
        if len(assignment) != len(samples):
            raise RuntimeError(f"Assignment/sample mismatch: {len(assignment)} != {len(samples)}")
        fold_map = assignment.set_index("sample_id")["fold"].astype(int).to_dict()
        folds = samples["sample_id"].map(fold_map)
        if folds.isna().any():
            raise RuntimeError("Repeated assignment missing sample IDs")
        folds = folds.astype(int).to_numpy()
        for fold in range(5):
            yield fold, np.where(folds != fold)[0], np.where(folds == fold)[0]
        return
    train_ids = assignment.loc[assignment["role"].eq("TRAIN"), "sample_id"]
    test_ids = assignment.loc[assignment["role"].eq("TEST"), "sample_id"]
    train_idx = np.asarray([lookup[sid] for sid in train_ids], dtype=int)
    test_idx = np.asarray([lookup[sid] for sid in test_ids], dtype=int)
    if len(train_idx) == 0 or len(test_idx) == 0:
        raise RuntimeError("Empty fixed/source train or test set")
    yield 0, train_idx, test_idx


def class_counts(values: pd.Series, labels: list[str]) -> str:
    counts = values.value_counts()
    return "|".join(f"{label}:{int(counts.get(label, 0))}" for label in labels)


def main() -> int:
    args = parse_args()
    paths = ProjectPaths(Path(args.root), Path(args.big))
    paths.task_results.mkdir(parents=True, exist_ok=True)
    config_table = read_tsv(paths.prepared / "stage3b_config_table.tsv")
    match = config_table[config_table["array_index"].astype(int).eq(args.array_index)]
    if len(match) != 1:
        raise RuntimeError(f"Array index {args.array_index} maps to {len(match)} configs")
    config = match.iloc[0]
    config_id = config["config_id"]
    status_path = paths.task_results / f"stage3b_{config_id}_status.tsv"
    prediction_path = paths.task_results / f"stage3b_{config_id}_predictions.tsv.gz"
    fold_metric_path = paths.task_results / f"stage3b_{config_id}_fold_metrics.tsv"
    training_audit_path = paths.task_results / f"stage3b_{config_id}_training_audit.tsv"
    started = time.time()

    try:
        all_samples = read_tsv(paths.prepared / "stage3b_path_samples.tsv.gz")
        samples = all_samples[all_samples["task_name"].eq(config["task_name"])].copy()
        samples = samples.sort_values("sample_id").reset_index(drop=True)
        labels = TASK_LABEL_ORDERS[config["task_name"]]
        if set(samples["label"]) != set(labels):
            raise RuntimeError(f"Task labels differ from contract: {sorted(samples['label'].unique())}")
        X = load_features(samples, config["feature_set"], paths)
        y = samples["label"].to_numpy(dtype=object)
        seed = int(config["split_seed"] or config_seed(config_id))
        predictions = []
        fold_metrics = []
        training_audits = []
        warning_messages = []

        with threadpool_limits(limits=max(1, args.cpus)):
            for fold, train_idx, test_idx in split_iterator(config, samples):
                selected_idx, sample_weight, training_audit = choose_training_rows(
                    samples, train_idx, config["training_mode"]
                )
                if set(y[selected_idx]) != set(labels) or set(y[test_idx]) != set(labels):
                    raise RuntimeError(
                        f"Fold {fold} lacks classes; train={set(y[selected_idx])}; test={set(y[test_idx])}"
                    )
                estimator = Pipeline([
                    ("scale", StandardScaler()),
                    ("classifier", LogisticRegression(
                        C=1.0,
                        solver="lbfgs",
                        max_iter=4000,
                        class_weight=TASK_CLASS_WEIGHT[config["task_name"]],
                        random_state=seed + fold,
                    )),
                ])
                fit_kwargs = {}
                if sample_weight is not None:
                    fit_kwargs["classifier__sample_weight"] = sample_weight
                fit_start = time.time()
                with warnings.catch_warnings(record=True) as caught:
                    warnings.simplefilter("always")
                    estimator.fit(X[selected_idx], y[selected_idx], **fit_kwargs)
                    warning_messages.extend(
                        f"fold={fold}:{w.category.__name__}:{w.message}" for w in caught
                    )
                fit_seconds = time.time() - fit_start
                pred_start = time.time()
                raw = estimator.predict_proba(X[test_idx])
                probs = align_probabilities(raw, list(estimator.classes_), labels)
                y_pred = np.asarray(labels, dtype=object)[np.argmax(probs, axis=1)]
                predict_seconds = time.time() - pred_start
                metrics = multiclass_metrics(y[test_idx], y_pred, labels, probs)
                fold_metrics.append({
                    "config_id": config_id,
                    "analysis_family": config["analysis_family"],
                    "task_name": config["task_name"],
                    "evaluation_design": config["evaluation_design"],
                    "feature_set": config["feature_set"],
                    "training_mode": config["training_mode"],
                    "repeat_index": config["repeat_index"],
                    "split_seed": config["split_seed"],
                    "heldout_source": config["heldout_source"],
                    "fold": fold,
                    "train_n_raw": len(train_idx),
                    "train_n_used": len(selected_idx),
                    "test_n": len(test_idx),
                    "train_class_counts": class_counts(samples.iloc[selected_idx]["label"], labels),
                    "test_class_counts": class_counts(samples.iloc[test_idx]["label"], labels),
                    "fit_seconds": fit_seconds,
                    "predict_seconds": predict_seconds,
                    **metrics,
                })
                training_audits.append({
                    "config_id": config_id,
                    "fold": fold,
                    "analysis_family": config["analysis_family"],
                    "task_name": config["task_name"],
                    "evaluation_design": config["evaluation_design"],
                    "feature_set": config["feature_set"],
                    "training_mode": config["training_mode"],
                    "repeat_index": config["repeat_index"],
                    "split_seed": config["split_seed"],
                    "heldout_source": config["heldout_source"],
                    **training_audit,
                })
                pred = samples.iloc[test_idx][[
                    "task_name", "sample_id", "label", "photo_path_id", "analysis_unit_id",
                    "exact_component_id", "final_strict_lineage_id", "final_split_block_id",
                    "ambiguity_sensitive_block_id", "source_dataset_id", "source_archive", "corrected_split",
                ]].copy().rename(columns={"label": "y_true"})
                pred["config_id"] = config_id
                pred["analysis_family"] = config["analysis_family"]
                pred["evaluation_design"] = config["evaluation_design"]
                pred["feature_set"] = config["feature_set"]
                pred["training_mode"] = config["training_mode"]
                pred["repeat_index"] = config["repeat_index"]
                pred["split_seed"] = config["split_seed"]
                pred["heldout_source"] = config["heldout_source"]
                pred["fold"] = fold
                pred["y_pred"] = y_pred
                for idx, label in enumerate(labels):
                    pred[f"prob__{label}"] = probs[:, idx]
                predictions.append(pred)

        prediction_df = pd.concat(predictions, ignore_index=True)
        fold_metric_df = pd.DataFrame(fold_metrics)
        training_audit_df = pd.DataFrame(training_audits)
        write_tsv(prediction_df, prediction_path)
        write_tsv(fold_metric_df, fold_metric_path)
        write_tsv(training_audit_df, training_audit_path)
        status = pd.DataFrame([{
            "array_index": args.array_index,
            "config_id": config_id,
            "analysis_family": config["analysis_family"],
            "task_name": config["task_name"],
            "evaluation_design": config["evaluation_design"],
            "feature_set": config["feature_set"],
            "training_mode": config["training_mode"],
            "repeat_index": config["repeat_index"],
            "split_seed": config["split_seed"],
            "heldout_source": config["heldout_source"],
            "prediction_rows": len(prediction_df),
            "folds_completed": len(fold_metric_df),
            "warning_count": len(warning_messages),
            "warning_messages": " || ".join(warning_messages[:20]),
            "elapsed_seconds": time.time() - started,
            "prediction_path": str(prediction_path),
            "fold_metric_path": str(fold_metric_path),
            "training_audit_path": str(training_audit_path),
            "status": "COMPLETED",
        }])
        write_tsv(status, status_path)
        print(status.to_string(index=False))
        print(fold_metric_df.to_string(index=False))
        return 0
    except Exception as exc:
        status = pd.DataFrame([{
            "array_index": args.array_index,
            "config_id": config_id,
            "analysis_family": config.get("analysis_family", ""),
            "task_name": config.get("task_name", ""),
            "evaluation_design": config.get("evaluation_design", ""),
            "feature_set": config.get("feature_set", ""),
            "training_mode": config.get("training_mode", ""),
            "repeat_index": config.get("repeat_index", ""),
            "split_seed": config.get("split_seed", ""),
            "heldout_source": config.get("heldout_source", ""),
            "prediction_rows": 0,
            "folds_completed": 0,
            "warning_count": 0,
            "warning_messages": "",
            "elapsed_seconds": time.time() - started,
            "prediction_path": str(prediction_path),
            "fold_metric_path": str(fold_metric_path),
            "training_audit_path": str(training_audit_path),
            "status": "FAILED",
            "error": f"{type(exc).__name__}: {exc}",
        }])
        write_tsv(status, status_path)
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""Run one matched-removal or observed fixed-test configuration."""
from __future__ import annotations

import argparse
import sys
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

from control_analysis_common import (
    TASK_CLASS_WEIGHT,
    TASK_LABEL_ORDERS,
    ProjectPaths,
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


def scenario_frames(samples: pd.DataFrame, config: pd.Series) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    task = samples[samples["task_name"].eq(config["task_name"])].copy()
    if config["scenario_id"] == "coco_public_test":
        test = task[task["corrected_split"].eq("test")].copy()
        naive_train = task[task["corrected_split"].isin(["train", "validation"])].copy()
    else:
        source = str(config["heldout_source"])
        test = task[task["source_dataset_id"].eq(source)].copy()
        naive_train = task[~task["source_dataset_id"].eq(source)].copy()
    test_blocks = set(test["final_split_block_id"])
    contaminated = naive_train[naive_train["final_split_block_id"].isin(test_blocks)].copy()
    scene_safe = naive_train[~naive_train["final_split_block_id"].isin(test_blocks)].copy()
    return naive_train, test, contaminated, scene_safe


def class_counts(frame: pd.DataFrame, labels: list[str]) -> str:
    return "|".join(f"{label}:{int((frame['label'] == label).sum())}" for label in labels)


def source_counts(frame: pd.DataFrame) -> str:
    return "|".join(
        f"{source}:{int((frame['source_dataset_id'] == source).sum())}"
        for source in sorted(frame["source_dataset_id"].unique())
    )


def main() -> int:
    args = parse_args()
    paths = ProjectPaths(args.root, args.big)
    paths.null_tasks.mkdir(parents=True, exist_ok=True)
    configs = read_tsv(paths.prepared / "stage3c_null_config_table.tsv")
    match = configs[pd.to_numeric(configs["array_index"]).eq(args.array_index)]
    if len(match) != 1:
        raise RuntimeError(f"Array index {args.array_index} maps to {len(match)} null configs")
    config = match.iloc[0]
    config_id = str(config["config_id"])
    status_path = paths.null_tasks / f"stage3c_{config_id}_status.tsv"
    metric_path = paths.null_tasks / f"stage3c_{config_id}_metrics.tsv"
    audit_path = paths.null_tasks / f"stage3c_{config_id}_training_audit.tsv"
    prediction_path = paths.null_tasks / f"stage3c_{config_id}_predictions.tsv.gz"
    started = time.time()
    try:
        samples = read_tsv(paths.stage3b_prepared / "stage3b_path_samples.tsv.gz")
        samples["embedding_row"] = pd.to_numeric(samples["embedding_row"], errors="raise").astype(int)
        naive_train, test, contaminated, scene_safe = scenario_frames(samples, config)
        config_type = str(config["config_type"])
        removed_blocks: set[str] = set()
        if config_type == "OBSERVED_NAIVE":
            train = naive_train.copy()
        elif config_type == "OBSERVED_SCENE_SAFE":
            train = scene_safe.copy()
            removed_blocks = set(contaminated["final_split_block_id"])
        elif config_type == "MATCHED_NULL":
            removals = read_tsv(paths.prepared / "stage3c_null_removal_sets.tsv.gz")
            subset = removals[
                removals["scenario_id"].eq(config["scenario_id"])
                & removals["null_design"].eq(config["null_design"])
                & pd.to_numeric(removals["replicate"]).eq(int(config["replicate"]))
            ]
            removed_blocks = set(subset["block_id"])
            if not removed_blocks:
                raise RuntimeError("Matched-null removal set is empty")
            train = naive_train[~naive_train["final_split_block_id"].isin(removed_blocks)].copy()
        else:
            raise ValueError(f"Unsupported config type: {config_type}")

        labels = TASK_LABEL_ORDERS[str(config["task_name"])]
        if set(train["label"]) != set(labels) or set(test["label"]) != set(labels):
            raise RuntimeError("Train or test set lacks one or more classes")
        matrix = np.load(
            paths.stage2c_matrix_dir / "stage2c_efficientnet_b0_embeddings.npy",
            mmap_mode="r",
        )
        train_rows = train["embedding_row"].to_numpy(int)
        test_rows = test["embedding_row"].to_numpy(int)
        X_train = np.asarray(matrix[train_rows], dtype=np.float32)
        X_test = np.asarray(matrix[test_rows], dtype=np.float32)
        y_train = train["label"].to_numpy(dtype=object)
        y_test = test["label"].to_numpy(dtype=object)
        seed = config_seed(config_id)
        estimator = Pipeline([
            ("scale", StandardScaler()),
            ("classifier", LogisticRegression(
                C=1.0,
                solver="lbfgs",
                max_iter=4000,
                class_weight=TASK_CLASS_WEIGHT[str(config["task_name"])],
                random_state=seed,
            )),
        ])
        warning_messages: list[str] = []
        with threadpool_limits(limits=max(1, args.cpus)):
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                estimator.fit(X_train, y_train)
                warning_messages.extend(f"{w.category.__name__}:{w.message}" for w in caught)
            raw = estimator.predict_proba(X_test)
        probs = align_probabilities(raw, list(estimator.classes_), labels)
        y_pred = np.asarray(labels, dtype=object)[np.argmax(probs, axis=1)]
        path_metrics = multiclass_metrics(y_test, y_pred, labels, probs)

        predictions = test[[
            "task_name", "sample_id", "label", "photo_path_id", "analysis_unit_id",
            "final_strict_lineage_id", "final_split_block_id", "source_dataset_id",
            "source_archive", "corrected_split",
        ]].copy().rename(columns={"label": "y_true"})
        predictions["config_id"] = config_id
        predictions["scenario_id"] = config["scenario_id"]
        predictions["config_type"] = config_type
        predictions["null_design"] = config["null_design"]
        predictions["replicate"] = config["replicate"]
        predictions["y_pred"] = y_pred
        for index, label in enumerate(labels):
            predictions[f"prob__{label}"] = probs[:, index]
        unit_predictions = aggregate_predictions_by_unit(predictions, labels)
        unit_probs = unit_predictions[[f"prob__{label}" for label in labels]].to_numpy(float)
        unit_metrics = multiclass_metrics(
            unit_predictions["y_true"], unit_predictions["y_pred"], labels, unit_probs
        )
        metric_rows = []
        for weighting, values in [("PATH", path_metrics), ("STRICT_LINEAGE", unit_metrics)]:
            metric_rows.append({
                "config_id": config_id,
                "scenario_id": config["scenario_id"],
                "task_name": config["task_name"],
                "heldout_source": config["heldout_source"],
                "config_type": config_type,
                "null_design": config["null_design"],
                "replicate": config["replicate"],
                "evaluation_weighting": weighting,
                **values,
            })
        write_tsv(pd.DataFrame(metric_rows), metric_path)
        removed = naive_train[naive_train["final_split_block_id"].isin(removed_blocks)]
        audit = pd.DataFrame([{
            "config_id": config_id,
            "scenario_id": config["scenario_id"],
            "config_type": config_type,
            "null_design": config["null_design"],
            "replicate": config["replicate"],
            "naive_train_paths": len(naive_train),
            "train_paths_used": len(train),
            "test_paths": len(test),
            "removed_paths": len(removed),
            "removed_blocks": removed["final_split_block_id"].nunique(),
            "removed_strict_lineages": removed["final_strict_lineage_id"].nunique(),
            "removed_class_counts": class_counts(removed, labels),
            "removed_source_counts": source_counts(removed) if len(removed) else "",
            "train_class_counts": class_counts(train, labels),
            "test_class_counts": class_counts(test, labels),
            "warning_count": len(warning_messages),
            "warning_messages": " || ".join(warning_messages[:20]),
        }])
        write_tsv(audit, audit_path)
        if config_type.startswith("OBSERVED"):
            write_tsv(predictions, prediction_path)
        status = pd.DataFrame([{
            "array_index": args.array_index,
            "config_id": config_id,
            "scenario_id": config["scenario_id"],
            "config_type": config_type,
            "null_design": config["null_design"],
            "replicate": config["replicate"],
            "train_paths": len(train),
            "test_paths": len(test),
            "warning_count": len(warning_messages),
            "elapsed_seconds": time.time() - started,
            "metric_path": str(metric_path),
            "training_audit_path": str(audit_path),
            "prediction_path": str(prediction_path) if config_type.startswith("OBSERVED") else "",
            "status": "COMPLETED",
        }])
        write_tsv(status, status_path)
        print(status.to_string(index=False))
        print(pd.DataFrame(metric_rows).to_string(index=False))
        return 0
    except Exception as exc:
        write_tsv(pd.DataFrame([{
            "array_index": args.array_index,
            "config_id": config_id,
            "scenario_id": config.get("scenario_id", ""),
            "config_type": config.get("config_type", ""),
            "null_design": config.get("null_design", ""),
            "replicate": config.get("replicate", ""),
            "train_paths": 0,
            "test_paths": 0,
            "warning_count": 0,
            "elapsed_seconds": time.time() - started,
            "metric_path": str(metric_path),
            "training_audit_path": str(audit_path),
            "prediction_path": "",
            "status": "FAILED",
            "message": f"{type(exc).__name__}: {exc}",
        }]), status_path)
        print(f"FATAL: {type(exc).__name__}: {exc}", file=sys.stderr)
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    raise SystemExit(main())

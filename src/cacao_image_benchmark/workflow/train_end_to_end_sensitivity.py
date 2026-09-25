#!/usr/bin/env python3
"""Run five end-to-end folds for one task, architecture, split design, and seed."""
from __future__ import annotations

import argparse
import copy
import os
import platform
import random
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torchvision.models import (
    EfficientNet_B0_Weights,
    ResNet18_Weights,
    efficientnet_b0,
    resnet18,
)

from end_to_end_common import (
    ProjectPaths,
    aggregate_lineage_predictions,
    class_weight_vector,
    multiclass_metrics,
    read_tsv,
    seed_everything,
    sha256_file,
    worker_seed,
    write_tsv,
)

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True, type=Path)
    parser.add_argument("--big", required=True, type=Path)
    parser.add_argument("--array-index", required=True, type=int)
    parser.add_argument("--threads", type=int, default=16)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--weight-mode", choices=["default", "none"], default="default")
    parser.add_argument("--max-epochs-override", type=int, default=0)
    parser.add_argument("--batch-size-override", type=int, default=0)
    return parser.parse_args()


class ImageTableDataset(Dataset):
    def __init__(self, frame: pd.DataFrame, label_to_index: dict[str, int], transform) -> None:
        self.paths = frame["cache_path"].astype(str).tolist()
        self.labels = [label_to_index[label] for label in frame["label"].astype(str)]
        self.row_indices = frame.index.to_numpy(dtype=int)
        self.transform = transform

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, index: int):
        path = self.paths[index]
        with Image.open(path) as raw:
            image = raw.convert("RGB")
        return int(self.row_indices[index]), self.transform(image), int(self.labels[index])


def make_transforms() -> tuple[object, object]:
    train_transform = transforms.Compose(
        [
            transforms.RandomResizedCrop(
                224,
                scale=(0.80, 1.00),
                ratio=(0.85, 1.15),
                interpolation=transforms.InterpolationMode.BILINEAR,
                antialias=True,
            ),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ]
    )
    eval_transform = transforms.Compose(
        [
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ]
    )
    return train_transform, eval_transform


def build_model(architecture: str, n_classes: int, weight_mode: str):
    if architecture == "resnet18":
        weights = ResNet18_Weights.DEFAULT if weight_mode == "default" else None
        model = resnet18(weights=weights)
        in_features = model.fc.in_features
        model.fc = nn.Linear(in_features, n_classes)
        head_parameters = list(model.fc.parameters())
        head_ids = {id(parameter) for parameter in head_parameters}
    elif architecture == "efficientnet_b0":
        weights = EfficientNet_B0_Weights.DEFAULT if weight_mode == "default" else None
        model = efficientnet_b0(weights=weights)
        in_features = model.classifier[1].in_features
        model.classifier[1] = nn.Linear(in_features, n_classes)
        head_parameters = list(model.classifier[1].parameters())
        head_ids = {id(parameter) for parameter in head_parameters}
    else:
        raise ValueError(f"Unsupported architecture: {architecture}")
    backbone_parameters = [parameter for parameter in model.parameters() if id(parameter) not in head_ids]
    return model, backbone_parameters, head_parameters, str(weights)


def make_loader(
    frame: pd.DataFrame,
    label_to_index: dict[str, int],
    transform,
    batch_size: int,
    workers: int,
    shuffle: bool,
    seed: int,
) -> DataLoader:
    dataset = ImageTableDataset(frame, label_to_index, transform)
    generator = torch.Generator()
    generator.manual_seed(seed)

    def init_worker(worker_id: int) -> None:
        value = worker_seed(seed, worker_id)
        random.seed(value)
        np.random.seed(value)
        torch.manual_seed(value)

    kwargs: dict[str, object] = {
        "dataset": dataset,
        "batch_size": batch_size,
        "shuffle": shuffle,
        "num_workers": workers,
        "pin_memory": False,
        "drop_last": False,
        "generator": generator,
        "worker_init_fn": init_worker,
    }
    if workers > 0:
        kwargs.update({"persistent_workers": True, "prefetch_factor": 2})
    return DataLoader(**kwargs)


def predict(
    model: nn.Module,
    loader: DataLoader,
    full_frame: pd.DataFrame,
    labels: list[str],
) -> tuple[pd.DataFrame, dict[str, float], pd.DataFrame, pd.DataFrame]:
    model.eval()
    rows: list[dict[str, object]] = []
    with torch.inference_mode():
        for row_indices, images, targets in loader:
            images = images.contiguous(memory_format=torch.channels_last)
            logits = model(images)
            probabilities = torch.softmax(logits, dim=1).cpu().numpy()
            predicted = probabilities.argmax(axis=1)
            row_idx_values = row_indices.cpu().numpy().astype(int)
            target_values = targets.cpu().numpy().astype(int)
            for local_index, row_index in enumerate(row_idx_values):
                metadata = full_frame.loc[row_index]
                record: dict[str, object] = {
                    "sample_id": metadata["sample_id"],
                    "true_label": labels[target_values[local_index]],
                    "predicted_label": labels[predicted[local_index]],
                    "final_strict_lineage_id": metadata["final_strict_lineage_id"],
                    "final_split_block_id": metadata["final_split_block_id"],
                    "exact_component_id": metadata["exact_component_id"],
                    "source_dataset_id": metadata["source_dataset_id"],
                    "cache_path": metadata["cache_path"],
                }
                for label_index, label in enumerate(labels):
                    record[f"prob__{label}"] = float(probabilities[local_index, label_index])
                rows.append(record)
    predictions = pd.DataFrame(rows)
    prob_columns = [f"prob__{label}" for label in labels]
    metrics, per_class, confusion = multiclass_metrics(
        predictions["true_label"],
        predictions["predicted_label"],
        predictions[prob_columns].to_numpy(dtype=float),
        labels,
    )
    return predictions, metrics, per_class, confusion


def evaluate_validation(
    model: nn.Module,
    loader: DataLoader,
    full_frame: pd.DataFrame,
    labels: list[str],
) -> dict[str, float]:
    _, metrics, _, _ = predict(model, loader, full_frame, labels)
    return metrics


def overlap_count(left: pd.DataFrame, right: pd.DataFrame, column: str) -> int:
    return len(set(left[column]) & set(right[column]))


def main() -> int:
    args = parse_args()
    paths = ProjectPaths(args.root, args.big)
    config_path = paths.prepared / "stage3d_config_table.tsv"
    sample_path = paths.prepared / "stage3d_path_samples.tsv.gz"
    cache_path = paths.cache_manifest
    if not (paths.prepared / ".stage3d_prepared").is_file():
        raise RuntimeError("end-to-end sensitivity preparation marker is missing")
    for required in [config_path, sample_path, cache_path]:
        if not required.is_file():
            raise RuntimeError(f"Required end-to-end sensitivity input is missing: {required}")

    config_table = read_tsv(config_path)
    selected = config_table[config_table["array_index"].astype(int).eq(args.array_index)]
    if len(selected) != 1:
        raise RuntimeError(f"Expected one config for array index {args.array_index}, found {len(selected)}")
    config = selected.iloc[0]
    config_id = str(config["config_id"])
    output_dir = paths.run_results / config_id
    output_dir.mkdir(parents=True, exist_ok=True)
    status_path = output_dir / f"stage3d_{config_id}_status.tsv"
    started = time.time()

    try:
        os.environ["TORCH_HOME"] = str(paths.model_cache)
        torch.set_num_threads(max(1, args.threads - max(1, args.workers)))
        try:
            torch.set_num_interop_threads(1)
        except RuntimeError:
            pass
        try:
            torch.set_float32_matmul_precision("high")
        except Exception:
            pass

        task_name = str(config["task_name"])
        architecture = str(config["architecture"])
        design = str(config["evaluation_design"])
        repeat_index = int(config["repeat_index"])
        split_seed = int(config["split_seed"])
        labels = str(config["label_order"]).split("|")
        assignment_path = Path(str(config["assignment_path"]))
        if sha256_file(assignment_path) != str(config["assignment_sha256"]):
            raise RuntimeError(f"Assignment checksum changed before training: {assignment_path}")

        samples = read_tsv(sample_path)
        samples = samples[samples["task_name"].eq(task_name)].copy()
        cache_manifest = read_tsv(cache_path)[["sample_id", "cache_path"]]
        samples = samples.merge(cache_manifest, on="sample_id", how="left", validate="many_to_one")
        if samples["cache_path"].eq("").any() or samples["cache_path"].isna().any():
            raise RuntimeError("Missing cache paths in end-to-end sensitivity task sample table")
        missing_cache = [value for value in samples["cache_path"] if not Path(str(value)).is_file()]
        if missing_cache:
            raise RuntimeError(f"Cached images are missing, first examples: {missing_cache[:10]}")
        assignment = read_tsv(assignment_path)
        assignment["fold"] = assignment["fold"].astype(int)
        samples = samples.merge(
            assignment[["sample_id", "fold"]],
            on="sample_id",
            how="inner",
            validate="one_to_one",
        )
        if len(samples) != len(assignment):
            raise RuntimeError("end-to-end sensitivity sample/assignment merge changed row count")
        samples = samples.sort_values("sample_id").reset_index(drop=True)
        samples.index = np.arange(len(samples), dtype=int)
        label_to_index = {label: index for index, label in enumerate(labels)}
        if set(samples["label"]) != set(labels):
            raise RuntimeError(f"Task labels differ from frozen label order: {sorted(samples['label'].unique())}")

        max_epochs = args.max_epochs_override or int(config["max_epochs"])
        batch_size = args.batch_size_override or int(config["batch_size"])
        patience = int(config["early_stopping_patience"])
        min_delta = float(config["early_stopping_min_delta"])
        backbone_lr = float(config["backbone_learning_rate"])
        head_lr = float(config["head_learning_rate"])
        weight_decay = float(config["weight_decay"])
        train_transform, eval_transform = make_transforms()

        fold_prediction_frames: list[pd.DataFrame] = []
        fold_metric_frames: list[pd.DataFrame] = []
        fold_per_class_frames: list[pd.DataFrame] = []
        fold_confusion_frames: list[pd.DataFrame] = []
        fold_history_frames: list[pd.DataFrame] = []
        fold_audit_frames: list[pd.DataFrame] = []

        for test_fold in range(5):
            fold_prefix = output_dir / f"fold_{test_fold}"
            fold_marker = output_dir / f"fold_{test_fold}.complete"
            prediction_file = fold_prefix.with_name(f"fold_{test_fold}_predictions.tsv.gz")
            metrics_file = fold_prefix.with_name(f"fold_{test_fold}_metrics.tsv")
            per_class_file = fold_prefix.with_name(f"fold_{test_fold}_per_class.tsv")
            confusion_file = fold_prefix.with_name(f"fold_{test_fold}_confusion.tsv")
            history_file = fold_prefix.with_name(f"fold_{test_fold}_history.tsv")
            audit_file = fold_prefix.with_name(f"fold_{test_fold}_training_audit.tsv")
            required_fold_outputs = [
                prediction_file,
                metrics_file,
                per_class_file,
                confusion_file,
                history_file,
                audit_file,
            ]
            if fold_marker.is_file() and all(path.is_file() and path.stat().st_size > 0 for path in required_fold_outputs):
                print(f"config={config_id} fold={test_fold} reuse_completed_fold=YES", flush=True)
                fold_prediction_frames.append(read_tsv(prediction_file))
                fold_metric_frames.append(read_tsv(metrics_file))
                fold_per_class_frames.append(read_tsv(per_class_file))
                fold_confusion_frames.append(read_tsv(confusion_file))
                fold_history_frames.append(read_tsv(history_file))
                fold_audit_frames.append(read_tsv(audit_file))
                continue

            validation_fold = (test_fold + 1) % 5
            train_frame = samples[~samples["fold"].isin([test_fold, validation_fold])].copy()
            validation_frame = samples[samples["fold"].eq(validation_fold)].copy()
            test_frame = samples[samples["fold"].eq(test_fold)].copy()
            for split_name, frame in [
                ("train", train_frame),
                ("validation", validation_frame),
                ("test", test_frame),
            ]:
                if set(frame["label"]) != set(labels):
                    raise RuntimeError(f"{config_id} fold {test_fold} {split_name} lacks one or more labels")

            arch_offset = 0 if architecture == "resnet18" else 100_000
            fold_seed = split_seed + arch_offset + test_fold * 9_973
            seed_everything(fold_seed)
            train_loader = make_loader(
                train_frame,
                label_to_index,
                train_transform,
                batch_size,
                args.workers,
                True,
                fold_seed,
            )
            validation_loader = make_loader(
                validation_frame,
                label_to_index,
                eval_transform,
                batch_size,
                args.workers,
                False,
                fold_seed + 1,
            )
            test_loader = make_loader(
                test_frame,
                label_to_index,
                eval_transform,
                batch_size,
                args.workers,
                False,
                fold_seed + 2,
            )

            model, backbone_parameters, head_parameters, weights_name = build_model(
                architecture, len(labels), args.weight_mode
            )
            model = model.to("cpu", memory_format=torch.channels_last)
            class_weights = torch.tensor(
                class_weight_vector(train_frame["label"], labels), dtype=torch.float32
            )
            criterion = nn.CrossEntropyLoss(weight=class_weights)
            optimizer = torch.optim.AdamW(
                [
                    {"params": backbone_parameters, "lr": backbone_lr},
                    {"params": head_parameters, "lr": head_lr},
                ],
                weight_decay=weight_decay,
            )
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=max(1, max_epochs), eta_min=1e-6
            )

            best_state = None
            best_epoch = 0
            best_balanced_accuracy = -np.inf
            best_log_loss = np.inf
            stale_epochs = 0
            history_rows: list[dict[str, object]] = []
            fold_started = time.time()
            for epoch in range(1, max_epochs + 1):
                model.train()
                running_loss = 0.0
                n_seen = 0
                for _, images, targets in train_loader:
                    images = images.contiguous(memory_format=torch.channels_last)
                    optimizer.zero_grad(set_to_none=True)
                    logits = model(images)
                    loss = criterion(logits, targets)
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
                    optimizer.step()
                    running_loss += float(loss.detach()) * len(targets)
                    n_seen += len(targets)
                validation_metrics = evaluate_validation(
                    model, validation_loader, samples, labels
                )
                train_loss = running_loss / max(n_seen, 1)
                current_lr_backbone = float(optimizer.param_groups[0]["lr"])
                current_lr_head = float(optimizer.param_groups[1]["lr"])
                history_rows.append(
                    {
                        "config_id": config_id,
                        "task_name": task_name,
                        "architecture": architecture,
                        "evaluation_design": design,
                        "repeat_index": repeat_index,
                        "test_fold": test_fold,
                        "validation_fold": validation_fold,
                        "epoch": epoch,
                        "train_loss": train_loss,
                        "validation_balanced_accuracy": validation_metrics["balanced_accuracy"],
                        "validation_macro_f1": validation_metrics["macro_f1"],
                        "validation_log_loss": validation_metrics["log_loss"],
                        "backbone_learning_rate": current_lr_backbone,
                        "head_learning_rate": current_lr_head,
                    }
                )
                improved = (
                    validation_metrics["balanced_accuracy"]
                    > best_balanced_accuracy + min_delta
                ) or (
                    abs(validation_metrics["balanced_accuracy"] - best_balanced_accuracy)
                    <= min_delta
                    and validation_metrics["log_loss"] < best_log_loss
                )
                if improved:
                    best_balanced_accuracy = float(validation_metrics["balanced_accuracy"])
                    best_log_loss = float(validation_metrics["log_loss"])
                    best_epoch = epoch
                    best_state = {
                        key: value.detach().cpu().clone()
                        for key, value in model.state_dict().items()
                    }
                    stale_epochs = 0
                else:
                    stale_epochs += 1
                scheduler.step()
                print(
                    f"config={config_id} fold={test_fold} epoch={epoch}/{max_epochs} "
                    f"train_loss={train_loss:.5f} val_ba={validation_metrics['balanced_accuracy']:.5f} "
                    f"best_epoch={best_epoch} stale={stale_epochs}",
                    flush=True,
                )
                if stale_epochs >= patience:
                    break
            if best_state is None:
                raise RuntimeError(f"No best model state recorded for {config_id}/fold{test_fold}")
            model.load_state_dict(best_state)
            predictions, path_metrics, path_per_class, path_confusion = predict(
                model, test_loader, samples, labels
            )
            predictions["config_id"] = config_id
            predictions["task_name"] = task_name
            predictions["architecture"] = architecture
            predictions["evaluation_design"] = design
            predictions["repeat_index"] = repeat_index
            predictions["split_seed"] = split_seed
            predictions["test_fold"] = test_fold
            predictions["validation_fold"] = validation_fold

            prob_columns = [f"prob__{label}" for label in labels]
            lineage_predictions = aggregate_lineage_predictions(predictions, labels)
            lineage_metrics, lineage_per_class, lineage_confusion = multiclass_metrics(
                lineage_predictions["true_label"],
                lineage_predictions["predicted_label"],
                lineage_predictions[prob_columns].to_numpy(dtype=float),
                labels,
            )
            metric_rows = []
            for evaluation_unit, values in [
                ("PHYSICAL_PATH", path_metrics),
                ("STRICT_LINEAGE", lineage_metrics),
            ]:
                metric_rows.append(
                    {
                        "config_id": config_id,
                        "task_name": task_name,
                        "architecture": architecture,
                        "evaluation_design": design,
                        "repeat_index": repeat_index,
                        "split_seed": split_seed,
                        "test_fold": test_fold,
                        "validation_fold": validation_fold,
                        "evaluation_unit": evaluation_unit,
                        "best_epoch": best_epoch,
                        "best_validation_balanced_accuracy": best_balanced_accuracy,
                        "best_validation_log_loss": best_log_loss,
                        "fold_elapsed_seconds": time.time() - fold_started,
                        **values,
                    }
                )
            metrics_frame = pd.DataFrame(metric_rows)
            path_per_class["evaluation_unit"] = "PHYSICAL_PATH"
            lineage_per_class["evaluation_unit"] = "STRICT_LINEAGE"
            per_class_frame = pd.concat([path_per_class, lineage_per_class], ignore_index=True)
            per_class_frame.insert(0, "test_fold", test_fold)
            per_class_frame.insert(0, "repeat_index", repeat_index)
            per_class_frame.insert(0, "evaluation_design", design)
            per_class_frame.insert(0, "architecture", architecture)
            per_class_frame.insert(0, "task_name", task_name)
            per_class_frame.insert(0, "config_id", config_id)
            path_confusion["evaluation_unit"] = "PHYSICAL_PATH"
            lineage_confusion["evaluation_unit"] = "STRICT_LINEAGE"
            confusion_frame = pd.concat([path_confusion, lineage_confusion], ignore_index=True)
            confusion_frame.insert(0, "test_fold", test_fold)
            confusion_frame.insert(0, "repeat_index", repeat_index)
            confusion_frame.insert(0, "evaluation_design", design)
            confusion_frame.insert(0, "architecture", architecture)
            confusion_frame.insert(0, "task_name", task_name)
            confusion_frame.insert(0, "config_id", config_id)
            history_frame = pd.DataFrame(history_rows)
            audit_frame = pd.DataFrame(
                [
                    {
                        "config_id": config_id,
                        "task_name": task_name,
                        "architecture": architecture,
                        "evaluation_design": design,
                        "repeat_index": repeat_index,
                        "split_seed": split_seed,
                        "test_fold": test_fold,
                        "validation_fold": validation_fold,
                        "fold_seed": fold_seed,
                        "train_paths": len(train_frame),
                        "validation_paths": len(validation_frame),
                        "test_paths": len(test_frame),
                        "train_strict_lineages": train_frame["final_strict_lineage_id"].nunique(),
                        "validation_strict_lineages": validation_frame["final_strict_lineage_id"].nunique(),
                        "test_strict_lineages": test_frame["final_strict_lineage_id"].nunique(),
                        "train_test_exact_overlap": overlap_count(train_frame, test_frame, "exact_component_id"),
                        "train_test_strict_overlap": overlap_count(train_frame, test_frame, "final_strict_lineage_id"),
                        "train_test_scene_overlap": overlap_count(train_frame, test_frame, "final_split_block_id"),
                        "validation_test_scene_overlap": overlap_count(validation_frame, test_frame, "final_split_block_id"),
                        "class_weight_order": "|".join(labels),
                        "class_weights": "|".join(f"{value:.8g}" for value in class_weights.tolist()),
                        "weights": weights_name,
                        "weight_mode": args.weight_mode,
                        "max_epochs": max_epochs,
                        "epochs_completed": len(history_rows),
                        "early_stopping_patience": patience,
                        "batch_size": batch_size,
                        "workers": args.workers,
                        "torch_threads": torch.get_num_threads(),
                        "python": platform.python_version(),
                        "torch": torch.__version__,
                    }
                ]
            )
            write_tsv(predictions, prediction_file)
            write_tsv(metrics_frame, metrics_file)
            write_tsv(per_class_frame, per_class_file)
            write_tsv(confusion_frame, confusion_file)
            write_tsv(history_frame, history_file)
            write_tsv(audit_frame, audit_file)
            fold_marker.write_text("COMPLETED\n", encoding="utf-8")

            fold_prediction_frames.append(predictions)
            fold_metric_frames.append(metrics_frame)
            fold_per_class_frames.append(per_class_frame)
            fold_confusion_frames.append(confusion_frame)
            fold_history_frames.append(history_frame)
            fold_audit_frames.append(audit_frame)
            del model, best_state, optimizer, scheduler, train_loader, validation_loader, test_loader

        combined_predictions = pd.concat(fold_prediction_frames, ignore_index=True)
        if combined_predictions["sample_id"].nunique() != len(samples):
            raise RuntimeError(
                f"Out-of-fold predictions do not cover each sample exactly once: "
                f"unique={combined_predictions['sample_id'].nunique()} expected={len(samples)}"
            )
        if combined_predictions["sample_id"].duplicated().any():
            raise RuntimeError("A sample appears more than once in out-of-fold predictions")
        combined_metrics = pd.concat(fold_metric_frames, ignore_index=True)
        combined_per_class = pd.concat(fold_per_class_frames, ignore_index=True)
        combined_confusion = pd.concat(fold_confusion_frames, ignore_index=True)
        combined_history = pd.concat(fold_history_frames, ignore_index=True)
        combined_audit = pd.concat(fold_audit_frames, ignore_index=True)

        prediction_output = output_dir / f"stage3d_{config_id}_oof_predictions.tsv.gz"
        metric_output = output_dir / f"stage3d_{config_id}_fold_metrics.tsv"
        per_class_output = output_dir / f"stage3d_{config_id}_per_class.tsv"
        confusion_output = output_dir / f"stage3d_{config_id}_confusion.tsv"
        history_output = output_dir / f"stage3d_{config_id}_history.tsv"
        audit_output = output_dir / f"stage3d_{config_id}_training_audit.tsv"
        write_tsv(combined_predictions, prediction_output)
        write_tsv(combined_metrics, metric_output)
        write_tsv(combined_per_class, per_class_output)
        write_tsv(combined_confusion, confusion_output)
        write_tsv(combined_history, history_output)
        write_tsv(combined_audit, audit_output)

        status = pd.DataFrame(
            [
                {
                    "array_index": args.array_index,
                    "config_id": config_id,
                    "task_name": task_name,
                    "architecture": architecture,
                    "evaluation_design": design,
                    "repeat_index": repeat_index,
                    "split_seed": split_seed,
                    "folds_expected": 5,
                    "folds_completed": 5,
                    "oof_prediction_rows": len(combined_predictions),
                    "elapsed_seconds": time.time() - started,
                    "prediction_path": str(prediction_output),
                    "metric_path": str(metric_output),
                    "status": "COMPLETED",
                    "message": "",
                }
            ]
        )
        write_tsv(status, status_path)
        print(status.to_string(index=False))
        return 0
    except Exception as exc:
        failure = pd.DataFrame(
            [
                {
                    "array_index": args.array_index,
                    "config_id": str(selected.iloc[0]["config_id"]) if len(selected) == 1 else "",
                    "task_name": str(selected.iloc[0]["task_name"]) if len(selected) == 1 else "",
                    "architecture": str(selected.iloc[0]["architecture"]) if len(selected) == 1 else "",
                    "evaluation_design": str(selected.iloc[0]["evaluation_design"]) if len(selected) == 1 else "",
                    "repeat_index": str(selected.iloc[0]["repeat_index"]) if len(selected) == 1 else "",
                    "split_seed": str(selected.iloc[0]["split_seed"]) if len(selected) == 1 else "",
                    "folds_expected": 5,
                    "folds_completed": len(list(output_dir.glob("fold_*.complete"))) if output_dir.exists() else 0,
                    "oof_prediction_rows": 0,
                    "elapsed_seconds": time.time() - started,
                    "prediction_path": "",
                    "metric_path": "",
                    "status": "FAILED",
                    "message": f"{type(exc).__name__}: {exc}",
                }
            ]
        )
        write_tsv(failure, status_path)
        print(f"FATAL: {type(exc).__name__}: {exc}", file=sys.stderr, flush=True)
        raise


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""Prepare frozen split assignments, image-cache chunks, and end-to-end run definitions."""
from __future__ import annotations

import argparse
import shutil
from pathlib import Path

import numpy as np
import pandas as pd

from end_to_end_common import (
    ARCHITECTURES,
    EVALUATION_DESIGNS,
    REPEAT_SEEDS,
    TASK_LABEL_ORDERS,
    ProjectPaths,
    read_tsv,
    sha256_file,
    stable_hash,
    write_tsv,
)

EXPECTED_TASK_PATHS = {
    "cacao_coarse_three_class": 12541,
    "cocoamonilia_four_stage": 1878,
}
EXPECTED_UNIQUE_CACHE_IMAGES = 12541


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True, type=Path)
    parser.add_argument("--big", required=True, type=Path)
    parser.add_argument("--expected-configs", type=int, default=40)
    parser.add_argument("--cache-tasks", type=int, default=32)
    parser.add_argument("--expected-unique-images", type=int, default=EXPECTED_UNIQUE_CACHE_IMAGES)
    parser.add_argument("--skip-path-check", action="store_true")
    parser.add_argument("--test-mode", action="store_true", help="Permit reduced counts for package smoke tests")
    return parser.parse_args()


def require_columns(frame: pd.DataFrame, columns: list[str], name: str) -> None:
    missing = [column for column in columns if column not in frame.columns]
    if missing:
        raise RuntimeError(f"{name} is missing columns: {missing}")


def crossing_count(frame: pd.DataFrame, column: str) -> int:
    return int((frame.groupby(column)["fold"].nunique() > 1).sum())


def copy_assignment(
    source_path: Path,
    expected_sha256: str,
    destination: Path,
) -> str:
    if not source_path.is_file():
        raise RuntimeError(f"Repeated assignment is missing: {source_path}")
    observed = sha256_file(source_path)
    if expected_sha256 and observed != expected_sha256:
        raise RuntimeError(
            f"Assignment checksum mismatch before end-to-end sensitivity: {source_path}\n"
            f"expected={expected_sha256}\nobserved={observed}"
        )
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source_path, destination)
    copied = sha256_file(destination)
    if copied != observed:
        raise RuntimeError(f"Copied assignment checksum changed: {destination}")
    return copied


def main() -> int:
    args = parse_args()
    paths = ProjectPaths(args.root, args.big)
    for directory in [
        paths.prepared,
        paths.prepared / "assignments",
        paths.cache_chunks,
        paths.cache_status,
        paths.run_results,
        paths.aggregate,
        paths.figures,
        paths.cache_dir,
    ]:
        directory.mkdir(parents=True, exist_ok=True)

    sample_path = paths.stage3b_prepared / "stage3b_path_samples.tsv.gz"
    index_path = paths.stage3b_prepared / "stage3b_repeated_assignment_index.tsv"
    seed_path = paths.stage3b_prepared / "stage3b_repeat_seeds.tsv"
    claim_path = paths.stage3b_results / "stage3b_claim_status.tsv"
    for required in [sample_path, index_path, seed_path, claim_path]:
        if not required.is_file():
            raise RuntimeError(f"Required design confirmation input is missing: {required}")

    claim = read_tsv(claim_path)
    if claim.empty or claim.iloc[0].get("all_model_configs_completed", "") != "YES":
        raise RuntimeError("design confirmation claim status does not confirm complete model execution")

    samples = read_tsv(sample_path)
    require_columns(
        samples,
        [
            "task_name",
            "sample_id",
            "label",
            "absolute_path",
            "exact_component_id",
            "final_strict_lineage_id",
            "final_split_block_id",
            "source_dataset_id",
            "source_archive",
        ],
        "design confirmation path sample table",
    )
    samples = samples[samples["task_name"].isin(TASK_LABEL_ORDERS)].copy()
    for task_name, label_order in TASK_LABEL_ORDERS.items():
        task = samples[samples["task_name"].eq(task_name)]
        expected = EXPECTED_TASK_PATHS.get(task_name)
        if not args.test_mode and expected is not None and len(task) != expected:
            raise RuntimeError(f"Unexpected {task_name} path count: {len(task)} != {expected}")
        if set(task["label"]) != set(label_order):
            raise RuntimeError(f"Unexpected label set for {task_name}: {sorted(task['label'].unique())}")
        if task["sample_id"].duplicated().any():
            raise RuntimeError(f"Duplicate sample IDs within {task_name}")

    unique_images = (
        samples.sort_values(["sample_id", "task_name"])
        .drop_duplicates("sample_id")
        .copy()
        .reset_index(drop=True)
    )
    if len(unique_images) != args.expected_unique_images:
        raise RuntimeError(
            f"Unexpected unique image count: {len(unique_images)} != {args.expected_unique_images}"
        )
    if unique_images["absolute_path"].duplicated().any():
        duplicate = unique_images[unique_images["absolute_path"].duplicated(keep=False)].head()
        raise RuntimeError(f"Multiple sample IDs point to the same absolute path:\n{duplicate}")
    if not args.skip_path_check:
        missing = [value for value in unique_images["absolute_path"] if not Path(str(value)).is_file()]
        if missing:
            raise RuntimeError(f"Missing source images before end-to-end sensitivity, first examples: {missing[:10]}")

    unique_images["cache_filename"] = unique_images["sample_id"].astype(str) + ".jpg"
    unique_images["cache_path"] = unique_images["cache_filename"].map(lambda name: str(paths.cache_dir / name))
    cache_columns = [
        "sample_id",
        "absolute_path",
        "cache_path",
        "source_dataset_id",
        "source_archive",
        "exact_component_id",
        "final_strict_lineage_id",
        "final_split_block_id",
    ]
    cache_manifest = unique_images[cache_columns].sort_values("sample_id").reset_index(drop=True)
    write_tsv(cache_manifest, paths.cache_manifest)

    chunk_ids = np.array_split(np.arange(len(cache_manifest)), args.cache_tasks)
    chunk_rows = []
    for task_index, indices in enumerate(chunk_ids, start=1):
        chunk = cache_manifest.iloc[indices].copy()
        chunk_path = paths.cache_chunks / f"stage3d_cache_chunk_{task_index:03d}.tsv"
        write_tsv(chunk, chunk_path)
        chunk_rows.append(
            {
                "cache_task_index": task_index,
                "rows": len(chunk),
                "chunk_path": str(chunk_path),
                "chunk_sha256": sha256_file(chunk_path),
            }
        )
    write_tsv(pd.DataFrame(chunk_rows), paths.prepared / "stage3d_cache_chunk_index.tsv")

    assignment_index = read_tsv(index_path)
    require_columns(
        assignment_index,
        [
            "task_name",
            "evaluation_design",
            "repeat_index",
            "split_seed",
            "assignment_path",
            "assignment_sha256",
            "assignment_rows",
        ],
        "design confirmation repeated assignment index",
    )
    seed_table = read_tsv(seed_path)
    observed_seeds = seed_table.head(5)["split_seed"].astype(int).tolist()
    if observed_seeds != REPEAT_SEEDS:
        raise RuntimeError(f"design confirmation first five repeat seeds changed: {observed_seeds}")

    configs: list[dict[str, object]] = []
    audits: list[dict[str, object]] = []
    copied_index: list[dict[str, object]] = []
    config_number = 0
    for task_name in TASK_LABEL_ORDERS:
        task_samples = samples[samples["task_name"].eq(task_name)].copy()
        sample_ids = set(task_samples["sample_id"])
        labels = set(TASK_LABEL_ORDERS[task_name])
        for design in EVALUATION_DESIGNS:
            for repeat_index, split_seed in enumerate(REPEAT_SEEDS, start=1):
                match = assignment_index[
                    assignment_index["task_name"].eq(task_name)
                    & assignment_index["evaluation_design"].eq(design)
                    & assignment_index["repeat_index"].astype(int).eq(repeat_index)
                ]
                if len(match) != 1:
                    raise RuntimeError(
                        f"Expected one design confirmation assignment for {task_name}/{design}/R{repeat_index:02d}, found {len(match)}"
                    )
                record = match.iloc[0]
                if int(record["split_seed"]) != split_seed:
                    raise RuntimeError("design confirmation assignment seed mismatch")
                destination = (
                    paths.prepared
                    / "assignments"
                    / f"{task_name}__{design}__R{repeat_index:02d}.tsv.gz"
                )
                copied_sha = copy_assignment(
                    Path(record["assignment_path"]),
                    str(record["assignment_sha256"]),
                    destination,
                )
                assigned = read_tsv(destination)
                require_columns(
                    assigned,
                    [
                        "sample_id",
                        "label",
                        "fold",
                        "exact_component_id",
                        "final_strict_lineage_id",
                        "final_split_block_id",
                    ],
                    "Copied end-to-end sensitivity assignment",
                )
                assigned["fold"] = assigned["fold"].astype(int)
                if set(assigned["sample_id"]) != sample_ids:
                    raise RuntimeError(f"Assignment sample set mismatch for {task_name}/{design}/R{repeat_index}")
                if set(assigned["label"]) != labels or set(assigned["fold"]) != set(range(5)):
                    raise RuntimeError(f"Assignment labels/folds are incomplete for {task_name}/{design}/R{repeat_index}")
                all_labels = all(
                    set(assigned.loc[assigned["fold"].eq(fold), "label"]) == labels
                    for fold in range(5)
                )
                if not all_labels:
                    raise RuntimeError(f"A test fold lacks a label for {task_name}/{design}/R{repeat_index}")
                scene_crossings = crossing_count(assigned, "final_split_block_id")
                if design == "VERIFIED_SCENE_BLOCK_5FOLD" and scene_crossings != 0:
                    raise RuntimeError(
                        f"Scene block crossings detected in scene-safe assignment: {scene_crossings}"
                    )
                audits.append(
                    {
                        "task_name": task_name,
                        "evaluation_design": design,
                        "repeat_index": repeat_index,
                        "split_seed": split_seed,
                        "rows": len(assigned),
                        "min_fold_size": int(assigned.groupby("fold").size().min()),
                        "max_fold_size": int(assigned.groupby("fold").size().max()),
                        "exact_component_crossings": crossing_count(assigned, "exact_component_id"),
                        "strict_lineage_crossings": crossing_count(assigned, "final_strict_lineage_id"),
                        "scene_block_crossings": scene_crossings,
                        "all_labels_present_in_every_fold": "YES" if all_labels else "NO",
                    }
                )
                copied_index.append(
                    {
                        "task_name": task_name,
                        "evaluation_design": design,
                        "repeat_index": repeat_index,
                        "split_seed": split_seed,
                        "assignment_path": str(destination),
                        "assignment_sha256": copied_sha,
                        "assignment_rows": len(assigned),
                    }
                )
                for architecture in ARCHITECTURES:
                    config_number += 1
                    configs.append(
                        {
                            "array_index": config_number,
                            "config_id": f"S3D{config_number:03d}",
                            "task_name": task_name,
                            "architecture": architecture,
                            "evaluation_design": design,
                            "repeat_index": repeat_index,
                            "split_seed": split_seed,
                            "assignment_path": str(destination),
                            "assignment_sha256": copied_sha,
                            "label_order": "|".join(TASK_LABEL_ORDERS[task_name]),
                            "n_folds": 5,
                            "validation_fold_rule": "(test_fold + 1) mod 5",
                            "input_short_side": 256,
                            "input_crop_size": 224,
                            "max_epochs": 12,
                            "early_stopping_patience": 3,
                            "early_stopping_min_delta": 0.001,
                            "batch_size": 64,
                            "backbone_learning_rate": 0.0001,
                            "head_learning_rate": 0.0005,
                            "weight_decay": 0.0001,
                            "class_balanced_loss": "YES",
                            "imagenet_pretrained": "YES",
                            "all_layers_trainable": "YES",
                        }
                    )

    config_table = pd.DataFrame(configs)
    if len(config_table) != args.expected_configs:
        raise RuntimeError(f"Expected {args.expected_configs} end-to-end sensitivity configs, generated {len(config_table)}")
    if config_table.duplicated(
        ["task_name", "architecture", "evaluation_design", "repeat_index"]
    ).any():
        raise RuntimeError("Duplicate end-to-end sensitivity configuration definitions")
    write_tsv(config_table, paths.prepared / "stage3d_config_table.tsv")
    write_tsv(pd.DataFrame(copied_index), paths.prepared / "stage3d_assignment_index.tsv")
    write_tsv(pd.DataFrame(audits), paths.prepared / "stage3d_assignment_integrity_audit.tsv")
    write_tsv(samples.sort_values(["task_name", "sample_id"]), paths.prepared / "stage3d_path_samples.tsv.gz")

    provenance = pd.DataFrame(
        [
            {
                "stage3b_path_samples": str(sample_path),
                "stage3b_path_samples_sha256": sha256_file(sample_path),
                "stage3b_assignment_index": str(index_path),
                "stage3b_assignment_index_sha256": sha256_file(index_path),
                "stage3d_config_rows": len(config_table),
                "stage3d_cache_images": len(cache_manifest),
                "stage3d_cache_tasks": args.cache_tasks,
                "first_five_repeat_seeds": "|".join(map(str, REPEAT_SEEDS)),
                "preprocessing": "EXIF transpose; RGB; resize shorter side to 256; JPEG quality 92",
                "cache_path_hash_example": stable_hash(str(cache_manifest.iloc[0]["sample_id"])),
            }
        ]
    )
    write_tsv(provenance, paths.prepared / "stage3d_prepare_provenance.tsv")
    summary = pd.DataFrame(
        [
            {"metric": "model_run_configs", "value": len(config_table)},
            {"metric": "fold_fits_expected", "value": len(config_table) * 5},
            {"metric": "cache_images", "value": len(cache_manifest)},
            {"metric": "cache_tasks", "value": args.cache_tasks},
            {"metric": "tasks", "value": len(TASK_LABEL_ORDERS)},
            {"metric": "architectures", "value": len(ARCHITECTURES)},
            {"metric": "evaluation_designs", "value": len(EVALUATION_DESIGNS)},
            {"metric": "repeat_seeds", "value": len(REPEAT_SEEDS)},
        ]
    )
    write_tsv(summary, paths.prepared / "stage3d_prepare_summary.tsv")
    (paths.prepared / ".stage3d_prepared").write_text("READY\n", encoding="utf-8")
    print(summary.to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

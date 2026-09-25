#!/usr/bin/env python3
"""Prepare matched-removal nulls and visual-shortcut designs for matched-removal and shortcut controls."""
from __future__ import annotations

import argparse
import hashlib
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import Bounds, LinearConstraint, milp
from sklearn.model_selection import StratifiedGroupKFold

from control_analysis_common import (
    FEATURE_SETS,
    REPEAT_SEEDS,
    SOURCE_HOLDOUTS,
    TASK_LABEL_ORDERS,
    ProjectPaths,
    config_seed,
    json_dump,
    read_tsv,
    sha256_file,
    write_tsv,
)

NULL_PLAN = [
    {
        "scenario_id": "coco_public_test",
        "task_name": "cocoamonilia_four_stage",
        "heldout_source": "",
        "null_design": "SOURCE_CLASS_BLOCK_MATCHED",
        "replicates": 500,
    },
    {
        "scenario_id": "roboflow_source_holdout",
        "task_name": "cacao_coarse_three_class",
        "heldout_source": "fig_roboflow_mixed",
        "null_design": "SOURCE_CLASS_BLOCK_MATCHED",
        "replicates": 300,
    },
    {
        "scenario_id": "roboflow_source_holdout",
        "task_name": "cacao_coarse_three_class",
        "heldout_source": "fig_roboflow_mixed",
        "null_design": "CLASS_BLOCK_MATCHED",
        "replicates": 300,
    },
    {
        "scenario_id": "spanish_source_holdout",
        "task_name": "cacao_coarse_three_class",
        "heldout_source": "fig_spanish_yolov4",
        "null_design": "SOURCE_CLASS_BLOCK_MATCHED",
        "replicates": 300,
    },
]
EXPECTED_NULL_CONFIGS = sum(item["replicates"] for item in NULL_PLAN) + 6
EXPECTED_SHORTCUT_CONFIGS = 168
SIZE_BINS = ["1", "2", "3_4", "5_plus"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True, type=Path)
    parser.add_argument("--big", required=True, type=Path)
    parser.add_argument("--expected-null-configs", type=int, default=EXPECTED_NULL_CONFIGS)
    parser.add_argument("--expected-shortcut-configs", type=int, default=EXPECTED_SHORTCUT_CONFIGS)
    return parser.parse_args()


def size_bin(values: pd.Series) -> pd.Series:
    numeric = pd.to_numeric(values, errors="raise")
    return pd.cut(
        numeric,
        bins=[0, 1, 2, 4, np.inf],
        labels=SIZE_BINS,
        include_lowest=True,
    ).astype(str)


def make_scenario(samples: pd.DataFrame, plan: dict[str, object]) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    task = samples[samples["task_name"].eq(plan["task_name"])].copy()
    if plan["scenario_id"] == "coco_public_test":
        test = task[task["corrected_split"].eq("test")].copy()
        naive_train = task[task["corrected_split"].isin(["train", "validation"])].copy()
    else:
        source = str(plan["heldout_source"])
        test = task[task["source_dataset_id"].eq(source)].copy()
        naive_train = task[~task["source_dataset_id"].eq(source)].copy()
    test_blocks = set(test["final_split_block_id"])
    contaminated = naive_train[naive_train["final_split_block_id"].isin(test_blocks)].copy()
    scene_safe = naive_train[~naive_train["final_split_block_id"].isin(test_blocks)].copy()
    if test.empty or naive_train.empty:
        raise RuntimeError(f"Empty train/test scenario: {plan['scenario_id']}")
    return task, naive_train, test, contaminated, scene_safe


def block_table(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.empty:
        return pd.DataFrame(columns=[
            "block_id", "source_n", "label_n", "source", "label", "paths", "strict", "size_bin"
        ])
    grouped = frame.groupby("final_split_block_id", sort=True)
    table = grouped.agg(
        source_n=("source_dataset_id", "nunique"),
        label_n=("label", "nunique"),
        source=("source_dataset_id", "first"),
        label=("label", "first"),
        paths=("sample_id", "size"),
        strict=("final_strict_lineage_id", "nunique"),
    ).reset_index().rename(columns={"final_split_block_id": "block_id"})
    table["size_bin"] = size_bin(table["paths"])
    return table


def solve_type_plan(
    contaminated: pd.DataFrame,
    clean: pd.DataFrame,
    null_design: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    source_match = null_design == "SOURCE_CLASS_BLOCK_MATCHED"
    candidates = block_table(clean)
    candidates = candidates[candidates["source_n"].eq(1) & candidates["label_n"].eq(1)].copy()
    if candidates.empty:
        raise RuntimeError("No pure candidate split blocks available for matched removal")

    type_columns = (["source", "label"] if source_match else ["label"]) + [
        "paths", "strict", "size_bin"
    ]
    types = candidates.groupby(type_columns, dropna=False).size().reset_index(name="available")
    if source_match:
        target_paths = contaminated.groupby(["source_dataset_id", "label"]).size()
        target_keys = ["source", "label"]
    else:
        target_paths = contaminated.groupby(["label"]).size()
        target_keys = ["label"]

    target_blocks = block_table(contaminated)
    target_bins = target_blocks["size_bin"].value_counts().reindex(SIZE_BINS, fill_value=0)
    target_block_n = int(len(target_blocks))
    target_strict_n = int(contaminated["final_strict_lineage_id"].nunique())

    n_types = len(types)
    objective: list[float] = [0.0] * n_types
    lower: list[float] = [0.0] * n_types
    upper: list[float] = types["available"].astype(float).tolist()
    integrality: list[int] = [1] * n_types
    slack_index: dict[tuple[str, str, str], int] = {}

    def add_slacks(kind: str, keys: list[str], weight: float) -> None:
        for key in keys:
            for sign in ["positive", "negative"]:
                slack_index[(kind, key, sign)] = len(objective)
                objective.append(weight)
                lower.append(0.0)
                upper.append(np.inf)
                integrality.append(1)

    add_slacks("bin", SIZE_BINS, 100.0)
    add_slacks("block", ["all"], 200.0)
    add_slacks("strict", ["all"], 20.0)

    rows: list[np.ndarray] = []
    targets: list[float] = []
    n_variables = len(objective)

    for key, target in target_paths.items():
        values = key if isinstance(key, tuple) else (key,)
        mask = np.ones(n_types, dtype=bool)
        for column, value in zip(target_keys, values):
            mask &= types[column].eq(value).to_numpy()
        if not np.any(mask):
            raise RuntimeError(f"No candidates for matched stratum {values}")
        row = np.zeros(n_variables, dtype=float)
        row[:n_types][mask] = types.loc[mask, "paths"].to_numpy(float)
        rows.append(row)
        targets.append(float(target))

    for category in SIZE_BINS:
        row = np.zeros(n_variables, dtype=float)
        row[:n_types][types["size_bin"].eq(category).to_numpy()] = 1.0
        row[slack_index[("bin", category, "positive")]] = -1.0
        row[slack_index[("bin", category, "negative")]] = 1.0
        rows.append(row)
        targets.append(float(target_bins[category]))

    row = np.zeros(n_variables, dtype=float)
    row[:n_types] = 1.0
    row[slack_index[("block", "all", "positive")]] = -1.0
    row[slack_index[("block", "all", "negative")]] = 1.0
    rows.append(row)
    targets.append(float(target_block_n))

    row = np.zeros(n_variables, dtype=float)
    row[:n_types] = types["strict"].to_numpy(float)
    row[slack_index[("strict", "all", "positive")]] = -1.0
    row[slack_index[("strict", "all", "negative")]] = 1.0
    rows.append(row)
    targets.append(float(target_strict_n))

    matrix = np.vstack(rows)
    result = milp(
        c=np.asarray(objective, dtype=float),
        integrality=np.asarray(integrality, dtype=int),
        bounds=Bounds(np.asarray(lower, float), np.asarray(upper, float)),
        constraints=LinearConstraint(matrix, np.asarray(targets), np.asarray(targets)),
        options={"time_limit": 120.0},
    )
    if not result.success or result.x is None:
        raise RuntimeError(f"Matched-removal MILP failed: {result.message}")

    counts = np.rint(result.x[:n_types]).astype(int)
    selected_types = types[counts > 0].copy()
    selected_types["n_select"] = counts[counts > 0]
    selected_types.insert(0, "null_design", null_design)

    matched_paths = int((selected_types["paths"] * selected_types["n_select"]).sum())
    matched_blocks = int(selected_types["n_select"].sum())
    matched_strict = int((selected_types["strict"] * selected_types["n_select"]).sum())
    audit = pd.DataFrame([{
        "null_design": null_design,
        "source_matching": "YES" if source_match else "NO",
        "target_paths": len(contaminated),
        "matched_paths": matched_paths,
        "target_blocks": target_block_n,
        "matched_blocks": matched_blocks,
        "target_strict_lineages": target_strict_n,
        "matched_strict_lineages": matched_strict,
        "milp_objective": float(result.fun),
        "milp_success": "YES",
        **{f"target_bin_{category}": int(target_bins[category]) for category in SIZE_BINS},
        **{
            f"matched_bin_{category}": int(
                selected_types.loc[selected_types["size_bin"].eq(category), "n_select"].sum()
            )
            for category in SIZE_BINS
        },
    }])
    return selected_types, audit


def sample_removal_sets(
    clean: pd.DataFrame,
    selected_types: pd.DataFrame,
    scenario_id: str,
    null_design: str,
    replicates: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    source_match = null_design == "SOURCE_CLASS_BLOCK_MATCHED"
    candidates = block_table(clean)
    candidates = candidates[candidates["source_n"].eq(1) & candidates["label_n"].eq(1)].copy()
    type_columns = (["source", "label"] if source_match else ["label"]) + [
        "paths", "strict", "size_bin"
    ]
    rng = np.random.default_rng(config_seed(scenario_id, null_design))
    removal_rows: list[dict[str, object]] = []
    audit_rows: list[dict[str, object]] = []
    observed_hashes: set[str] = set()
    attempts = 0
    replicate = 1
    while replicate <= replicates:
        attempts += 1
        if attempts > replicates * 50:
            raise RuntimeError(f"Could not generate {replicates} unique matched-removal sets")
        chosen_ids: list[str] = []
        for type_row in selected_types.itertuples(index=False):
            mask = np.ones(len(candidates), dtype=bool)
            for column in type_columns:
                mask &= candidates[column].eq(getattr(type_row, column)).to_numpy()
            pool = candidates.loc[mask, "block_id"].to_numpy(dtype=object)
            n_select = int(type_row.n_select)
            if len(pool) < n_select:
                raise RuntimeError(f"Candidate type availability changed for {scenario_id}/{null_design}")
            chosen = rng.choice(pool, size=n_select, replace=False)
            chosen_ids.extend(str(value) for value in chosen)
        chosen_ids = sorted(set(chosen_ids))
        digest = hashlib.sha256("|".join(chosen_ids).encode()).hexdigest()
        if digest in observed_hashes:
            continue
        observed_hashes.add(digest)
        removed = clean[clean["final_split_block_id"].isin(chosen_ids)].copy()
        for block_id in chosen_ids:
            removal_rows.append({
                "scenario_id": scenario_id,
                "null_design": null_design,
                "replicate": replicate,
                "block_id": block_id,
            })
        audit_rows.append({
            "scenario_id": scenario_id,
            "null_design": null_design,
            "replicate": replicate,
            "removal_set_sha256": digest,
            "removed_paths": len(removed),
            "removed_blocks": removed["final_split_block_id"].nunique(),
            "removed_strict_lineages": removed["final_strict_lineage_id"].nunique(),
            "removed_class_counts": "|".join(
                f"{label}:{int((removed['label'] == label).sum())}"
                for label in sorted(removed["label"].unique())
            ),
            "removed_source_counts": "|".join(
                f"{source}:{int((removed['source_dataset_id'] == source).sum())}"
                for source in sorted(removed["source_dataset_id"].unique())
            ),
        })
        replicate += 1
    return pd.DataFrame(removal_rows), pd.DataFrame(audit_rows)


def build_shortcut_assignments(samples: pd.DataFrame, output_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    output_dir.mkdir(parents=True, exist_ok=True)
    index_rows: list[dict[str, object]] = []
    audit_rows: list[dict[str, object]] = []
    for task_name in ["cacao_coarse_three_class", "cocoamonilia_four_stage"]:
        task = samples[samples["task_name"].eq(task_name)].sort_values("sample_id").reset_index(drop=True)
        labels = set(TASK_LABEL_ORDERS[task_name])
        for repeat_index, seed in enumerate(REPEAT_SEEDS, start=1):
            splitter = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=seed)
            folds = np.full(len(task), -1, dtype=int)
            for fold, (_, test_idx) in enumerate(
                splitter.split(
                    np.zeros((len(task), 1)),
                    task["label"].to_numpy(),
                    groups=task["final_split_block_id"].to_numpy(),
                )
            ):
                folds[test_idx] = fold
            if np.any(folds < 0):
                raise RuntimeError("Incomplete visual-shortcut assignment")
            assignment = task[[
                "sample_id", "label", "final_strict_lineage_id", "final_split_block_id"
            ]].copy()
            assignment["fold"] = folds
            crossings = int((assignment.groupby("final_split_block_id")["fold"].nunique() > 1).sum())
            label_ok = all(set(assignment.loc[assignment["fold"].eq(fold), "label"]) == labels for fold in range(5))
            if crossings or not label_ok:
                raise RuntimeError(f"Shortcut split integrity failure {task_name}/R{repeat_index}")
            path = output_dir / f"{task_name}__VERIFIED_SCENE_BLOCK_5FOLD__R{repeat_index:02d}.tsv.gz"
            write_tsv(assignment, path)
            index_rows.append({
                "analysis_family": "REPEATED_SCENE_CV",
                "task_name": task_name,
                "evaluation_design": "VERIFIED_SCENE_BLOCK_5FOLD",
                "repeat_index": repeat_index,
                "split_seed": seed,
                "heldout_source": "",
                "assignment_path": str(path),
                "assignment_sha256": sha256_file(path),
            })
            audit_rows.append({
                "assignment_path": str(path),
                "rows": len(assignment),
                "split_block_crossings": crossings,
                "all_labels_in_every_fold": "YES" if label_ok else "NO",
            })

    coco = samples[samples["task_name"].eq("cocoamonilia_four_stage")].copy()
    test = coco[coco["corrected_split"].eq("test")].copy()
    train = coco[coco["corrected_split"].isin(["train", "validation"])].copy()
    test_blocks = set(test["final_split_block_id"])
    train = train[~train["final_split_block_id"].isin(test_blocks)].copy()
    assignment = pd.concat([
        train[["sample_id", "label"]].assign(role="TRAIN"),
        test[["sample_id", "label"]].assign(role="TEST"),
    ], ignore_index=True)
    path = output_dir / "cocoamonilia_four_stage__COCO_TEST_SCENE_SAFE.tsv.gz"
    write_tsv(assignment, path)
    index_rows.append({
        "analysis_family": "COCO_FIXED_TEST",
        "task_name": "cocoamonilia_four_stage",
        "evaluation_design": "COCO_TEST_SCENE_SAFE",
        "repeat_index": 0,
        "split_seed": 0,
        "heldout_source": "",
        "assignment_path": str(path),
        "assignment_sha256": sha256_file(path),
    })
    audit_rows.append({
        "assignment_path": str(path),
        "rows": len(assignment),
        "split_block_crossings": len(set(train["final_split_block_id"]) & set(test["final_split_block_id"])),
        "all_labels_in_every_fold": "NA",
    })

    coarse = samples[samples["task_name"].eq("cacao_coarse_three_class")].copy()
    for source in SOURCE_HOLDOUTS:
        test = coarse[coarse["source_dataset_id"].eq(source)].copy()
        train = coarse[~coarse["source_dataset_id"].eq(source)].copy()
        test_blocks = set(test["final_split_block_id"])
        train = train[~train["final_split_block_id"].isin(test_blocks)].copy()
        assignment = pd.concat([
            train[["sample_id", "label"]].assign(role="TRAIN"),
            test[["sample_id", "label"]].assign(role="TEST"),
        ], ignore_index=True)
        path = output_dir / f"{source}__SOURCE_HOLDOUT_SCENE_SAFE.tsv.gz"
        write_tsv(assignment, path)
        index_rows.append({
            "analysis_family": "SOURCE_HOLDOUT",
            "task_name": "cacao_coarse_three_class",
            "evaluation_design": "SOURCE_HOLDOUT_SCENE_SAFE",
            "repeat_index": 0,
            "split_seed": 0,
            "heldout_source": source,
            "assignment_path": str(path),
            "assignment_sha256": sha256_file(path),
        })
        audit_rows.append({
            "assignment_path": str(path),
            "rows": len(assignment),
            "split_block_crossings": len(set(train["final_split_block_id"]) & set(test["final_split_block_id"])),
            "all_labels_in_every_fold": "NA",
        })
    return pd.DataFrame(index_rows), pd.DataFrame(audit_rows)


def build_shortcut_config(index: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    counter = 1
    for assignment in index.itertuples(index=False):
        for feature_set in FEATURE_SETS:
            rows.append({
                "array_index": counter,
                "config_id": f"S3C_VIS_{counter:04d}",
                "analysis_family": assignment.analysis_family,
                "task_name": assignment.task_name,
                "evaluation_design": assignment.evaluation_design,
                "feature_set": feature_set,
                "repeat_index": assignment.repeat_index,
                "split_seed": assignment.split_seed,
                "heldout_source": assignment.heldout_source,
                "assignment_path": assignment.assignment_path,
            })
            counter += 1
    return pd.DataFrame(rows)


def main() -> int:
    args = parse_args()
    paths = ProjectPaths(args.root, args.big)
    paths.prepared.mkdir(parents=True, exist_ok=True)
    (paths.prepared / "shortcut_assignments").mkdir(parents=True, exist_ok=True)

    required = [
        paths.stage3b_prepared / "stage3b_path_samples.tsv.gz",
        paths.stage3b / "stage3b_claim_status.tsv",
        paths.stage2c_meta / "stage2c_embedding_manifest.tsv",
        paths.stage2c_matrix_dir / "stage2c_efficientnet_b0_embeddings.npy",
    ]
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise RuntimeError(f"Missing matched-removal and shortcut controls prerequisites: {missing}")
    claim = read_tsv(required[1])
    if claim.iloc[0].get("all_model_configs_completed", "") != "YES":
        raise RuntimeError("design confirmation completion claim is not YES")

    samples = read_tsv(required[0])
    for column in ["embedding_row", "width", "height", "bytes"]:
        samples[column] = pd.to_numeric(samples[column], errors="raise")
    if samples["photo_path_id"].nunique() != 13354:
        raise RuntimeError(f"Unexpected eligible physical paths: {samples['photo_path_id'].nunique()}")

    feature_manifest = (
        samples.sort_values("photo_path_id")
        .drop_duplicates("photo_path_id")
        [[
            "photo_path_id", "absolute_path", "relative_path", "embedding_row", "width", "height",
            "bytes", "image_format", "mode", "exif_make", "exif_model", "source_dataset_id",
            "source_archive", "corrected_split", "exact_component_id", "final_strict_lineage_id",
            "final_split_block_id",
        ]]
        .reset_index(drop=True)
    )
    feature_manifest.insert(0, "visual_feature_row", np.arange(len(feature_manifest), dtype=int))
    write_tsv(feature_manifest, paths.prepared / "stage3c_feature_manifest.tsv")

    scenario_rows: list[dict[str, object]] = []
    type_plan_rows: list[pd.DataFrame] = []
    plan_audit_rows: list[pd.DataFrame] = []
    removal_rows: list[pd.DataFrame] = []
    removal_audits: list[pd.DataFrame] = []
    processed_scenarios: set[str] = set()
    for plan in NULL_PLAN:
        task, naive_train, test, contaminated, scene_safe = make_scenario(samples, plan)
        if plan["scenario_id"] not in processed_scenarios:
            scenario_rows.append({
                "scenario_id": plan["scenario_id"],
                "task_name": plan["task_name"],
                "heldout_source": plan["heldout_source"],
                "naive_train_paths": len(naive_train),
                "scene_safe_train_paths": len(scene_safe),
                "test_paths": len(test),
                "contaminated_train_paths": len(contaminated),
                "contaminated_blocks": contaminated["final_split_block_id"].nunique(),
                "contaminated_strict_lineages": contaminated["final_strict_lineage_id"].nunique(),
                "contaminated_class_counts": "|".join(
                    f"{label}:{int((contaminated['label'] == label).sum())}"
                    for label in TASK_LABEL_ORDERS[str(plan["task_name"])]
                ),
                "contaminated_source_counts": "|".join(
                    f"{source}:{int((contaminated['source_dataset_id'] == source).sum())}"
                    for source in sorted(contaminated["source_dataset_id"].unique())
                ),
            })
            processed_scenarios.add(str(plan["scenario_id"]))
        selected_types, plan_audit = solve_type_plan(contaminated, scene_safe, str(plan["null_design"]))
        selected_types.insert(0, "scenario_id", plan["scenario_id"])
        plan_audit.insert(0, "scenario_id", plan["scenario_id"])
        removals, removal_audit = sample_removal_sets(
            scene_safe,
            selected_types,
            str(plan["scenario_id"]),
            str(plan["null_design"]),
            int(plan["replicates"]),
        )
        type_plan_rows.append(selected_types)
        plan_audit_rows.append(plan_audit)
        removal_rows.append(removals)
        removal_audits.append(removal_audit)

    scenario_table = pd.DataFrame(scenario_rows)
    type_plan = pd.concat(type_plan_rows, ignore_index=True)
    plan_audit = pd.concat(plan_audit_rows, ignore_index=True)
    removal_table = pd.concat(removal_rows, ignore_index=True)
    removal_audit = pd.concat(removal_audits, ignore_index=True)
    write_tsv(scenario_table, paths.prepared / "stage3c_null_scenarios.tsv")
    write_tsv(type_plan, paths.prepared / "stage3c_null_type_plan.tsv")
    write_tsv(plan_audit, paths.prepared / "stage3c_null_plan_audit.tsv")
    write_tsv(removal_table, paths.prepared / "stage3c_null_removal_sets.tsv.gz")
    write_tsv(removal_audit, paths.prepared / "stage3c_null_removal_audit.tsv.gz")

    null_config_rows: list[dict[str, object]] = []
    index = 1
    for scenario in scenario_table.itertuples(index=False):
        for role in ["OBSERVED_NAIVE", "OBSERVED_SCENE_SAFE"]:
            null_config_rows.append({
                "array_index": index,
                "config_id": f"S3C_NULL_{index:04d}",
                "scenario_id": scenario.scenario_id,
                "task_name": scenario.task_name,
                "heldout_source": scenario.heldout_source,
                "config_type": role,
                "null_design": "OBSERVED",
                "replicate": 0,
            })
            index += 1
    for plan in NULL_PLAN:
        for replicate in range(1, int(plan["replicates"]) + 1):
            null_config_rows.append({
                "array_index": index,
                "config_id": f"S3C_NULL_{index:04d}",
                "scenario_id": plan["scenario_id"],
                "task_name": plan["task_name"],
                "heldout_source": plan["heldout_source"],
                "config_type": "MATCHED_NULL",
                "null_design": plan["null_design"],
                "replicate": replicate,
            })
            index += 1
    null_config = pd.DataFrame(null_config_rows)
    if len(null_config) != args.expected_null_configs:
        raise RuntimeError(f"Null config count mismatch: {len(null_config)} != {args.expected_null_configs}")
    write_tsv(null_config, paths.prepared / "stage3c_null_config_table.tsv")

    shortcut_index, shortcut_audit = build_shortcut_assignments(
        samples, paths.prepared / "shortcut_assignments"
    )
    shortcut_config = build_shortcut_config(shortcut_index)
    if len(shortcut_config) != args.expected_shortcut_configs:
        raise RuntimeError(
            f"Shortcut config count mismatch: {len(shortcut_config)} != {args.expected_shortcut_configs}"
        )
    write_tsv(shortcut_index, paths.prepared / "stage3c_shortcut_assignment_index.tsv")
    write_tsv(shortcut_audit, paths.prepared / "stage3c_shortcut_assignment_audit.tsv")
    write_tsv(shortcut_config, paths.prepared / "stage3c_shortcut_config_table.tsv")
    json_dump(TASK_LABEL_ORDERS, paths.prepared / "stage3c_label_orders.json")

    provenance = pd.DataFrame([
        {"item": "stage3b_path_samples", "path": str(required[0]), "sha256": sha256_file(required[0])},
        {"item": "stage3b_claim_status", "path": str(required[1]), "sha256": sha256_file(required[1])},
        {"item": "stage2c_embedding_manifest", "path": str(required[2]), "sha256": sha256_file(required[2])},
        {"item": "stage2c_efficientnet_matrix", "path": str(required[3]), "sha256": sha256_file(required[3])},
    ])
    write_tsv(provenance, paths.prepared / "stage3c_prepare_provenance.tsv")
    summary = pd.DataFrame([
        {"metric": "eligible_physical_paths", "value": feature_manifest["photo_path_id"].nunique()},
        {"metric": "null_scenarios", "value": scenario_table["scenario_id"].nunique()},
        {"metric": "matched_null_designs", "value": len(NULL_PLAN)},
        {"metric": "matched_null_replicates", "value": int(sum(item["replicates"] for item in NULL_PLAN))},
        {"metric": "null_model_configs", "value": len(null_config)},
        {"metric": "visual_feature_tasks", "value": 5},
        {"metric": "shortcut_model_configs", "value": len(shortcut_config)},
        {"metric": "shortcut_repeat_seeds", "value": len(REPEAT_SEEDS)},
    ])
    write_tsv(summary, paths.prepared / "stage3c_prepare_summary.tsv")
    (paths.prepared / ".stage3c_prepared").write_text("OK\n", encoding="utf-8")
    print(summary.to_string(index=False))
    print("\nMatched-removal plan audit:")
    print(plan_audit.to_string(index=False))
    print("\nShortcut configurations:")
    print(shortcut_config.groupby(["analysis_family", "task_name"]).size().rename("configs").reset_index().to_string(index=False))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"FATAL: {type(exc).__name__}: {exc}", file=sys.stderr, flush=True)
        raise

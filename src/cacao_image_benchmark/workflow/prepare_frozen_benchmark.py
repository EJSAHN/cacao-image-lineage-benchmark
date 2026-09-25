#!/usr/bin/env python3
"""Prepare deterministic frozen-feature benchmark leakage-aware benchmark manifests and model configs."""
from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold, StratifiedKFold

from frozen_benchmark_common import (
    CV_DESIGNS,
    FIXED_DESIGNS,
    SEED,
    SOURCE_HOLDOUT_DESIGNS,
    TASK_LABEL_ORDERS,
    ProjectPaths,
    UnionFind,
    add_metadata_features,
    json_dump,
    read_tsv,
    sha256_file,
    stable_id,
    write_tsv,
)

TASK_SPECS = {
    "cacao_coarse_three_class": {
        "eligible_column": "coarse3_task_eligible",
        "label_column": "coarse_label",
    },
    "cacao_causal_five_class": {
        "eligible_column": "causal5_task_eligible",
        "label_column": "fine_label",
    },
    "cocoamonilia_four_stage": {
        "eligible_column": "stage4_task_eligible",
        "label_column": "stage_label",
    },
}

SOURCE_HOLDOUTS = [
    "fig_ghana_balanced",
    "fig_roboflow_mixed",
    "fig_spanish_yolov4",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True, help="Project root under /project")
    parser.add_argument("--big", required=True, help="Large-data root under /90daydata")
    parser.add_argument("--expected-configs", type=int, default=176)
    parser.add_argument("--expected-ambiguity-edges", type=int, default=102)
    parser.add_argument("--expected-primary-blocks", type=int, default=8945)
    parser.add_argument("--expected-ambiguity-blocks", type=int, default=8871)
    return parser.parse_args()


def require_columns(df: pd.DataFrame, columns: Iterable[str], name: str) -> None:
    missing = [column for column in columns if column not in df.columns]
    if missing:
        raise RuntimeError(f"{name} is missing required columns: {missing}")


def yes(series: pd.Series) -> pd.Series:
    return series.fillna("").astype(str).str.upper().eq("YES")


def build_ambiguity_map(
    manifest: pd.DataFrame,
    split_blocks: pd.DataFrame,
    ambiguous: pd.DataFrame,
    expected_edges: int,
    expected_primary_blocks: int,
    expected_ambiguity_blocks: int,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, str]]:
    require_columns(
        ambiguous,
        [
            "component_a",
            "component_b",
            "manual_review_priority",
            "scientific_risk",
            "verification_relation",
        ],
        "final lineage freeze ambiguous-pair table",
    )
    component_to_block = (
        manifest[["exact_component_id", "final_split_block_id"]]
        .drop_duplicates()
        .set_index("exact_component_id")["final_split_block_id"]
        .to_dict()
    )
    work = ambiguous.copy()
    work["block_a"] = work["component_a"].map(component_to_block)
    work["block_b"] = work["component_b"].map(component_to_block)
    high = work[
        work["verification_relation"].eq("AMBIGUOUS_MANUAL_REVIEW")
        & yes(work["manual_review_priority"])
        & work["block_a"].ne("")
        & work["block_b"].ne("")
        & work["block_a"].notna()
        & work["block_b"].notna()
        & work["block_a"].ne(work["block_b"])
        & work["scientific_risk"].ne("GENERAL")
    ].copy()
    high = high.sort_values(["scientific_risk", "component_a", "component_b"]).reset_index(drop=True)
    if len(high) != expected_edges:
        raise RuntimeError(
            f"Expected {expected_edges} high-risk reserve ambiguous edges, found {len(high)}. "
            "Do not silently change the ambiguity sensitivity definition."
        )

    primary_blocks = sorted(split_blocks["final_split_block_id"].dropna().astype(str).unique())
    if len(primary_blocks) != expected_primary_blocks:
        raise RuntimeError(
            f"Expected {expected_primary_blocks} primary split blocks, found {len(primary_blocks)}"
        )
    uf = UnionFind(primary_blocks)
    for row in high.itertuples(index=False):
        uf.union(str(row.block_a), str(row.block_b))

    groups: dict[str, list[str]] = {}
    for block in primary_blocks:
        root = uf.find(block)
        groups.setdefault(root, []).append(block)
    if len(groups) != expected_ambiguity_blocks:
        raise RuntimeError(
            f"Expected {expected_ambiguity_blocks} ambiguity-sensitive blocks, found {len(groups)}"
        )

    block_to_ambiguity: dict[str, str] = {}
    map_rows: list[dict[str, object]] = []
    for members in sorted(groups.values(), key=lambda values: min(values)):
        ambiguity_id = stable_id("CIFAMBIGBLOCK", members)
        for block in sorted(members):
            block_to_ambiguity[block] = ambiguity_id
            map_rows.append(
                {
                    "final_split_block_id": block,
                    "ambiguity_sensitive_block_id": ambiguity_id,
                    "n_primary_blocks_in_ambiguity_group": len(members),
                }
            )
    map_df = pd.DataFrame(map_rows).sort_values(
        ["ambiguity_sensitive_block_id", "final_split_block_id"]
    )
    high["ambiguity_sensitive_block_a"] = high["block_a"].map(block_to_ambiguity)
    high["ambiguity_sensitive_block_b"] = high["block_b"].map(block_to_ambiguity)
    high["merged_for_split_sensitivity"] = np.where(
        high["ambiguity_sensitive_block_a"].eq(high["ambiguity_sensitive_block_b"]),
        "YES",
        "NO",
    )
    return map_df, high, block_to_ambiguity


def build_path_samples(
    manifest: pd.DataFrame,
    embedding_manifest: pd.DataFrame,
    block_to_ambiguity: dict[str, str],
) -> pd.DataFrame:
    require_columns(
        manifest,
        [
            "role",
            "photo_path_id",
            "exact_component_id",
            "final_strict_lineage_id",
            "final_split_block_id",
            "source_dataset_id",
            "source_archive",
            "corrected_split",
            "width",
            "height",
            "bytes",
            "image_format",
            "mode",
            "exif_make",
            "exif_model",
            "analysis_unit_id",
        ],
        "final lineage freeze final benchmark manifest",
    )
    require_columns(
        embedding_manifest,
        ["embedding_row", "exact_component_id"],
        "embedding manifest",
    )
    row_map = (
        embedding_manifest[["exact_component_id", "embedding_row"]]
        .drop_duplicates()
        .set_index("exact_component_id")["embedding_row"]
        .astype(int)
        .to_dict()
    )

    rows: list[pd.DataFrame] = []
    for task_name, spec in TASK_SPECS.items():
        eligible_column = spec["eligible_column"]
        label_column = spec["label_column"]
        labels = TASK_LABEL_ORDERS[task_name]
        require_columns(manifest, [eligible_column, label_column], "final lineage freeze final benchmark manifest")
        subset = manifest[
            manifest["role"].eq("photo_candidate")
            & manifest[eligible_column].eq("YES")
            & manifest[label_column].isin(labels)
        ].copy()
        subset["task_name"] = task_name
        subset["label"] = subset[label_column]
        subset["sample_id"] = subset["photo_path_id"]
        subset["embedding_row"] = subset["exact_component_id"].map(row_map)
        missing = subset[subset["embedding_row"].isna()]
        if not missing.empty:
            raise RuntimeError(
                f"{task_name}: {len(missing)} eligible paths lack image embeddings; "
                f"first={missing.iloc[0]['exact_component_id']}"
            )
        subset["embedding_row"] = subset["embedding_row"].astype(int)
        subset["ambiguity_sensitive_block_id"] = subset["final_split_block_id"].map(
            block_to_ambiguity
        )
        if subset["ambiguity_sensitive_block_id"].isna().any():
            bad = subset[subset["ambiguity_sensitive_block_id"].isna()].iloc[0]
            raise RuntimeError(
                f"Missing ambiguity block for {bad['final_split_block_id']}"
            )
        subset = add_metadata_features(subset)
        subset["n_paths_in_task_strict_lineage"] = subset.groupby(
            "final_strict_lineage_id"
        )["sample_id"].transform("size")
        keep = [
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
            "embedding_row",
            "absolute_path",
            "relative_path",
            "width",
            "height",
            "bytes",
            "image_format",
            "mode",
            "exif_make",
            "exif_model",
            "log_width",
            "log_height",
            "log_bytes",
            "aspect_ratio",
            "megapixels",
            "n_paths_in_task_strict_lineage",
        ]
        rows.append(subset[keep])
    combined = pd.concat(rows, ignore_index=True)
    combined = combined.sort_values(["task_name", "sample_id"]).reset_index(drop=True)
    duplicate = combined.duplicated(["task_name", "sample_id"])
    if duplicate.any():
        raise RuntimeError(f"Duplicate task/sample IDs found: {duplicate.sum()}")
    return combined


def crossing_count(frame: pd.DataFrame, group_column: str) -> int:
    return int((frame.groupby(group_column, sort=False)["fold"].nunique() > 1).sum())


def build_cv_assignments(samples: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    assignment_rows: list[pd.DataFrame] = []
    audit_rows: list[dict[str, object]] = []
    hierarchy_columns = [
        "exact_component_id",
        "final_strict_lineage_id",
        "final_split_block_id",
        "ambiguity_sensitive_block_id",
    ]
    for task_name, labels in TASK_LABEL_ORDERS.items():
        task = samples[samples["task_name"].eq(task_name)].copy()
        task = task.sort_values("sample_id").reset_index(drop=True)
        y = task["label"].to_numpy()
        for design, group_column in CV_DESIGNS.items():
            if design == "PATH_RANDOM_5FOLD":
                splitter = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
                split_iterator = splitter.split(np.zeros((len(task), 1)), y)
            else:
                splitter = StratifiedGroupKFold(
                    n_splits=5, shuffle=True, random_state=SEED
                )
                split_iterator = splitter.split(
                    np.zeros((len(task), 1)), y, groups=task[group_column].to_numpy()
                )
            folds = np.full(len(task), -1, dtype=int)
            for fold, (_, test_idx) in enumerate(split_iterator):
                folds[test_idx] = fold
            if np.any(folds < 0):
                raise RuntimeError(f"Incomplete CV assignment for {task_name}/{design}")
            assigned = task[
                [
                    "task_name",
                    "sample_id",
                    "label",
                    "exact_component_id",
                    "final_strict_lineage_id",
                    "final_split_block_id",
                    "ambiguity_sensitive_block_id",
                ]
            ].copy()
            assigned["evaluation_design"] = design
            assigned["fold"] = folds
            assignment_rows.append(assigned)

            all_labels_present = True
            for fold in range(5):
                if set(assigned.loc[assigned["fold"].eq(fold), "label"]) != set(labels):
                    all_labels_present = False
            row: dict[str, object] = {
                "task_name": task_name,
                "evaluation_design": design,
                "n_samples": len(assigned),
                "n_folds": 5,
                "intended_group_column": group_column,
                "intended_group_crossings": crossing_count(assigned, group_column),
                "all_labels_present_in_every_test_fold": "YES" if all_labels_present else "NO",
                "min_fold_size": int(assigned.groupby("fold").size().min()),
                "max_fold_size": int(assigned.groupby("fold").size().max()),
            }
            for hierarchy in hierarchy_columns:
                row[f"crossing_{hierarchy}"] = crossing_count(assigned, hierarchy)
            audit_rows.append(row)
            if row["intended_group_crossings"] != 0 or not all_labels_present:
                raise RuntimeError(f"CV integrity failed for {task_name}/{design}: {row}")
    assignments = pd.concat(assignment_rows, ignore_index=True)
    audit = pd.DataFrame(audit_rows)
    return assignments, audit


def group_overlap(train: pd.DataFrame, test: pd.DataFrame, column: str) -> int:
    return len(set(train[column]) & set(test[column]))


def build_fixed_assignments(samples: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    task = samples[samples["task_name"].eq("cocoamonilia_four_stage")].copy()
    rows: list[pd.DataFrame] = []
    audits: list[dict[str, object]] = []
    designs = {
        "ORIGINAL_TRAIN_TO_VALIDATION": ({"train"}, {"validation"}),
        "ORIGINAL_TRAINVAL_TO_TEST": ({"train", "validation"}, {"test"}),
    }
    expected_labels = set(TASK_LABEL_ORDERS["cocoamonilia_four_stage"])
    for design, (train_splits, test_splits) in designs.items():
        train = task[task["corrected_split"].isin(train_splits)].copy()
        test = task[task["corrected_split"].isin(test_splits)].copy()
        if set(train["label"]) != expected_labels or set(test["label"]) != expected_labels:
            raise RuntimeError(f"Original split lacks one or more labels for {design}")
        train_rows = train[["task_name", "sample_id", "label"]].copy()
        train_rows["evaluation_design"] = design
        train_rows["role"] = "TRAIN"
        test_rows = test[["task_name", "sample_id", "label"]].copy()
        test_rows["evaluation_design"] = design
        test_rows["role"] = "TEST"
        rows.extend([train_rows, test_rows])
        audits.append(
            {
                "task_name": "cocoamonilia_four_stage",
                "evaluation_design": design,
                "train_n": len(train),
                "test_n": len(test),
                "train_labels": "|".join(sorted(train["label"].unique())),
                "test_labels": "|".join(sorted(test["label"].unique())),
                "exact_component_overlap": group_overlap(train, test, "exact_component_id"),
                "strict_lineage_overlap": group_overlap(train, test, "final_strict_lineage_id"),
                "verified_scene_block_overlap": group_overlap(train, test, "final_split_block_id"),
                "ambiguity_block_overlap": group_overlap(
                    train, test, "ambiguity_sensitive_block_id"
                ),
            }
        )
    return pd.concat(rows, ignore_index=True), pd.DataFrame(audits)


def build_source_holdout_assignments(
    samples: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    task = samples[samples["task_name"].eq("cacao_coarse_three_class")].copy()
    expected_labels = set(TASK_LABEL_ORDERS["cacao_coarse_three_class"])
    rows: list[pd.DataFrame] = []
    audits: list[dict[str, object]] = []
    for heldout in SOURCE_HOLDOUTS:
        test = task[task["source_dataset_id"].eq(heldout)].copy()
        other = task[~task["source_dataset_id"].eq(heldout)].copy()
        heldout_strict = set(test["final_strict_lineage_id"])
        heldout_scene = set(test["final_split_block_id"])
        designs = {
            "SOURCE_HOLDOUT_NAIVE": other,
            "SOURCE_HOLDOUT_STRICT_SAFE": other[
                ~other["final_strict_lineage_id"].isin(heldout_strict)
            ],
            "SOURCE_HOLDOUT_SCENE_SAFE": other[
                ~other["final_split_block_id"].isin(heldout_scene)
            ],
        }
        if set(test["label"]) != expected_labels:
            raise RuntimeError(f"Held-out source {heldout} does not contain all coarse labels")
        for design, train in designs.items():
            if set(train["label"]) != expected_labels:
                raise RuntimeError(f"Training set lacks labels for {heldout}/{design}")
            train_rows = train[["task_name", "sample_id", "label"]].copy()
            train_rows["evaluation_design"] = design
            train_rows["heldout_source"] = heldout
            train_rows["role"] = "TRAIN"
            test_rows = test[["task_name", "sample_id", "label"]].copy()
            test_rows["evaluation_design"] = design
            test_rows["heldout_source"] = heldout
            test_rows["role"] = "TEST"
            rows.extend([train_rows, test_rows])
            audits.append(
                {
                    "task_name": "cacao_coarse_three_class",
                    "heldout_source": heldout,
                    "evaluation_design": design,
                    "train_n": len(train),
                    "test_n": len(test),
                    "train_labels": "|".join(sorted(train["label"].unique())),
                    "test_labels": "|".join(sorted(test["label"].unique())),
                    "exact_component_overlap": group_overlap(train, test, "exact_component_id"),
                    "strict_lineage_overlap": group_overlap(
                        train, test, "final_strict_lineage_id"
                    ),
                    "verified_scene_block_overlap": group_overlap(
                        train, test, "final_split_block_id"
                    ),
                    "ambiguity_block_overlap": group_overlap(
                        train, test, "ambiguity_sensitive_block_id"
                    ),
                }
            )
    return pd.concat(rows, ignore_index=True), pd.DataFrame(audits)


def build_config_table(expected_configs: int) -> pd.DataFrame:
    features = ["metadata", "resnet18", "efficientnet_b0", "concat"]
    classifiers = ["logistic", "linear_svm"]
    rows: list[dict[str, object]] = []

    def add(
        task_name: str,
        design_family: str,
        evaluation_design: str,
        feature_set: str,
        classifier: str,
        heldout_source: str = "",
        notes: str = "",
    ) -> None:
        rows.append(
            {
                "task_name": task_name,
                "design_family": design_family,
                "evaluation_design": evaluation_design,
                "feature_set": feature_set,
                "classifier": classifier,
                "heldout_source": heldout_source,
                "n_folds": 5 if design_family == "CV" else 1,
                "primary_model": (
                    "YES"
                    if feature_set == "efficientnet_b0" and classifier == "logistic"
                    else "NO"
                ),
                "notes": notes,
                "label_order": "|".join(TASK_LABEL_ORDERS[task_name]),
            }
        )

    for task_name in ["cacao_coarse_three_class", "cocoamonilia_four_stage"]:
        for design in CV_DESIGNS:
            for feature in features:
                for classifier in classifiers:
                    add(task_name, "CV", design, feature, classifier)

    for design in ["ORIGINAL_TRAIN_TO_VALIDATION", "ORIGINAL_TRAINVAL_TO_TEST"]:
        for feature in features:
            for classifier in classifiers:
                add(
                    "cocoamonilia_four_stage",
                    "FIXED",
                    design,
                    feature,
                    classifier,
                    notes="Original public split; contamination is measured rather than repaired.",
                )

    for heldout in SOURCE_HOLDOUTS:
        for design in [
            "SOURCE_HOLDOUT_NAIVE",
            "SOURCE_HOLDOUT_STRICT_SAFE",
            "SOURCE_HOLDOUT_SCENE_SAFE",
        ]:
            for feature in features:
                for classifier in classifiers:
                    add(
                        "cacao_coarse_three_class",
                        "SOURCE_HOLDOUT",
                        design,
                        feature,
                        classifier,
                        heldout_source=heldout,
                        notes="Test paths are from the named held-out source; safety level changes training exclusions.",
                    )

    # Five-class is deliberately secondary and excludes unrestricted source-holdout claims.
    for design in [
        "VERIFIED_SCENE_BLOCK_5FOLD",
        "AMBIGUITY_SENS_BLOCK_5FOLD",
    ]:
        for feature in features:
            add(
                "cacao_causal_five_class",
                "CV",
                design,
                feature,
                "logistic",
                notes="Secondary grouped benchmark; no unrestricted LOSO because rare causal labels are source-exclusive.",
            )

    config = pd.DataFrame(rows)
    config.insert(0, "config_id", [f"S3A{i:04d}" for i in range(1, len(config) + 1)])
    config.insert(1, "array_index", np.arange(1, len(config) + 1))
    if len(config) != expected_configs:
        raise RuntimeError(f"Expected {expected_configs} configs, generated {len(config)}")
    return config


def verify_embedding_inputs(paths: ProjectPaths) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, object]]:
    manifest_path = paths.stage2c_embeddings_meta / "stage2c_embedding_manifest.tsv"
    provenance_path = paths.stage2c_embeddings_meta / "stage2c_embedding_provenance.tsv"
    embedding_manifest = read_tsv(manifest_path)
    provenance = read_tsv(provenance_path)
    require_columns(
        provenance,
        ["model", "embedding_rows", "embedding_dim", "output_path", "output_sha256"],
        "embedding provenance",
    )
    details: dict[str, object] = {}
    for model in ["resnet18", "efficientnet_b0"]:
        row = provenance[provenance["model"].eq(model)]
        if len(row) != 1:
            raise RuntimeError(f"Expected one embedding retrieval provenance row for {model}, found {len(row)}")
        record = row.iloc[0]
        matrix_path = Path(record["output_path"])
        if not matrix_path.is_file():
            raise RuntimeError(f"Missing embedding matrix: {matrix_path}")
        observed_sha = sha256_file(matrix_path)
        if observed_sha != record["output_sha256"]:
            raise RuntimeError(
                f"Embedding SHA-256 mismatch for {model}: {observed_sha} != {record['output_sha256']}"
            )
        matrix = np.load(matrix_path, mmap_mode="r")
        expected_shape = (int(record["embedding_rows"]), int(record["embedding_dim"]))
        if tuple(matrix.shape) != expected_shape:
            raise RuntimeError(
                f"Embedding shape mismatch for {model}: {matrix.shape} != {expected_shape}"
            )
        details[f"{model}_path"] = str(matrix_path)
        details[f"{model}_sha256"] = observed_sha
        details[f"{model}_shape"] = "x".join(map(str, matrix.shape))
    if len(embedding_manifest) != int(provenance.iloc[0]["embedding_rows"]):
        raise RuntimeError("Embedding manifest row count does not match matrix rows")
    if embedding_manifest["exact_component_id"].duplicated().any():
        raise RuntimeError("Embedding manifest contains duplicate exact_component_id values")
    return embedding_manifest, provenance, details


def main() -> int:
    args = parse_args()
    paths = ProjectPaths(Path(args.root), Path(args.big))
    paths.prepared.mkdir(parents=True, exist_ok=True)
    paths.task_results.mkdir(parents=True, exist_ok=True)
    paths.aggregate.mkdir(parents=True, exist_ok=True)
    paths.figures.mkdir(parents=True, exist_ok=True)

    claim_path = paths.stage2e / "stage2e_claim_status.tsv"
    manifest_path = paths.stage2e / "stage2e_final_benchmark_manifest.tsv"
    split_blocks_path = paths.stage2e / "stage2e_final_split_blocks.tsv"
    ambiguous_path = paths.stage2e / "stage2e_reserve_ambiguous_manual_review.tsv"
    for path in [claim_path, manifest_path, split_blocks_path, ambiguous_path]:
        if not path.is_file():
            raise RuntimeError(f"Required final lineage freeze input is missing: {path}")

    claim = read_tsv(claim_path)
    if len(claim) != 1:
        raise RuntimeError("final lineage freeze claim-status table must contain exactly one row")
    required_claims = [
        "frozen_verifier_control_qc_pass",
        "evidence_gated_reserve_exhausted",
        "conservative_strict_lineage_freeze_final",
        "conservative_split_block_freeze_final",
    ]
    for column in required_claims:
        if claim.iloc[0].get(column, "") != "YES":
            raise RuntimeError(f"final lineage freeze prerequisite not satisfied: {column}")

    manifest = read_tsv(manifest_path)
    split_blocks = read_tsv(split_blocks_path)
    ambiguous = read_tsv(ambiguous_path)
    embedding_manifest, embedding_provenance, embedding_details = verify_embedding_inputs(paths)

    ambiguity_map, ambiguity_edges, block_to_ambiguity = build_ambiguity_map(
        manifest,
        split_blocks,
        ambiguous,
        args.expected_ambiguity_edges,
        args.expected_primary_blocks,
        args.expected_ambiguity_blocks,
    )
    samples = build_path_samples(manifest, embedding_manifest, block_to_ambiguity)
    cv_assignments, cv_audit = build_cv_assignments(samples)
    fixed_assignments, fixed_audit = build_fixed_assignments(samples)
    source_assignments, source_audit = build_source_holdout_assignments(samples)
    configs = build_config_table(args.expected_configs)

    write_tsv(samples, paths.prepared / "stage3a_path_samples.tsv.gz")
    write_tsv(cv_assignments, paths.prepared / "stage3a_cv_assignments.tsv.gz")
    write_tsv(fixed_assignments, paths.prepared / "stage3a_fixed_assignments.tsv.gz")
    write_tsv(
        source_assignments,
        paths.prepared / "stage3a_source_holdout_assignments.tsv.gz",
    )
    write_tsv(ambiguity_map, paths.prepared / "stage3a_ambiguity_block_map.tsv")
    write_tsv(ambiguity_edges, paths.prepared / "stage3a_ambiguity_edge_audit.tsv")
    write_tsv(cv_audit, paths.prepared / "stage3a_cv_split_integrity_audit.tsv")
    write_tsv(fixed_audit, paths.prepared / "stage3a_original_split_integrity_audit.tsv")
    write_tsv(source_audit, paths.prepared / "stage3a_source_holdout_integrity_audit.tsv")
    write_tsv(configs, paths.prepared / "stage3a_config_table.tsv")

    sample_summary = (
        samples.groupby(["task_name", "label"], as_index=False)
        .agg(
            physical_paths=("sample_id", "size"),
            exact_components=("exact_component_id", "nunique"),
            strict_lineages=("final_strict_lineage_id", "nunique"),
            verified_scene_blocks=("final_split_block_id", "nunique"),
            ambiguity_sensitive_blocks=("ambiguity_sensitive_block_id", "nunique"),
        )
    )
    write_tsv(sample_summary, paths.prepared / "stage3a_sample_summary.tsv")
    source_label_summary = (
        samples.groupby(["task_name", "source_dataset_id", "source_archive", "label"], as_index=False)
        .agg(
            physical_paths=("sample_id", "size"),
            exact_components=("exact_component_id", "nunique"),
            strict_lineages=("final_strict_lineage_id", "nunique"),
            verified_scene_blocks=("final_split_block_id", "nunique"),
        )
    )
    write_tsv(source_label_summary, paths.prepared / "stage3a_source_label_counts.tsv")
    original_split_counts = (
        samples[samples["task_name"].eq("cocoamonilia_four_stage")]
        .groupby(["corrected_split", "label"], as_index=False)
        .agg(
            physical_paths=("sample_id", "size"),
            exact_components=("exact_component_id", "nunique"),
            strict_lineages=("final_strict_lineage_id", "nunique"),
            verified_scene_blocks=("final_split_block_id", "nunique"),
        )
    )
    write_tsv(original_split_counts, paths.prepared / "stage3a_cocoamonilia_original_split_counts.tsv")

    provenance_rows = [
        {"item": "stage2e_manifest", "path": str(manifest_path), "sha256": sha256_file(manifest_path)},
        {"item": "stage2e_split_blocks", "path": str(split_blocks_path), "sha256": sha256_file(split_blocks_path)},
        {"item": "stage2e_ambiguous_reserve", "path": str(ambiguous_path), "sha256": sha256_file(ambiguous_path)},
        {
            "item": "stage2c_embedding_manifest",
            "path": str(paths.stage2c_embeddings_meta / "stage2c_embedding_manifest.tsv"),
            "sha256": sha256_file(paths.stage2c_embeddings_meta / "stage2c_embedding_manifest.tsv"),
        },
    ]
    for key, value in embedding_details.items():
        provenance_rows.append({"item": key, "path": value if key.endswith("_path") else "", "sha256": value if key.endswith("_sha256") else "", "value": value})
    provenance = pd.DataFrame(provenance_rows)
    write_tsv(provenance, paths.prepared / "stage3a_prepare_provenance.tsv")

    summary_rows = [
        {"metric": "model_configs", "value": len(configs)},
        {"metric": "path_sample_rows_all_tasks", "value": len(samples)},
        {"metric": "primary_verified_split_blocks", "value": split_blocks["final_split_block_id"].nunique()},
        {"metric": "high_risk_ambiguous_edges", "value": len(ambiguity_edges)},
        {"metric": "ambiguity_sensitive_blocks", "value": ambiguity_map["ambiguity_sensitive_block_id"].nunique()},
        {"metric": "cv_assignment_rows", "value": len(cv_assignments)},
        {"metric": "source_holdout_assignment_rows", "value": len(source_assignments)},
        {"metric": "fixed_original_assignment_rows", "value": len(fixed_assignments)},
    ]
    write_tsv(pd.DataFrame(summary_rows), paths.prepared / "stage3a_prepare_summary.tsv")

    json_dump(TASK_LABEL_ORDERS, paths.prepared / "stage3a_label_orders.json")
    (paths.prepared / ".stage3a_prepared").touch()
    print(pd.DataFrame(summary_rows).to_string(index=False))
    print("\nModel configs by task/design family:")
    print(
        configs.groupby(["task_name", "design_family"]).size().rename("configs").reset_index().to_string(index=False)
    )
    print("\nAmbiguity sensitivity:")
    print(
        pd.DataFrame(
            {
                "high_risk_edges": [len(ambiguity_edges)],
                "primary_blocks": [len(split_blocks)],
                "ambiguity_sensitive_blocks": [ambiguity_map["ambiguity_sensitive_block_id"].nunique()],
            }
        ).to_string(index=False)
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

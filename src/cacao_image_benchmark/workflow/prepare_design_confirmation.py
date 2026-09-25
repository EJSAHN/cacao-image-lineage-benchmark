#!/usr/bin/env python3
"""Prepare design confirmation repeated-CV, fixed-test, source-holdout, and corrected five-class designs."""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold, StratifiedKFold

from design_confirmation_common import (
    COCO_TEST_DESIGNS,
    CV_DESIGNS,
    FIVECLASS_CV_DESIGNS,
    REPEAT_SEEDS,
    ROBUST_REPEAT_SEEDS,
    SOURCE_HOLDOUTS,
    SOURCE_HOLDOUT_DESIGNS,
    TASK_EXPECTED_UNITS,
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

EXPECTED_CONFIGS = 656
EXPECTED_AMBIGUITY_EDGES = 102
EXPECTED_PRIMARY_BLOCKS = 8945
EXPECTED_AMBIGUITY_BLOCKS = 8871

TASK_SPECS = {
    "cacao_coarse_three_class": {
        "eligible_column": "coarse3_task_eligible",
        "label_column": "coarse_label",
        "unit_table": "stage2e_task_cacao_coarse_three_class_units.tsv",
    },
    "cacao_causal_five_class": {
        "eligible_column": "causal5_task_eligible",
        # Critical frozen-feature benchmark correction: CocoaMonilia stages map to the causal class frosty_pod.
        "label_column": "coarse_label",
        "unit_table": "stage2e_task_cacao_causal_five_class_units.tsv",
    },
    "cocoamonilia_four_stage": {
        "eligible_column": "stage4_task_eligible",
        "label_column": "stage_label",
        "unit_table": "stage2e_task_cocoamonilia_four_stage_units.tsv",
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True)
    parser.add_argument("--big", required=True)
    parser.add_argument("--expected-configs", type=int, default=EXPECTED_CONFIGS)
    return parser.parse_args()


def require_columns(df: pd.DataFrame, columns: Iterable[str], name: str) -> None:
    missing = [column for column in columns if column not in df.columns]
    if missing:
        raise RuntimeError(f"{name} is missing required columns: {missing}")


def yes(series: pd.Series) -> pd.Series:
    return series.fillna("").astype(str).str.upper().eq("YES")


def build_ambiguity_map(
    manifest: pd.DataFrame, split_blocks: pd.DataFrame, ambiguous: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, str]]:
    require_columns(
        ambiguous,
        ["component_a", "component_b", "manual_review_priority", "scientific_risk", "verification_relation"],
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
        & work["block_a"].notna()
        & work["block_b"].notna()
        & work["block_a"].ne("")
        & work["block_b"].ne("")
        & work["block_a"].ne(work["block_b"])
        & work["scientific_risk"].ne("GENERAL")
    ].copy()
    high = high.sort_values(["scientific_risk", "component_a", "component_b"]).reset_index(drop=True)
    if len(high) != EXPECTED_AMBIGUITY_EDGES:
        raise RuntimeError(f"Expected {EXPECTED_AMBIGUITY_EDGES} high-risk ambiguous edges, found {len(high)}")

    blocks = sorted(split_blocks["final_split_block_id"].astype(str).unique())
    if len(blocks) != EXPECTED_PRIMARY_BLOCKS:
        raise RuntimeError(f"Expected {EXPECTED_PRIMARY_BLOCKS} primary blocks, found {len(blocks)}")
    uf = UnionFind(blocks)
    for row in high.itertuples(index=False):
        uf.union(str(row.block_a), str(row.block_b))
    groups: dict[str, list[str]] = {}
    for block in blocks:
        groups.setdefault(uf.find(block), []).append(block)
    if len(groups) != EXPECTED_AMBIGUITY_BLOCKS:
        raise RuntimeError(f"Expected {EXPECTED_AMBIGUITY_BLOCKS} ambiguity blocks, found {len(groups)}")

    mapping: dict[str, str] = {}
    rows = []
    for members in sorted(groups.values(), key=min):
        aid = stable_id("CIFAMBIGBLOCK", members)
        for block in sorted(members):
            mapping[block] = aid
            rows.append({
                "final_split_block_id": block,
                "ambiguity_sensitive_block_id": aid,
                "n_primary_blocks_in_ambiguity_group": len(members),
            })
    map_df = pd.DataFrame(rows)
    high["ambiguity_sensitive_block_a"] = high["block_a"].map(mapping)
    high["ambiguity_sensitive_block_b"] = high["block_b"].map(mapping)
    high["merged_for_split_sensitivity"] = np.where(
        high["ambiguity_sensitive_block_a"].eq(high["ambiguity_sensitive_block_b"]), "YES", "NO"
    )
    return map_df, high, mapping


def verify_embeddings(paths: ProjectPaths) -> tuple[pd.DataFrame, pd.DataFrame, list[dict[str, str]]]:
    manifest_path = paths.stage2c_meta / "stage2c_embedding_manifest.tsv"
    provenance_path = paths.stage2c_meta / "stage2c_embedding_provenance.tsv"
    embedding_manifest = read_tsv(manifest_path)
    provenance = read_tsv(provenance_path)
    require_columns(provenance, ["model", "embedding_rows", "embedding_dim", "output_path", "output_sha256"], "embedding provenance")
    records = []
    for model in ["resnet18", "efficientnet_b0"]:
        row = provenance[provenance["model"].eq(model)]
        if len(row) != 1:
            raise RuntimeError(f"Expected one provenance row for {model}")
        record = row.iloc[0]
        matrix_path = Path(record["output_path"])
        if not matrix_path.is_file():
            raise RuntimeError(f"Missing embedding matrix: {matrix_path}")
        observed = sha256_file(matrix_path)
        if observed != record["output_sha256"]:
            raise RuntimeError(f"Embedding checksum mismatch for {model}")
        matrix = np.load(matrix_path, mmap_mode="r")
        shape = (int(record["embedding_rows"]), int(record["embedding_dim"]))
        if tuple(matrix.shape) != shape:
            raise RuntimeError(f"Embedding shape mismatch for {model}: {matrix.shape} != {shape}")
        records.append({"item": f"{model}_embedding", "path": str(matrix_path), "sha256": observed, "value": "x".join(map(str, matrix.shape))})
    if embedding_manifest["exact_component_id"].duplicated().any():
        raise RuntimeError("Duplicate exact_component_id in embedding manifest")
    return embedding_manifest, provenance, records


def build_path_samples(
    paths: ProjectPaths,
    manifest: pd.DataFrame,
    embedding_manifest: pd.DataFrame,
    ambiguity_map: dict[str, str],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    require_columns(
        manifest,
        [
            "role", "photo_path_id", "analysis_unit_id", "exact_component_id",
            "final_strict_lineage_id", "final_split_block_id", "source_dataset_id",
            "source_archive", "corrected_split", "width", "height", "bytes",
            "image_format", "mode", "exif_make", "exif_model", "absolute_path", "relative_path",
        ],
        "final lineage freeze final benchmark manifest",
    )
    row_map = (
        embedding_manifest[["exact_component_id", "embedding_row"]]
        .drop_duplicates()
        .set_index("exact_component_id")["embedding_row"].astype(int).to_dict()
    )
    frames = []
    scope_rows = []
    for task_name, spec in TASK_SPECS.items():
        labels = TASK_LABEL_ORDERS[task_name]
        eligible = spec["eligible_column"]
        label_column = spec["label_column"]
        require_columns(manifest, [eligible, label_column], "final lineage freeze final benchmark manifest")
        unit_path = paths.stage2e / spec["unit_table"]
        units = read_tsv(unit_path)
        require_columns(units, ["analysis_unit_id", "final_strict_lineage_id", "representative_photo_path_id", "task_label"], unit_path.name)
        if len(units) != TASK_EXPECTED_UNITS[task_name]:
            raise RuntimeError(f"{task_name}: expected {TASK_EXPECTED_UNITS[task_name]} unit rows, found {len(units)}")
        representative = units.set_index("final_strict_lineage_id")["representative_photo_path_id"].to_dict()
        task_label_map = units.set_index("final_strict_lineage_id")["task_label"].to_dict()

        subset = manifest[
            manifest["role"].eq("photo_candidate")
            & manifest[eligible].eq("YES")
            & manifest[label_column].isin(labels)
        ].copy()
        subset["task_name"] = task_name
        subset["label"] = subset[label_column]
        subset["sample_id"] = subset["photo_path_id"]
        subset["embedding_row"] = subset["exact_component_id"].map(row_map)
        if subset["embedding_row"].isna().any():
            bad = subset[subset["embedding_row"].isna()].iloc[0]
            raise RuntimeError(f"{task_name}: missing embedding for {bad['exact_component_id']}")
        subset["embedding_row"] = subset["embedding_row"].astype(int)
        subset["ambiguity_sensitive_block_id"] = subset["final_split_block_id"].map(ambiguity_map)
        if subset["ambiguity_sensitive_block_id"].isna().any():
            raise RuntimeError(f"{task_name}: missing ambiguity block")
        subset["lineage_representative_sample_id"] = subset["final_strict_lineage_id"].map(representative)
        subset["is_lineage_representative"] = np.where(
            subset["sample_id"].eq(subset["lineage_representative_sample_id"]), "YES", "NO"
        )
        subset["stage2e_task_label"] = subset["final_strict_lineage_id"].map(task_label_map)
        mismatch = subset.groupby("final_strict_lineage_id")["label"].first().ne(
            pd.Series(task_label_map)
        )
        if mismatch.any():
            first = mismatch[mismatch].index[0]
            raise RuntimeError(f"{task_name}: path label disagrees with final lineage freeze task_label for {first}")
        subset = add_metadata_features(subset)
        subset["n_paths_in_task_strict_lineage"] = subset.groupby("final_strict_lineage_id")["sample_id"].transform("size")
        unique_units = subset["analysis_unit_id"].nunique()
        if unique_units != TASK_EXPECTED_UNITS[task_name]:
            raise RuntimeError(f"{task_name}: expected {TASK_EXPECTED_UNITS[task_name]} unique units, found {unique_units}")
        reps_per_lineage = subset.groupby("final_strict_lineage_id")["is_lineage_representative"].apply(lambda s: int((s == "YES").sum()))
        if not reps_per_lineage.eq(1).all():
            raise RuntimeError(f"{task_name}: each strict lineage must have one representative")
        keep = [
            "task_name", "sample_id", "label", "photo_path_id", "analysis_unit_id",
            "exact_component_id", "final_strict_lineage_id", "final_split_block_id",
            "ambiguity_sensitive_block_id", "source_dataset_id", "source_archive",
            "corrected_split", "embedding_row", "absolute_path", "relative_path",
            "width", "height", "bytes", "image_format", "mode", "exif_make", "exif_model",
            "log_width", "log_height", "log_bytes", "aspect_ratio", "megapixels",
            "n_paths_in_task_strict_lineage", "lineage_representative_sample_id",
            "is_lineage_representative", "stage2e_task_label",
        ]
        frames.append(subset[keep])
        for label, group in subset.groupby("label"):
            scope_rows.append({
                "task_name": task_name,
                "label": label,
                "physical_paths": len(group),
                "strict_analysis_units": group["analysis_unit_id"].nunique(),
                "verified_scene_blocks": group["final_split_block_id"].nunique(),
                "ambiguity_sensitive_blocks": group["ambiguity_sensitive_block_id"].nunique(),
            })
    combined = pd.concat(frames, ignore_index=True).sort_values(["task_name", "sample_id"]).reset_index(drop=True)
    if combined.duplicated(["task_name", "sample_id"]).any():
        raise RuntimeError("Duplicate task/sample rows")
    return combined, pd.DataFrame(scope_rows)


def crossing_count(frame: pd.DataFrame, group_column: str) -> int:
    return int((frame.groupby(group_column)["fold"].nunique() > 1).sum())


def write_repeated_assignments(paths: ProjectPaths, samples: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    assignment_dir = paths.prepared / "repeated_assignments"
    assignment_dir.mkdir(parents=True, exist_ok=True)
    index_rows = []
    audit_rows = []
    task_designs = {
        "cacao_coarse_three_class": CV_DESIGNS,
        "cocoamonilia_four_stage": CV_DESIGNS,
        "cacao_causal_five_class": FIVECLASS_CV_DESIGNS,
    }
    for task_name, designs in task_designs.items():
        task = samples[samples["task_name"].eq(task_name)].sort_values("sample_id").reset_index(drop=True)
        y = task["label"].to_numpy()
        labels = set(TASK_LABEL_ORDERS[task_name])
        for design, group_column in designs.items():
            for repeat_index, seed in enumerate(REPEAT_SEEDS, start=1):
                if design == "PATH_RANDOM_5FOLD":
                    splitter = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
                    iterator = splitter.split(np.zeros((len(task), 1)), y)
                else:
                    splitter = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=seed)
                    iterator = splitter.split(np.zeros((len(task), 1)), y, groups=task[group_column].to_numpy())
                folds = np.full(len(task), -1, dtype=int)
                for fold, (_, test_idx) in enumerate(iterator):
                    folds[test_idx] = fold
                if np.any(folds < 0):
                    raise RuntimeError(f"Incomplete assignment {task_name}/{design}/repeat{repeat_index}")
                assigned = task[["sample_id", "label", "exact_component_id", "final_strict_lineage_id", "final_split_block_id", "ambiguity_sensitive_block_id"]].copy()
                assigned["fold"] = folds
                path = assignment_dir / f"{task_name}__{design}__R{repeat_index:02d}.tsv.gz"
                write_tsv(assigned, path)
                label_ok = all(set(assigned.loc[assigned["fold"].eq(f), "label"]) == labels for f in range(5))
                crossings = crossing_count(assigned, group_column)
                if not label_ok or crossings != 0:
                    raise RuntimeError(f"Split integrity failure {task_name}/{design}/R{repeat_index}: labels={label_ok}, crossings={crossings}")
                index_rows.append({
                    "task_name": task_name,
                    "evaluation_design": design,
                    "repeat_index": repeat_index,
                    "split_seed": seed,
                    "assignment_path": str(path),
                    "assignment_sha256": sha256_file(path),
                    "assignment_rows": len(assigned),
                })
                audit_rows.append({
                    "task_name": task_name,
                    "evaluation_design": design,
                    "repeat_index": repeat_index,
                    "split_seed": seed,
                    "intended_group_column": group_column,
                    "intended_group_crossings": crossings,
                    "all_labels_present_in_every_test_fold": "YES" if label_ok else "NO",
                    "min_fold_size": int(assigned.groupby("fold").size().min()),
                    "max_fold_size": int(assigned.groupby("fold").size().max()),
                })
    return pd.DataFrame(index_rows), pd.DataFrame(audit_rows)


def group_overlap(train: pd.DataFrame, test: pd.DataFrame, column: str) -> int:
    return len(set(train[column]) & set(test[column]))


def make_fixed_assignments(paths: ProjectPaths, samples: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    out_dir = paths.prepared / "fixed_assignments"
    out_dir.mkdir(parents=True, exist_ok=True)
    task = samples[samples["task_name"].eq("cocoamonilia_four_stage")].copy()
    base_train = task[task["corrected_split"].isin(["train", "validation"])].copy()
    test = task[task["corrected_split"].eq("test")].copy()
    if set(base_train["label"]) != set(TASK_LABEL_ORDERS["cocoamonilia_four_stage"]) or set(test["label"]) != set(TASK_LABEL_ORDERS["cocoamonilia_four_stage"]):
        raise RuntimeError("CocoaMonilia fixed test lacks one or more labels")
    test_sets = {
        "exact_component_id": set(test["exact_component_id"]),
        "final_strict_lineage_id": set(test["final_strict_lineage_id"]),
        "final_split_block_id": set(test["final_split_block_id"]),
        "ambiguity_sensitive_block_id": set(test["ambiguity_sensitive_block_id"]),
    }
    design_train = {
        "COCO_TEST_NAIVE": base_train,
        "COCO_TEST_EXACT_SAFE": base_train[~base_train["exact_component_id"].isin(test_sets["exact_component_id"])],
        "COCO_TEST_STRICT_SAFE": base_train[~base_train["final_strict_lineage_id"].isin(test_sets["final_strict_lineage_id"])],
        "COCO_TEST_SCENE_SAFE": base_train[~base_train["final_split_block_id"].isin(test_sets["final_split_block_id"])],
        "COCO_TEST_AMBIGUITY_SAFE": base_train[~base_train["ambiguity_sensitive_block_id"].isin(test_sets["ambiguity_sensitive_block_id"])],
    }
    index_rows = []
    audit_rows = []
    expected_test_ids = set(test["sample_id"])
    for design in COCO_TEST_DESIGNS:
        train = design_train[design]
        if set(train["label"]) != set(TASK_LABEL_ORDERS["cocoamonilia_four_stage"]):
            raise RuntimeError(f"Training labels missing for {design}")
        assignments = pd.concat([
            train[["sample_id", "label"]].assign(role="TRAIN"),
            test[["sample_id", "label"]].assign(role="TEST"),
        ], ignore_index=True)
        path = out_dir / f"cocoamonilia_four_stage__{design}.tsv.gz"
        write_tsv(assignments, path)
        if set(assignments.loc[assignments["role"].eq("TEST"), "sample_id"]) != expected_test_ids:
            raise RuntimeError(f"Fixed test IDs changed for {design}")
        index_rows.append({
            "analysis_family": "COCO_FIXED_TEST",
            "task_name": "cocoamonilia_four_stage",
            "evaluation_design": design,
            "heldout_source": "",
            "assignment_path": str(path),
            "assignment_sha256": sha256_file(path),
        })
        audit_rows.append({
            "analysis_family": "COCO_FIXED_TEST",
            "evaluation_design": design,
            "base_train_n": len(base_train),
            "train_n": len(train),
            "train_paths_removed": len(base_train) - len(train),
            "test_n": len(test),
            "exact_component_overlap": group_overlap(train, test, "exact_component_id"),
            "strict_lineage_overlap": group_overlap(train, test, "final_strict_lineage_id"),
            "verified_scene_block_overlap": group_overlap(train, test, "final_split_block_id"),
            "ambiguity_block_overlap": group_overlap(train, test, "ambiguity_sensitive_block_id"),
        })
    return pd.DataFrame(index_rows), pd.DataFrame(audit_rows)


def make_source_assignments(paths: ProjectPaths, samples: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    out_dir = paths.prepared / "source_assignments"
    out_dir.mkdir(parents=True, exist_ok=True)
    task = samples[samples["task_name"].eq("cacao_coarse_three_class")].copy()
    index_rows = []
    audit_rows = []
    labels = set(TASK_LABEL_ORDERS["cacao_coarse_three_class"])
    for heldout in SOURCE_HOLDOUTS:
        test = task[task["source_dataset_id"].eq(heldout)].copy()
        base_train = task[~task["source_dataset_id"].eq(heldout)].copy()
        if set(test["label"]) != labels:
            raise RuntimeError(f"Held-out source {heldout} lacks coarse labels")
        test_sets = {
            "exact_component_id": set(test["exact_component_id"]),
            "final_strict_lineage_id": set(test["final_strict_lineage_id"]),
            "final_split_block_id": set(test["final_split_block_id"]),
            "ambiguity_sensitive_block_id": set(test["ambiguity_sensitive_block_id"]),
        }
        design_train = {
            "SOURCE_HOLDOUT_NAIVE": base_train,
            "SOURCE_HOLDOUT_EXACT_SAFE": base_train[~base_train["exact_component_id"].isin(test_sets["exact_component_id"])],
            "SOURCE_HOLDOUT_STRICT_SAFE": base_train[~base_train["final_strict_lineage_id"].isin(test_sets["final_strict_lineage_id"])],
            "SOURCE_HOLDOUT_SCENE_SAFE": base_train[~base_train["final_split_block_id"].isin(test_sets["final_split_block_id"])],
            "SOURCE_HOLDOUT_AMBIGUITY_SAFE": base_train[~base_train["ambiguity_sensitive_block_id"].isin(test_sets["ambiguity_sensitive_block_id"])],
        }
        for design in SOURCE_HOLDOUT_DESIGNS:
            train = design_train[design]
            if set(train["label"]) != labels:
                raise RuntimeError(f"Training labels missing for {heldout}/{design}")
            assignments = pd.concat([
                train[["sample_id", "label"]].assign(role="TRAIN"),
                test[["sample_id", "label"]].assign(role="TEST"),
            ], ignore_index=True)
            path = out_dir / f"{heldout}__{design}.tsv.gz"
            write_tsv(assignments, path)
            index_rows.append({
                "analysis_family": "SOURCE_HOLDOUT",
                "task_name": "cacao_coarse_three_class",
                "evaluation_design": design,
                "heldout_source": heldout,
                "assignment_path": str(path),
                "assignment_sha256": sha256_file(path),
            })
            audit_rows.append({
                "analysis_family": "SOURCE_HOLDOUT",
                "heldout_source": heldout,
                "evaluation_design": design,
                "base_train_n": len(base_train),
                "train_n": len(train),
                "train_paths_removed": len(base_train) - len(train),
                "test_n": len(test),
                "exact_component_overlap": group_overlap(train, test, "exact_component_id"),
                "strict_lineage_overlap": group_overlap(train, test, "final_strict_lineage_id"),
                "verified_scene_block_overlap": group_overlap(train, test, "final_split_block_id"),
                "ambiguity_block_overlap": group_overlap(train, test, "ambiguity_sensitive_block_id"),
            })
    return pd.DataFrame(index_rows), pd.DataFrame(audit_rows)


def build_config_table(
    repeated_index: pd.DataFrame,
    fixed_index: pd.DataFrame,
    source_index: pd.DataFrame,
    expected_configs: int,
) -> pd.DataFrame:
    repeated_lookup = {
        (r.task_name, r.evaluation_design, int(r.repeat_index)): r.assignment_path
        for r in repeated_index.itertuples(index=False)
    }
    fixed_lookup = {r.evaluation_design: r.assignment_path for r in fixed_index.itertuples(index=False)}
    source_lookup = {
        (r.heldout_source, r.evaluation_design): r.assignment_path
        for r in source_index.itertuples(index=False)
    }
    rows: list[dict[str, object]] = []

    def add(
        analysis_family: str,
        task_name: str,
        evaluation_design: str,
        feature_set: str,
        training_mode: str,
        repeat_index: int = 0,
        split_seed: int = 0,
        heldout_source: str = "",
        assignment_path: str = "",
        notes: str = "",
    ) -> None:
        rows.append({
            "analysis_family": analysis_family,
            "task_name": task_name,
            "evaluation_design": evaluation_design,
            "feature_set": feature_set,
            "classifier": "logistic",
            "training_mode": training_mode,
            "repeat_index": repeat_index,
            "split_seed": split_seed,
            "heldout_source": heldout_source,
            "assignment_path": assignment_path,
            "n_folds": 5 if analysis_family == "REPEATED_CV" else 1,
            "primary_analysis": "YES" if feature_set == "efficientnet_b0" and training_mode == "ALL_PATHS" else "NO",
            "label_order": "|".join(TASK_LABEL_ORDERS[task_name]),
            "notes": notes,
        })

    # Repeated waterfall: primary encoder 20 seeds, concatenated robustness 10 seeds.
    for task_name in ["cacao_coarse_three_class", "cocoamonilia_four_stage"]:
        for design in CV_DESIGNS:
            for repeat_index, seed in enumerate(REPEAT_SEEDS, start=1):
                add("REPEATED_CV", task_name, design, "efficientnet_b0", "ALL_PATHS", repeat_index, seed, assignment_path=repeated_lookup[(task_name, design, repeat_index)], notes="Primary repeated-split waterfall")
            for repeat_index, seed in enumerate(ROBUST_REPEAT_SEEDS, start=1):
                add("REPEATED_CV", task_name, design, "concat", "ALL_PATHS", repeat_index, seed, assignment_path=repeated_lookup[(task_name, design, repeat_index)], notes="Two-encoder repeated-split robustness")

    # Training duplication sensitivity under the two conservative groupings.
    for task_name in ["cacao_coarse_three_class", "cocoamonilia_four_stage"]:
        for design in ["VERIFIED_SCENE_BLOCK_5FOLD", "AMBIGUITY_SENS_BLOCK_5FOLD"]:
            for mode in ["LINEAGE_WEIGHTED", "LINEAGE_REPRESENTATIVE"]:
                for repeat_index, seed in enumerate(REPEAT_SEEDS, start=1):
                    add("REPEATED_CV", task_name, design, "efficientnet_b0", mode, repeat_index, seed, assignment_path=repeated_lookup[(task_name, design, repeat_index)], notes="Lineage-balanced training sensitivity")

    # Corrected five-class task: all 9,132 final lineage freeze units, including CocoaMonilia stages as frosty_pod.
    for design in FIVECLASS_CV_DESIGNS:
        for mode in ["ALL_PATHS", "LINEAGE_WEIGHTED", "LINEAGE_REPRESENTATIVE"]:
            for repeat_index, seed in enumerate(REPEAT_SEEDS, start=1):
                add("REPEATED_CV", "cacao_causal_five_class", design, "efficientnet_b0", mode, repeat_index, seed, assignment_path=repeated_lookup[("cacao_causal_five_class", design, repeat_index)], notes="Corrected five-class grouped benchmark")
        for repeat_index, seed in enumerate(ROBUST_REPEAT_SEEDS, start=1):
            add("REPEATED_CV", "cacao_causal_five_class", design, "concat", "ALL_PATHS", repeat_index, seed, assignment_path=repeated_lookup[("cacao_causal_five_class", design, repeat_index)], notes="Corrected five-class two-encoder robustness")

    # Fixed public CocoaMonilia test; same test paths under progressively safer training exclusions.
    for design in COCO_TEST_DESIGNS:
        for feature in ["efficientnet_b0", "concat"]:
            add("COCO_FIXED_TEST", "cocoamonilia_four_stage", design, feature, "ALL_PATHS", assignment_path=fixed_lookup[design], notes="Public test set held fixed; only training exclusions change")
    for design in ["COCO_TEST_SCENE_SAFE", "COCO_TEST_AMBIGUITY_SAFE"]:
        for mode in ["LINEAGE_WEIGHTED", "LINEAGE_REPRESENTATIVE"]:
            add("COCO_FIXED_TEST", "cocoamonilia_four_stage", design, "efficientnet_b0", mode, assignment_path=fixed_lookup[design], notes="Fixed-test lineage-balanced training sensitivity")

    # Source holdout, adding exact-safe and ambiguity-safe training definitions.
    for heldout in SOURCE_HOLDOUTS:
        for design in SOURCE_HOLDOUT_DESIGNS:
            for feature in ["efficientnet_b0", "concat"]:
                add("SOURCE_HOLDOUT", "cacao_coarse_three_class", design, feature, "ALL_PATHS", heldout_source=heldout, assignment_path=source_lookup[(heldout, design)], notes="Held-out archive test paths fixed; safety definition changes training exclusions")
        for design in ["SOURCE_HOLDOUT_SCENE_SAFE", "SOURCE_HOLDOUT_AMBIGUITY_SAFE"]:
            for mode in ["LINEAGE_WEIGHTED", "LINEAGE_REPRESENTATIVE"]:
                add("SOURCE_HOLDOUT", "cacao_coarse_three_class", design, "efficientnet_b0", mode, heldout_source=heldout, assignment_path=source_lookup[(heldout, design)], notes="Source-held-out lineage-balanced training sensitivity")

    config = pd.DataFrame(rows)
    config.insert(0, "config_id", [f"S3B{i:04d}" for i in range(1, len(config) + 1)])
    config.insert(1, "array_index", np.arange(1, len(config) + 1))
    if len(config) != expected_configs:
        raise RuntimeError(f"Expected {expected_configs} configs, generated {len(config)}")
    if config.duplicated(["analysis_family", "task_name", "evaluation_design", "feature_set", "training_mode", "repeat_index", "heldout_source"]).any():
        raise RuntimeError("Duplicate design confirmation config definitions")
    return config


def main() -> int:
    args = parse_args()
    paths = ProjectPaths(Path(args.root), Path(args.big))
    for directory in [paths.prepared, paths.task_results, paths.aggregate, paths.figures]:
        directory.mkdir(parents=True, exist_ok=True)

    required = [
        paths.stage2e / "stage2e_claim_status.tsv",
        paths.stage2e / "stage2e_final_benchmark_manifest.tsv",
        paths.stage2e / "stage2e_final_split_blocks.tsv",
        paths.stage2e / "stage2e_reserve_ambiguous_manual_review.tsv",
        paths.stage3a / "stage3a_claim_status.tsv",
    ]
    for path in required:
        if not path.is_file():
            raise RuntimeError(f"Required input missing: {path}")
    stage2e_claim = read_tsv(required[0])
    for column in ["evidence_gated_reserve_exhausted", "conservative_strict_lineage_freeze_final", "conservative_split_block_freeze_final"]:
        if stage2e_claim.iloc[0].get(column, "") != "YES":
            raise RuntimeError(f"final lineage freeze prerequisite not met: {column}")
    stage3a_claim = read_tsv(required[-1])
    if stage3a_claim.iloc[0].get("all_model_configs_completed", "") != "YES":
        raise RuntimeError("frozen-feature benchmark prerequisite not complete")

    manifest = read_tsv(required[1])
    split_blocks = read_tsv(required[2])
    ambiguous = read_tsv(required[3])
    embedding_manifest, embedding_provenance, embedding_records = verify_embeddings(paths)
    ambiguity_map_df, ambiguity_edges, ambiguity_map = build_ambiguity_map(manifest, split_blocks, ambiguous)
    samples, scope_audit = build_path_samples(paths, manifest, embedding_manifest, ambiguity_map)
    repeated_index, repeated_audit = write_repeated_assignments(paths, samples)
    fixed_index, fixed_audit = make_fixed_assignments(paths, samples)
    source_index, source_audit = make_source_assignments(paths, samples)
    config = build_config_table(repeated_index, fixed_index, source_index, args.expected_configs)

    write_tsv(samples, paths.prepared / "stage3b_path_samples.tsv.gz")
    write_tsv(scope_audit, paths.prepared / "stage3b_task_scope_audit.tsv")
    write_tsv(ambiguity_map_df, paths.prepared / "stage3b_ambiguity_block_map.tsv")
    write_tsv(ambiguity_edges, paths.prepared / "stage3b_ambiguity_edge_audit.tsv")
    write_tsv(repeated_index, paths.prepared / "stage3b_repeated_assignment_index.tsv")
    write_tsv(repeated_audit, paths.prepared / "stage3b_repeated_split_integrity_audit.tsv")
    write_tsv(fixed_index, paths.prepared / "stage3b_fixed_assignment_index.tsv")
    write_tsv(fixed_audit, paths.prepared / "stage3b_fixed_test_integrity_audit.tsv")
    write_tsv(source_index, paths.prepared / "stage3b_source_assignment_index.tsv")
    write_tsv(source_audit, paths.prepared / "stage3b_source_holdout_integrity_audit.tsv")
    write_tsv(config, paths.prepared / "stage3b_config_table.tsv")
    write_tsv(pd.DataFrame({"repeat_index": range(1, 21), "split_seed": REPEAT_SEEDS}), paths.prepared / "stage3b_repeat_seeds.tsv")
    json_dump(TASK_LABEL_ORDERS, paths.prepared / "stage3b_label_orders.json")

    provenance = [
        {"item": "stage2e_manifest", "path": str(required[1]), "sha256": sha256_file(required[1]), "value": ""},
        {"item": "stage2e_split_blocks", "path": str(required[2]), "sha256": sha256_file(required[2]), "value": ""},
        {"item": "stage2e_ambiguous_pairs", "path": str(required[3]), "sha256": sha256_file(required[3]), "value": ""},
        {"item": "stage3a_claim_status", "path": str(required[-1]), "sha256": sha256_file(required[-1]), "value": ""},
        {"item": "stage2c_embedding_manifest", "path": str(paths.stage2c_meta / "stage2c_embedding_manifest.tsv"), "sha256": sha256_file(paths.stage2c_meta / "stage2c_embedding_manifest.tsv"), "value": ""},
        *embedding_records,
    ]
    write_tsv(pd.DataFrame(provenance), paths.prepared / "stage3b_prepare_provenance.tsv")

    summary = pd.DataFrame([
        {"metric": "model_configs", "value": len(config)},
        {"metric": "repeat_seeds_primary", "value": len(REPEAT_SEEDS)},
        {"metric": "repeat_seeds_concat", "value": len(ROBUST_REPEAT_SEEDS)},
        {"metric": "path_sample_rows_all_tasks", "value": len(samples)},
        {"metric": "corrected_fiveclass_analysis_units", "value": samples[samples["task_name"].eq("cacao_causal_five_class")]["analysis_unit_id"].nunique()},
        {"metric": "repeated_assignment_files", "value": len(repeated_index)},
        {"metric": "fixed_assignment_files", "value": len(fixed_index)},
        {"metric": "source_assignment_files", "value": len(source_index)},
        {"metric": "ambiguity_sensitive_blocks", "value": ambiguity_map_df["ambiguity_sensitive_block_id"].nunique()},
    ])
    write_tsv(summary, paths.prepared / "stage3b_prepare_summary.tsv")
    (paths.prepared / ".stage3b_prepared").touch()
    print(summary.to_string(index=False))
    print("\nConfigs by family:")
    print(config.groupby(["analysis_family", "task_name"]).size().rename("configs").reset_index().to_string(index=False))
    print("\nCorrected five-class scope:")
    print(scope_audit[scope_audit["task_name"].eq("cacao_causal_five_class")].to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

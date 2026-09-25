#!/usr/bin/env python3
"""manifest reconstruction.1: correct ontology/split semantics and summarize exact lineage.

This script is intentionally post-processing only. It reuses the immutable manifest reconstruction
file and pixel hashes and does not reread image payloads.
"""
from __future__ import annotations

import argparse
import hashlib
import math
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Iterable

import pandas as pd


CM_ARCHIVE = "CocoaMoniliaDataSet.zip"
GH_ARCHIVE = "Cocoa_Disease_Gh.rar"
SPANISH_ARCHIVE = "enfermedades-cacao-yolov4.rar"
ROBO_ARCHIVE = "coffee and cocoa.v2i.folder.zip"
BLACKPOD_ARCHIVE = "Black Pod rot and pod borer on cocoa pod three classes.zip"
CACAO_DISEASES_ARCHIVE = "Cacao Diseases.zip"

ARCHIVE_IDS = {
    BLACKPOD_ARCHIVE: "fig_blackpod_borer",
    CACAO_DISEASES_ARCHIVE: "fig_cacao_diseases",
    GH_ARCHIVE: "fig_ghana_balanced",
    ROBO_ARCHIVE: "fig_roboflow_mixed",
    SPANISH_ARCHIVE: "fig_spanish_yolov4",
    CM_ARCHIVE: "zen_cocoamonilia",
}

LABEL_INFO = {
    "healthy": ("healthy", "none/asymptomatic"),
    "black_pod": ("oomycete_disease", "Phytophthora spp."),
    "frosty_pod": ("fungal_disease", "Moniliophthora roreri"),
    "frosty_pod_stage_m1": ("disease_stage", "Moniliophthora roreri: hump stage"),
    "frosty_pod_stage_m2": ("disease_stage", "Moniliophthora roreri: oily/brown-spot stage"),
    "frosty_pod_stage_m3": ("disease_stage", "Moniliophthora roreri: sporulation stage"),
    "pod_borer_damage": ("non_disease_damage", "insect damage"),
    "mirid_damage": ("non_disease_damage", "mirid injury"),
    "coffee_healthy": ("non_cacao", "coffee image"),
    "unknown": ("unresolved", "unresolved"),
}


def write_tsv(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, sep="\t", index=False, na_rep="")


def safe(value: object) -> str:
    if value is None:
        return ""
    if isinstance(value, float) and math.isnan(value):
        return ""
    return str(value)


def normalize_path(value: str) -> str:
    value = value.strip().replace("\\", "/")
    value = re.sub(r"^\./+", "", value)
    return re.sub(r"/+", "/", value).lower()


def corrected_label(row: pd.Series) -> tuple[str, str]:
    archive = safe(row["source_archive"])
    rel = "/" + safe(row["relative_path"]).lower().replace("\\", "/").strip("/") + "/"
    old = safe(row.get("harmonized_label", "unknown")) or "unknown"

    if archive == GH_ARCHIVE:
        if "/moni/" in rel:
            return "frosty_pod", "archive_path_rule:Moni"
        if "/phyto/" in rel:
            return "black_pod", "archive_path_rule:Phyto"
        if "/healthy/" in rel:
            return "healthy", "archive_path_rule:healthy"

    if archive == SPANISH_ARCHIVE:
        if "/fito/" in rel:
            return "black_pod", "archive_path_rule:fito"
        if "/monilia/" in rel:
            return "frosty_pod", "archive_path_rule:monilia"
        if "/healthy/" in rel:
            return "healthy", "archive_path_rule:healthy"

    return old, "stage1_inference"


def source_origin_key(row: pd.Series) -> tuple[str, str]:
    archive = safe(row["source_archive"])
    filename = safe(row["file_name"]).lower()
    label = safe(row["corrected_label"])
    stem = Path(filename).stem

    match = re.match(r"(.+?)_(?:jpg|jpeg|png)\.rf\.[^.]+\.(?:jpg|jpeg|png)$", filename)
    if archive == ROBO_ARCHIVE and match:
        # Label deliberately omitted: the same origin can appear under conflicting classes.
        return f"roboflow_origin:{match.group(1)}", "roboflow_label_agnostic"

    match = re.match(r"(black_pod_rot|pod_borer|healthy)[_-]0*(\d+)$", stem)
    if archive in {BLACKPOD_ARCHIVE, CACAO_DISEASES_ARCHIVE} and match:
        return f"class_number:{match.group(1)}:{int(match.group(2))}", "cross_archive_class_number"

    if archive == CM_ARCHIVE:
        return f"cocoamonilia_basename:{filename}", "cocoamonilia_basename"

    normalized = re.sub(r"[^a-z0-9]+", "_", stem).strip("_")
    return f"archive_basename:{ARCHIVE_IDS.get(archive, archive)}:{normalized}", "archive_scoped_basename"


def build_split_map(raw_dir: Path) -> tuple[pd.DataFrame, dict[str, str]]:
    rows: list[dict[str, object]] = []
    path_to_split: dict[str, str] = {}
    for filename, split in (("train.txt", "train"), ("val.txt", "validation"), ("test.txt", "test")):
        candidates = sorted(raw_dir.rglob(filename))
        if not candidates:
            raise FileNotFoundError(f"Missing external split file: {filename} under {raw_dir}")
        # The Zenodo record has one record-level file of each name.
        path = candidates[0]
        for line_number, line in enumerate(path.read_text(encoding="utf-8", errors="replace").splitlines(), 1):
            raw = line.strip()
            if not raw or raw.startswith("#"):
                continue
            key = normalize_path(raw)
            old = path_to_split.get(key)
            if old and old != split:
                raise RuntimeError(f"Same full path appears in multiple split files: {raw}: {old}, {split}")
            path_to_split[key] = split
            rows.append(
                {
                    "split_file": str(path),
                    "split": split,
                    "line_number": line_number,
                    "raw_entry": raw,
                    "normalized_full_path": key,
                    "basename": Path(raw.replace("\\", "/")).name,
                }
            )
    return pd.DataFrame(rows), path_to_split


def bool_yes(series: pd.Series) -> pd.Series:
    return series.astype(str).str.upper().eq("YES")


def set_join(values: Iterable[object]) -> str:
    vals = sorted({safe(v) for v in values if safe(v) and safe(v) != "unspecified"})
    return "|".join(vals)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage1-dir", required=True, type=Path)
    parser.add_argument("--raw-dir", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()

    out = args.output_dir
    out.mkdir(parents=True, exist_ok=True)

    photo_path = args.stage1_dir / "stage1_photo_manifest.tsv"
    coco_path = args.stage1_dir / "stage1_coco_images.tsv"
    if not photo_path.is_file():
        raise FileNotFoundError(photo_path)
    if not coco_path.is_file():
        raise FileNotFoundError(coco_path)

    photos = pd.read_csv(photo_path, sep="\t", low_memory=False, keep_default_na=False)
    coco = pd.read_csv(coco_path, sep="\t", low_memory=False, keep_default_na=False)

    split_entries, split_map = build_split_map(args.raw_dir)
    write_tsv(split_entries, out / "stage1_1_external_split_paths.tsv")

    labels = photos.apply(corrected_label, axis=1, result_type="expand")
    photos["corrected_label"] = labels[0]
    photos["label_rule"] = labels[1]
    photos["source_dataset_id"] = photos["source_archive"].map(ARCHIVE_IDS).fillna(photos["source_archive"])

    origin = photos.apply(source_origin_key, axis=1, result_type="expand")
    photos["source_origin_key"] = origin[0]
    photos["source_origin_rule"] = origin[1]

    photos["corrected_split"] = photos["source_split"].replace("", "unspecified")
    cm_mask = photos["source_archive"].eq(CM_ARCHIVE)
    cm_keys = photos.loc[cm_mask, "relative_path"].map(normalize_path)
    photos.loc[cm_mask, "corrected_split"] = cm_keys.map(split_map).fillna("UNMATCHED")

    # Full-path split reconciliation must be exact.
    cm_paths = set(cm_keys)
    split_paths = set(split_map)
    split_only = sorted(split_paths - cm_paths)
    photo_only = sorted(cm_paths - split_paths)
    if split_only or photo_only:
        mismatch = pd.DataFrame(
            [{"side": "split_only", "normalized_path": x} for x in split_only]
            + [{"side": "photo_only", "normalized_path": x} for x in photo_only]
        )
        write_tsv(mismatch, out / "stage1_1_external_split_mismatches.tsv")
    else:
        write_tsv(pd.DataFrame(columns=["side", "normalized_path"]), out / "stage1_1_external_split_mismatches.tsv")

    # Stable exact-component IDs.
    unique_hashes = sorted(x for x in photos["sha256"].unique() if safe(x))
    hash_to_component = {h: f"EXACTC{i:06d}" for i, h in enumerate(unique_hashes, 1)}
    photos["exact_component_id"] = photos["sha256"].map(hash_to_component).fillna("NO_HASH")

    component_rows: list[dict[str, object]] = []
    for component_id, group in photos.groupby("exact_component_id", sort=True):
        archives = sorted(set(group["source_archive"]))
        labels_set = sorted(set(group["corrected_label"]))
        splits = sorted(
            {
                x
                for x in group["corrected_split"]
                if x not in {"", "unspecified", "UNMATCHED"}
            }
        )
        paths = sorted(group["relative_path"])
        component_rows.append(
            {
                "exact_component_id": component_id,
                "sha256": group["sha256"].iloc[0],
                "component_size": len(group),
                "archives": "|".join(archives),
                "archive_count": len(archives),
                "labels": "|".join(labels_set),
                "label_count": len(labels_set),
                "splits": "|".join(splits),
                "split_count": len(splits),
                "cross_archive": "YES" if len(archives) > 1 else "NO",
                "label_conflict": "YES" if len(labels_set) > 1 else "NO",
                "cross_split": "YES" if len(splits) > 1 else "NO",
                "representative_archive": group.sort_values(["source_archive", "relative_path"])["source_archive"].iloc[0],
                "representative_path": group.sort_values(["source_archive", "relative_path"])["relative_path"].iloc[0],
                "member_paths": "|".join(paths),
            }
        )
    components = pd.DataFrame(component_rows)
    write_tsv(components, out / "stage1_1_exact_components.tsv")

    comp_flags = components.set_index("exact_component_id")[
        ["component_size", "cross_archive", "label_conflict", "cross_split", "representative_path"]
    ]
    photos = photos.join(comp_flags, on="exact_component_id", rsuffix="_component")
    photos["exact_representative"] = (
        photos["relative_path"].eq(photos["representative_path"]).map({True: "YES", False: "NO"})
    )

    label_types = photos["corrected_label"].map(lambda x: LABEL_INFO.get(x, ("unresolved", "unresolved"))[0])
    label_agents = photos["corrected_label"].map(lambda x: LABEL_INFO.get(x, ("unresolved", "unresolved"))[1])
    photos["biological_type"] = label_types
    photos["causal_agent_or_damage"] = label_agents

    supervised_labels = {
        "healthy", "black_pod", "frosty_pod",
        "frosty_pod_stage_m1", "frosty_pod_stage_m2", "frosty_pod_stage_m3",
    }
    reasons: list[str] = []
    eligible: list[str] = []
    for _, row in photos.iterrows():
        reason_parts = []
        if safe(row["image_read_ok"]).upper() != "YES":
            reason_parts.append("unreadable_image")
        if safe(row["is_cacao"]).upper() != "YES":
            reason_parts.append("non_cacao")
        if safe(row["corrected_label"]) not in supervised_labels:
            reason_parts.append("out_of_scope_or_unresolved_label")
        if safe(row["label_conflict"]).upper() == "YES":
            reason_parts.append("exact_component_label_conflict")
        if safe(row["corrected_split"]) == "UNMATCHED":
            reason_parts.append("split_path_unmatched")
        reasons.append("|".join(reason_parts))
        eligible.append("YES" if not reason_parts else "NO")
    photos["supervised_eligible"] = eligible
    photos["exclusion_reason"] = reasons

    # Use a stable path-level ID that never depends on row order.
    photos["photo_path_id"] = photos.apply(
        lambda r: "PHOTO_" + hashlib.sha256(
            (safe(r["source_archive"]) + "\n" + safe(r["relative_path"])).encode("utf-8")
        ).hexdigest()[:20],
        axis=1,
    )

    write_tsv(photos, out / "stage1_1_canonical_photo_manifest.tsv")

    # Exact conflict tables.
    label_conflicts = components[components["label_conflict"].eq("YES")].copy()
    split_leakage = components[components["cross_split"].eq("YES")].copy()
    cross_archive = components[components["cross_archive"].eq("YES")].copy()
    write_tsv(label_conflicts, out / "stage1_1_exact_label_conflicts.tsv")
    write_tsv(split_leakage, out / "stage1_1_exact_split_leakage.tsv")
    write_tsv(cross_archive, out / "stage1_1_exact_cross_archive_components.tsv")

    # Pairwise source overlap.
    archive_hashes = {
        archive: set(group["sha256"]) - {""}
        for archive, group in photos.groupby("source_archive")
    }
    overlap_rows: list[dict[str, object]] = []
    archives = sorted(archive_hashes)
    for i, archive_a in enumerate(archives):
        for archive_b in archives[i + 1 :]:
            shared = archive_hashes[archive_a] & archive_hashes[archive_b]
            overlap_rows.append(
                {
                    "archive_a": archive_a,
                    "archive_b": archive_b,
                    "unique_hashes_a": len(archive_hashes[archive_a]),
                    "unique_hashes_b": len(archive_hashes[archive_b]),
                    "shared_exact_hashes": len(shared),
                    "pct_a_shared": 100 * len(shared) / max(1, len(archive_hashes[archive_a])),
                    "pct_b_shared": 100 * len(shared) / max(1, len(archive_hashes[archive_b])),
                }
            )
    overlap = pd.DataFrame(overlap_rows)
    write_tsv(overlap, out / "stage1_1_pairwise_source_overlap.tsv")

    # Directional nominal leave-one-source-out contamination.
    lodo_rows: list[dict[str, object]] = []
    for heldout, group in photos.groupby("source_archive"):
        other_hashes = set(photos.loc[photos["source_archive"].ne(heldout), "sha256"]) - {""}
        contaminated = group["sha256"].isin(other_hashes)
        lodo_rows.append(
            {
                "heldout_archive": heldout,
                "scope": "all_physical_photo_paths",
                "heldout_photo_paths": len(group),
                "heldout_unique_exact_hashes": group["sha256"].nunique(),
                "paths_with_exact_counterpart_in_other_archive": int(contaminated.sum()),
                "pct_paths_contaminated": 100 * contaminated.mean(),
                "unique_hashes_shared_with_other_archive": len(set(group["sha256"]) & other_hashes),
                "pct_unique_hashes_shared": 100 * len(set(group["sha256"]) & other_hashes) / max(1, group["sha256"].nunique()),
            }
        )

    # COCO-declared BlackPod subset: exclude the three stray high-resolution files.
    bp_coco = coco[coco["source_archive"].eq(BLACKPOD_ARCHIVE)]
    declared_paths = {
        p
        for cell in bp_coco["matched_paths"]
        for p in safe(cell).split("|")
        if p
    }
    bp_declared = photos[
        photos["source_archive"].eq(BLACKPOD_ARCHIVE)
        & photos["relative_path"].isin(declared_paths)
    ]
    other_hashes = set(photos.loc[photos["source_archive"].ne(BLACKPOD_ARCHIVE), "sha256"]) - {""}
    contaminated = bp_declared["sha256"].isin(other_hashes)
    lodo_rows.append(
        {
            "heldout_archive": BLACKPOD_ARCHIVE,
            "scope": "COCO_declared_2436_images",
            "heldout_photo_paths": len(bp_declared),
            "heldout_unique_exact_hashes": bp_declared["sha256"].nunique(),
            "paths_with_exact_counterpart_in_other_archive": int(contaminated.sum()),
            "pct_paths_contaminated": 100 * contaminated.mean(),
            "unique_hashes_shared_with_other_archive": len(set(bp_declared["sha256"]) & other_hashes),
            "pct_unique_hashes_shared": 100 * len(set(bp_declared["sha256"]) & other_hashes) / max(1, bp_declared["sha256"].nunique()),
        }
    )
    lodo = pd.DataFrame(lodo_rows)
    write_tsv(lodo, out / "stage1_1_nominal_lodo_contamination.tsv")

    # Archive-level corrected QC.
    archive_rows: list[dict[str, object]] = []
    for archive, group in photos.groupby("source_archive"):
        comp = components[components["exact_component_id"].isin(group["exact_component_id"])]
        archive_rows.append(
            {
                "source_archive": archive,
                "source_dataset_id": ARCHIVE_IDS.get(archive, archive),
                "photo_paths": len(group),
                "cacao_photo_paths": int(bool_yes(group["is_cacao"]).sum()),
                "readable_photo_paths": int(bool_yes(group["image_read_ok"]).sum()),
                "unique_exact_hashes": group["sha256"].nunique(),
                "exact_excess_copies": len(group) - group["sha256"].nunique(),
                "exact_duplicate_rate_pct": 100 * (len(group) - group["sha256"].nunique()) / max(1, len(group)),
                "label_conflict_components": int(comp["label_conflict"].eq("YES").sum()),
                "cross_split_components": int(comp["cross_split"].eq("YES").sum()),
                "corrected_labels": set_join(group["corrected_label"]),
                "corrected_splits": set_join(group["corrected_split"]),
            }
        )
    archive_qc = pd.DataFrame(archive_rows)
    write_tsv(archive_qc, out / "stage1_1_archive_qc.tsv")

    # CocoaMonilia COCO reconciliation.
    cm_coco = coco[coco["source_archive"].eq(CM_ARCHIVE)].copy()
    cm_coco["basename_lower"] = cm_coco["basename"].str.lower()
    cm_coco["annotation_class"] = cm_coco["annotation_path"].str.extract(r"instances_(h0|m1|m2|m3)\.json", expand=False)
    basename_counts = cm_coco.groupby("basename_lower").agg(
        coco_records=("basename_lower", "size"),
        annotation_classes=("annotation_class", set_join),
    ).reset_index()
    duplicate_coco_basenames = basename_counts[basename_counts["coco_records"].gt(1)]
    cross_class_coco = duplicate_coco_basenames[
        duplicate_coco_basenames["annotation_classes"].str.contains(r"\|", regex=True)
    ]
    same_class_duplicate_records = duplicate_coco_basenames[
        ~duplicate_coco_basenames["annotation_classes"].str.contains(r"\|", regex=True)
    ]

    cm_physical = photos[photos["source_archive"].eq(CM_ARCHIVE)]
    cm_audit = pd.DataFrame(
        [
            {"metric": "physical_photo_paths", "value": len(cm_physical)},
            {"metric": "unique_physical_basenames", "value": cm_physical["file_name"].str.lower().nunique()},
            {"metric": "unique_exact_hashes", "value": cm_physical["sha256"].nunique()},
            {"metric": "COCO_image_records", "value": len(cm_coco)},
            {"metric": "unique_COCO_basenames", "value": cm_coco["basename_lower"].nunique()},
            {"metric": "COCO_basenames_with_multiple_records", "value": len(duplicate_coco_basenames)},
            {"metric": "COCO_cross_class_duplicate_basenames", "value": len(cross_class_coco)},
            {"metric": "COCO_same_class_duplicate_basenames", "value": len(same_class_duplicate_records)},
            {"metric": "exact_label_conflict_components", "value": int(
                components[
                    components["exact_component_id"].isin(cm_physical["exact_component_id"])
                    & components["label_conflict"].eq("YES")
                ].shape[0]
            )},
            {"metric": "exact_cross_split_components", "value": int(
                components[
                    components["exact_component_id"].isin(cm_physical["exact_component_id"])
                    & components["cross_split"].eq("YES")
                ].shape[0]
            )},
            {"metric": "external_split_full_paths", "value": len(split_entries)},
            {"metric": "external_split_paths_unmatched", "value": len(split_only) + len(photo_only)},
        ]
    )
    write_tsv(cm_audit, out / "stage1_1_cocoamonilia_audit.tsv")
    write_tsv(duplicate_coco_basenames, out / "stage1_1_cocoamonilia_duplicate_coco_records.tsv")

    # Corrected label counts.
    label_counts = (
        photos.groupby(["source_archive", "corrected_label"], as_index=False)
        .size()
        .rename(columns={"size": "photo_paths"})
    )
    write_tsv(label_counts, out / "stage1_1_corrected_label_counts.tsv")

    # Key finding table for downstream manuscript-oriented stages.
    shared_bp = int(
        overlap.loc[
            ((overlap["archive_a"] == BLACKPOD_ARCHIVE) & (overlap["archive_b"] == CACAO_DISEASES_ARCHIVE))
            | ((overlap["archive_b"] == BLACKPOD_ARCHIVE) & (overlap["archive_a"] == CACAO_DISEASES_ARCHIVE)),
            "shared_exact_hashes",
        ].sum()
    )
    findings = pd.DataFrame(
        [
            {"finding_id": "F01", "finding": "Total physical photograph paths", "value": len(photos), "claim_level": "exact"},
            {"finding_id": "F02", "finding": "Cacao photograph paths", "value": int(bool_yes(photos["is_cacao"]).sum()), "claim_level": "exact"},
            {"finding_id": "F03", "finding": "Unique exact file hashes", "value": photos["sha256"].nunique(), "claim_level": "exact"},
            {"finding_id": "F04", "finding": "Exact duplicate components", "value": int(components["component_size"].gt(1).sum()), "claim_level": "exact"},
            {"finding_id": "F05", "finding": "Cross-archive exact duplicate components", "value": int(components["cross_archive"].eq("YES").sum()), "claim_level": "exact"},
            {"finding_id": "F06", "finding": "Exact components with conflicting labels", "value": int(components["label_conflict"].eq("YES").sum()), "claim_level": "exact"},
            {"finding_id": "F07", "finding": "Exact components crossing official splits", "value": int(components["cross_split"].eq("YES").sum()), "claim_level": "exact"},
            {"finding_id": "F08", "finding": "Exact hashes shared by BlackPod COCO archive and Cacao Diseases", "value": shared_bp, "claim_level": "exact"},
            {"finding_id": "F09", "finding": "CocoaMonilia exact label-conflict components", "value": int(cm_audit.loc[cm_audit.metric.eq("exact_label_conflict_components"), "value"].iloc[0]), "claim_level": "exact"},
            {"finding_id": "F10", "finding": "CocoaMonilia exact cross-split components", "value": int(cm_audit.loc[cm_audit.metric.eq("exact_cross_split_components"), "value"].iloc[0]), "claim_level": "exact"},
        ]
    )
    write_tsv(findings, out / "stage1_1_key_findings.tsv")

    summary_lines = [
        "# manifest reconstruction.1 corrected canonical exact-lineage audit",
        "",
        f"- Physical photograph paths: {len(photos):,}",
        f"- Cacao photograph paths: {int(bool_yes(photos['is_cacao']).sum()):,}",
        f"- Unique exact file hashes: {photos['sha256'].nunique():,}",
        f"- Exact duplicate components: {int(components['component_size'].gt(1).sum()):,}",
        f"- Cross-archive exact duplicate components: {int(components['cross_archive'].eq('YES').sum()):,}",
        f"- Exact label-conflict components: {int(components['label_conflict'].eq('YES').sum()):,}",
        f"- Exact cross-split components: {int(components['cross_split'].eq('YES').sum()):,}",
        f"- BlackPod/Cacao Diseases shared exact hashes: {shared_bp:,}",
        "",
        "## Corrected ontology",
        "",
    ]
    for row in label_counts.groupby("corrected_label", as_index=False)["photo_paths"].sum().sort_values("photo_paths", ascending=False).itertuples():
        summary_lines.append(f"- {row.corrected_label}: {row.photo_paths:,}")
    summary_lines.extend(
        [
            "",
            "## Critical correction to manifest reconstruction",
            "",
            "- Ghana Moni and Phyto folders are now mapped to frosty_pod and black_pod.",
            "- The Spanish fito folder is now mapped to black_pod.",
            "- CocoaMonilia split assignment now uses the complete relative path, preserving spaces.",
            "- Cross-split leakage is computed from exact-hash components and atomic split labels.",
            "- Roboflow origin families are label-agnostic so cross-label reuse is not hidden.",
            "",
            "## Claim boundary",
            "",
            "These outputs establish exact file reuse, exact decoded-image reuse already recorded in manifest reconstruction,",
            "exact label contradictions, and exact split crossing. Perceptual similarity remains a candidate",
            "signal until geometric verification.",
            "",
        ]
    )
    (out / "stage1_1_summary.md").write_text("\n".join(summary_lines), encoding="utf-8")
    (out / ".stage2a_canonical_ready").write_text("OK\n", encoding="utf-8")
    print("\n".join(summary_lines))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"FATAL: {type(exc).__name__}: {exc}", file=sys.stderr)
        raise

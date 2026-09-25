#!/usr/bin/env python3
"""Build the manifest reconstruction canonical file/photo manifest and exact-lineage audit.

This stage deliberately stops before perceptual hashing and model benchmarking.
It establishes which extracted files are biological photographs, masks,
annotations, metadata, or non-cacao material; parses available annotation/split
structures; and identifies exact-file, exact-decoded-pixel, and filename-family
lineage groups.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import re
import sys
import time
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Iterable

from PIL import ExifTags, Image, UnidentifiedImageError


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}
ANNOTATION_EXTENSIONS = {".json", ".xml", ".txt", ".csv", ".tsv", ".yaml", ".yml", ".pbtxt"}
MASK_TOKENS = {
    "mask",
    "masks",
    "mask_segmentation",
    "segmentationclass",
    "segmentationobject",
    "segmentation_class",
    "segmentation_object",
}
SPLIT_MAP = {
    "train": "train",
    "training": "train",
    "val": "validation",
    "valid": "validation",
    "validation": "validation",
    "test": "test",
    "testing": "test",
}
RAW_TO_HARMONIZED = {
    "healthy": "healthy",
    "black_pod_rot": "black_pod",
    "black_pod": "black_pod",
    "phytophthora": "black_pod",
    "frosty_pod": "frosty_pod",
    "monilia": "frosty_pod",
    "moniliophthora": "frosty_pod",
    "monilia_stage_m1": "frosty_pod_stage_m1",
    "monilia_stage_m2": "frosty_pod_stage_m2",
    "monilia_stage_m3": "frosty_pod_stage_m3",
    "pod_borer": "pod_borer_damage",
    "mirid_damage": "mirid_damage",
    "coffee_normal": "coffee_healthy",
    "unknown": "unknown",
}


def now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%S%z")


def safe_string(value: Any) -> str:
    return str(value).encode("utf-8", "replace").decode("utf-8")


def read_tsv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8", errors="replace") as handle:
        return list(csv.DictReader(handle, delimiter="\t"))


def write_tsv(path: Path, rows: Iterable[dict[str, Any]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", newline="", encoding="utf-8", errors="replace") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, delimiter="\t", extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({field: safe_string(row.get(field, "")) for field in fields})
    tmp.replace(path)


def sha256_file(path: Path, chunk_size: int = 4 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def relative_context(path: Path, archive_root: Path) -> str:
    return safe_string(path.relative_to(archive_root).as_posix())


def infer_split(relative_path: str) -> str:
    parts = [p.lower() for p in Path(relative_path).parts]
    for part in parts:
        if part in SPLIT_MAP:
            return SPLIT_MAP[part]
    return "unspecified"


def infer_raw_label(relative_path: str) -> str:
    lower = relative_path.lower().replace("\\", "/")
    parts = [part.lower() for part in Path(lower).parts]
    filename = Path(lower).name

    exact_part_map = {
        "coffee_normal": "coffee_normal",
        "cocoa_blackpod": "black_pod_rot",
        "cocoa_frostypod": "frosty_pod",
        "cocoa_mirid": "mirid_damage",
        "cocoa_normal": "healthy",
        "black_pod_rot": "black_pod_rot",
        "blackpod": "black_pod_rot",
        "pod_borer": "pod_borer",
        "healthy": "healthy",
        "h0": "healthy",
        "m1": "monilia_stage_m1",
        "m2": "monilia_stage_m2",
        "m3": "monilia_stage_m3",
    }
    for part in parts:
        if part in exact_part_map:
            return exact_part_map[part]

    prefix_map = [
        ("black_pod_rot_", "black_pod_rot"),
        ("pod_borer_", "pod_borer"),
        ("healthy_", "healthy"),
    ]
    for prefix, label in prefix_map:
        if filename.startswith(prefix):
            return label

    if any(token in lower for token in ("phytophthora", "black_pod", "blackpod", "mazorca_negra")):
        return "phytophthora"
    if any(token in lower for token in ("moniliophthora", "monilia", "frosty", "moniliasis")):
        return "monilia"
    if any(token in lower for token in ("pod_borer", "borer", "carmenta")):
        return "pod_borer"
    if "mirid" in lower:
        return "mirid_damage"
    if any(token in lower for token in ("/healthy", "/normal", "/sana", "/sano")):
        return "healthy"
    return "unknown"


def classify_role(relative_path: str, extension: str) -> tuple[str, str]:
    lower = relative_path.lower().replace("\\", "/")
    parts = [p.lower() for p in Path(lower).parts]
    filename = Path(lower).name

    if filename.startswith(".stage1_extract_complete"):
        return "extraction_marker", "stage1 completion marker"
    if extension in ANNOTATION_EXTENSIONS:
        if filename.startswith("readme") or "labelmap" in filename:
            return "metadata", "README or label-map text"
        return "annotation", f"annotation extension {extension}"
    if extension in IMAGE_EXTENSIONS:
        is_mask = (
            "___fuse" in filename
            or any(token in lower for token in MASK_TOKENS)
            or any(part.startswith("segmentationclass") for part in parts)
            or any(part.startswith("segmentationobject") for part in parts)
        )
        if is_mask:
            return "mask", "mask/segmentation path or filename"
        if "coffee_normal" in parts:
            return "photo_non_cacao", "coffee class in mixed coffee/cacao archive"
        return "photo_candidate", "image extension outside mask/segmentation paths"
    return "other", "unclassified non-image file"


def family_hint(filename: str, raw_label: str, source_archive: str) -> tuple[str, str]:
    lower = filename.lower()
    # Roboflow export/augmentation descendants share the prefix before
    # _jpg.rf.<hash>.jpg (or corresponding png/jpeg variants).
    match = re.match(r"(.+?)_(?:jpg|jpeg|png)\.rf\.[^.]+\.(?:jpg|jpeg|png)$", lower)
    if match:
        base = match.group(1)
        return f"roboflow:{raw_label}:{base}", "roboflow_export_family"

    stem = Path(lower).stem
    match = re.match(r"(black_pod_rot|pod_borer|healthy)[_-]0*(\d+)$", stem)
    if match:
        label = match.group(1)
        number = int(match.group(2))
        return f"class_number:{label}:{number}", "normalized_class_number"

    # A timestamp-like basename is useful within CocoaMonilia but should not
    # collide automatically with unrelated archives.
    if re.fullmatch(r"\d{10,}", stem):
        return f"timestamp:{source_archive}:{raw_label}:{stem}", "timestamp_basename"

    normalized = re.sub(r"[^a-z0-9]+", "_", stem).strip("_")
    return f"basename:{source_archive}:{raw_label}:{normalized}", "archive_scoped_basename"


def exif_value(exif: Any, tag_name: str) -> str:
    tag_id = next((key for key, value in ExifTags.TAGS.items() if value == tag_name), None)
    if tag_id is None:
        return ""
    try:
        value = exif.get(tag_id, "")
    except Exception:
        return ""
    if isinstance(value, bytes):
        return value.decode("utf-8", "replace")
    return safe_string(value)


def inspect_file(task: dict[str, Any]) -> dict[str, Any]:
    path: Path = task["absolute_path_obj"]
    row = {key: value for key, value in task.items() if key != "absolute_path_obj"}
    try:
        row["sha256"] = sha256_file(path)
    except Exception as exc:
        row["sha256"] = ""
        row["file_error"] = f"sha256:{type(exc).__name__}:{exc}"
        return row

    row.update(
        {
            "image_read_ok": "",
            "image_format": "",
            "width": "",
            "height": "",
            "mode": "",
            "pixel_sha256": "",
            "exif_make": "",
            "exif_model": "",
            "exif_datetime_original": "",
            "image_error": "",
        }
    )
    if row["role"] not in {"photo_candidate", "photo_non_cacao", "mask"}:
        return row

    try:
        with Image.open(path) as image:
            row["image_format"] = safe_string(image.format or "")
            row["width"], row["height"] = image.size
            row["mode"] = safe_string(image.mode)
            exif = image.getexif()
            row["exif_make"] = exif_value(exif, "Make")
            row["exif_model"] = exif_value(exif, "Model")
            row["exif_datetime_original"] = exif_value(exif, "DateTimeOriginal")
            # Force a complete decode so truncated/corrupt payloads fail here.
            image.load()
            if row["role"] in {"photo_candidate", "photo_non_cacao"}:
                rgb = image.convert("RGB")
                digest = hashlib.sha256()
                digest.update(f"{rgb.width}x{rgb.height}|RGB|".encode("ascii"))
                digest.update(rgb.tobytes())
                row["pixel_sha256"] = digest.hexdigest()
        row["image_read_ok"] = "YES"
    except Exception as exc:
        row["image_read_ok"] = "NO"
        row["image_error"] = f"{type(exc).__name__}:{exc}"
    return row


def parse_coco_json(
    path: Path,
    source: str,
    archive_name: str,
    relative_path: str,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    summary = {
        "source": source,
        "source_archive": archive_name,
        "annotation_path": relative_path,
        "annotation_format": "json_other",
        "image_entries": 0,
        "annotation_entries": 0,
        "category_count": 0,
        "categories": "",
        "parse_status": "OK",
        "message": "",
    }
    image_rows: list[dict[str, Any]] = []
    try:
        data = json.loads(path.read_text(encoding="utf-8", errors="replace"))
        if not isinstance(data, dict) or "images" not in data:
            summary["message"] = "JSON does not contain a COCO images collection."
            return summary, image_rows
        images = data.get("images", []) or []
        annotations = data.get("annotations", []) or []
        categories = data.get("categories", []) or []
        category_names = {
            str(item.get("id", "")): safe_string(item.get("name", ""))
            for item in categories
            if isinstance(item, dict)
        }
        annotation_counts: Counter[str] = Counter()
        category_by_image: defaultdict[str, set[str]] = defaultdict(set)
        for annotation in annotations:
            if not isinstance(annotation, dict):
                continue
            image_id = str(annotation.get("image_id", ""))
            annotation_counts[image_id] += 1
            category_id = str(annotation.get("category_id", ""))
            if category_id:
                category_by_image[image_id].add(category_id)

        summary.update(
            {
                "annotation_format": "COCO",
                "image_entries": len(images),
                "annotation_entries": len(annotations),
                "category_count": len(categories),
                "categories": "|".join(
                    f"{key}:{value}" for key, value in sorted(category_names.items())
                ),
            }
        )
        for item in images:
            if not isinstance(item, dict):
                continue
            image_id = str(item.get("id", ""))
            category_ids = sorted(category_by_image.get(image_id, set()))
            image_rows.append(
                {
                    "source": source,
                    "source_archive": archive_name,
                    "annotation_path": relative_path,
                    "image_id": image_id,
                    "file_name": safe_string(item.get("file_name", "")),
                    "basename": Path(safe_string(item.get("file_name", ""))).name,
                    "declared_width": item.get("width", ""),
                    "declared_height": item.get("height", ""),
                    "annotation_count": annotation_counts.get(image_id, 0),
                    "category_ids": "|".join(category_ids),
                    "category_names": "|".join(
                        category_names.get(category_id, "") for category_id in category_ids
                    ),
                }
            )
    except Exception as exc:
        summary["parse_status"] = "FAILED"
        summary["message"] = f"{type(exc).__name__}:{exc}"
    return summary, image_rows


def parse_external_split_line(line: str) -> str:
    stripped = line.strip()
    if not stripped or stripped.startswith("#"):
        return ""
    match = re.search(r"([^\s,;]+\.(?:jpg|jpeg|png|bmp|tif|tiff|webp))", stripped, flags=re.I)
    if match:
        return Path(match.group(1).replace("\\", "/")).name
    return Path(stripped.replace("\\", "/")).name


def group_members(
    rows: list[dict[str, Any]],
    key_field: str,
    prefix: str,
    output_path: Path,
) -> tuple[int, int, int]:
    groups: defaultdict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        key = safe_string(row.get(key_field, ""))
        if key:
            groups[key].append(row)

    output_rows: list[dict[str, Any]] = []
    group_index = 0
    cross_archive_groups = 0
    cross_split_groups = 0
    for key in sorted(groups):
        members = groups[key]
        if len(members) < 2:
            continue
        group_index += 1
        archives = sorted({safe_string(item.get("source_archive", "")) for item in members})
        splits = sorted(
            {
                safe_string(item.get("final_split", "") or item.get("source_split", ""))
                for item in members
                if safe_string(item.get("final_split", "") or item.get("source_split", ""))
                not in {"", "unspecified"}
            }
        )
        labels = sorted({safe_string(item.get("harmonized_label", "")) for item in members})
        cross_archive = len(archives) > 1
        cross_split = len(splits) > 1
        if cross_archive:
            cross_archive_groups += 1
        if cross_split:
            cross_split_groups += 1
        group_id = f"{prefix}{group_index:06d}"
        for item in sorted(
            members,
            key=lambda value: (
                safe_string(value.get("source_archive", "")),
                safe_string(value.get("relative_path", "")),
            ),
        ):
            output_rows.append(
                {
                    "group_id": group_id,
                    "group_key": key,
                    "group_size": len(members),
                    "cross_archive": "YES" if cross_archive else "NO",
                    "cross_split": "YES" if cross_split else "NO",
                    "label_conflict": "YES" if len(labels) > 1 else "NO",
                    "archives": "|".join(archives),
                    "splits": "|".join(splits),
                    "labels": "|".join(labels),
                    "source": item.get("source", ""),
                    "source_archive": item.get("source_archive", ""),
                    "source_split": item.get("source_split", ""),
                    "external_split": item.get("external_split", ""),
                    "final_split": item.get("final_split", ""),
                    "raw_label": item.get("raw_label", ""),
                    "harmonized_label": item.get("harmonized_label", ""),
                    "relative_path": item.get("relative_path", ""),
                    "bytes": item.get("bytes", ""),
                    "sha256": item.get("sha256", ""),
                    "pixel_sha256": item.get("pixel_sha256", ""),
                    "family_key": item.get("family_key", ""),
                }
            )

    write_tsv(
        output_path,
        output_rows,
        [
            "group_id",
            "group_key",
            "group_size",
            "cross_archive",
            "cross_split",
            "label_conflict",
            "archives",
            "splits",
            "labels",
            "source",
            "source_archive",
            "source_split",
            "external_split",
            "final_split",
            "raw_label",
            "harmonized_label",
            "relative_path",
            "bytes",
            "sha256",
            "pixel_sha256",
            "family_key",
        ],
    )
    return group_index, cross_archive_groups, cross_split_groups


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True, type=Path)
    parser.add_argument("--big", required=True, type=Path)
    parser.add_argument("--archive-manifest", required=True, type=Path)
    parser.add_argument("--extraction-status-dir", required=True, type=Path)
    parser.add_argument("--raw-dir", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()

    output = args.output
    output.mkdir(parents=True, exist_ok=True)

    archive_tasks = read_tsv(args.archive_manifest)
    extraction_status_rows: list[dict[str, str]] = []
    for status_file in sorted(args.extraction_status_dir.glob("task_*.tsv")):
        extraction_status_rows.extend(read_tsv(status_file))
    write_tsv(
        output / "stage1_extraction_status.tsv",
        extraction_status_rows,
        [
            "task_id",
            "source",
            "archive_name",
            "archive_path",
            "destination_dir",
            "status",
            "extractor",
            "started_at",
            "finished_at",
            "archive_bytes",
            "extracted_files",
            "extracted_directories",
            "extracted_bytes",
            "message",
        ],
    )

    successful_status = {
        int(row["task_id"]): row
        for row in extraction_status_rows
        if row.get("status") in {"COMPLETED", "SKIPPED_VALID_EXISTING"}
    }
    missing_tasks = [
        int(task["task_id"])
        for task in archive_tasks
        if int(task["task_id"]) not in successful_status
    ]
    if missing_tasks:
        raise RuntimeError(f"Extraction did not complete for task IDs: {missing_tasks}")

    inspection_tasks: list[dict[str, Any]] = []
    archive_root_map: dict[str, tuple[str, str, str]] = {}
    for task in archive_tasks:
        archive_root = Path(task["destination_dir"])
        marker = archive_root / ".stage1_extract_complete.json"
        if not marker.is_file():
            raise FileNotFoundError(f"Missing extraction marker: {marker}")
        archive_root_map[str(archive_root)] = (
            task["source"],
            task["archive_name"],
            task["archive_sha256"],
        )
        for path in sorted(archive_root.rglob("*")):
            if not path.is_file():
                continue
            relative_path = relative_context(path, archive_root)
            extension = path.suffix.lower()
            role, role_reason = classify_role(relative_path, extension)
            raw_label = infer_raw_label(relative_path)
            harmonized = RAW_TO_HARMONIZED.get(raw_label, raw_label)
            family_key, family_type = family_hint(path.name, raw_label, task["archive_name"])
            inspection_tasks.append(
                {
                    "source": task["source"],
                    "source_archive": task["archive_name"],
                    "archive_sha256": task["archive_sha256"],
                    "archive_root": str(archive_root),
                    "relative_path": relative_path,
                    "absolute_path": str(path),
                    "file_name": safe_string(path.name),
                    "extension": extension or "[no_extension]",
                    "bytes": path.stat().st_size,
                    "role": role,
                    "role_reason": role_reason,
                    "source_split": infer_split(relative_path),
                    "external_split": "",
                    "final_split": "",
                    "raw_label": raw_label,
                    "harmonized_label": harmonized,
                    "is_cacao": "NO" if raw_label == "coffee_normal" else "YES",
                    "family_key": family_key,
                    "family_type": family_type,
                    "file_error": "",
                    "absolute_path_obj": path,
                }
            )

    print(f"stage1_files_to_inspect={len(inspection_tasks)} workers={args.workers}")
    with ThreadPoolExecutor(max_workers=max(1, args.workers)) as executor:
        inspected_rows = list(executor.map(inspect_file, inspection_tasks))

    # Parse external train/validation/test lists supplied beside CocoaMonilia.
    split_entries: list[dict[str, Any]] = []
    split_by_basename: defaultdict[str, set[str]] = defaultdict(set)
    for split_name in ("train", "val", "test"):
        for path in sorted(args.raw_dir.rglob(f"{split_name}.txt")):
            # Exclude extracted annotation text; RAW_DIR contains the three
            # record-level split files and public archives only.
            with path.open(encoding="utf-8", errors="replace") as handle:
                for line_number, line in enumerate(handle, start=1):
                    basename = parse_external_split_line(line)
                    if not basename:
                        continue
                    normalized = basename.lower()
                    canonical_split = "validation" if split_name == "val" else split_name
                    split_by_basename[normalized].add(canonical_split)
                    split_entries.append(
                        {
                            "split_file": str(path),
                            "split": canonical_split,
                            "line_number": line_number,
                            "raw_entry": line.strip(),
                            "basename": basename,
                            "normalized_basename": normalized,
                        }
                    )

    basename_photo_index: defaultdict[str, list[int]] = defaultdict(list)
    for index, row in enumerate(inspected_rows):
        if row["role"] in {"photo_candidate", "photo_non_cacao"}:
            basename_photo_index[row["file_name"].lower()].append(index)

    for index, row in enumerate(inspected_rows):
        if row["role"] not in {"photo_candidate", "photo_non_cacao"}:
            row["final_split"] = row["source_split"]
            continue
        splits = (
            split_by_basename.get(row["file_name"].lower(), set())
            if row["source_archive"] == "CocoaMoniliaDataSet.zip"
            else set()
        )
        if len(splits) == 1:
            row["external_split"] = next(iter(splits))
        elif len(splits) > 1:
            row["external_split"] = "CONFLICT:" + "|".join(sorted(splits))
        row["final_split"] = row["external_split"] or row["source_split"]

    # Parse COCO JSON annotations and YOLO text-label files.
    annotation_summaries: list[dict[str, Any]] = []
    coco_image_rows: list[dict[str, Any]] = []
    yolo_rows: list[dict[str, Any]] = []
    for row in inspected_rows:
        if row["role"] != "annotation":
            continue
        path = Path(row["absolute_path"])
        if row["extension"] == ".json":
            summary, images = parse_coco_json(
                path,
                row["source"],
                row["source_archive"],
                row["relative_path"],
            )
            annotation_summaries.append(summary)
            coco_image_rows.extend(images)
        elif row["extension"] == ".txt" and (
            "yolo_annotations" in row["relative_path"].lower()
            or "/labels/" in row["relative_path"].lower().replace("\\", "/")
        ):
            try:
                lines = [
                    line.strip()
                    for line in path.read_text(encoding="utf-8", errors="replace").splitlines()
                    if line.strip()
                ]
                class_ids = sorted({line.split()[0] for line in lines if line.split()})
                yolo_rows.append(
                    {
                        "source": row["source"],
                        "source_archive": row["source_archive"],
                        "annotation_path": row["relative_path"],
                        "basename": path.stem,
                        "object_lines": len(lines),
                        "class_ids": "|".join(class_ids),
                        "parse_status": "OK",
                        "message": "",
                    }
                )
            except Exception as exc:
                yolo_rows.append(
                    {
                        "source": row["source"],
                        "source_archive": row["source_archive"],
                        "annotation_path": row["relative_path"],
                        "basename": path.stem,
                        "object_lines": "",
                        "class_ids": "",
                        "parse_status": "FAILED",
                        "message": f"{type(exc).__name__}:{exc}",
                    }
                )

    # Reconcile COCO/YOLO declarations against physical photographs within
    # the same source archive. Basename matching is conservative and retains
    # ambiguity when more than one physical file shares a basename.
    archive_basename_index: defaultdict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for photo_row in inspected_rows:
        if photo_row["role"] in {"photo_candidate", "photo_non_cacao"}:
            archive_basename_index[
                (photo_row["source_archive"], photo_row["file_name"].lower())
            ].append(photo_row)

    for coco_row in coco_image_rows:
        matches = archive_basename_index.get(
            (coco_row["source_archive"], coco_row["basename"].lower()), []
        )
        coco_row["matched_photo_files"] = len(matches)
        coco_row["matched_paths"] = "|".join(
            safe_string(match["relative_path"]) for match in matches[:20]
        )

    for yolo_row in yolo_rows:
        candidate_names = {
            yolo_row["basename"].lower(),
            (yolo_row["basename"] + ".jpg").lower(),
            (yolo_row["basename"] + ".jpeg").lower(),
            (yolo_row["basename"] + ".png").lower(),
        }
        matches: list[dict[str, Any]] = []
        for candidate in candidate_names:
            matches.extend(
                archive_basename_index.get((yolo_row["source_archive"], candidate), [])
            )
        # Deduplicate paths if multiple candidate spellings converge.
        unique_matches = {
            safe_string(match["relative_path"]): match for match in matches
        }
        yolo_row["matched_photo_files"] = len(unique_matches)
        yolo_row["matched_paths"] = "|".join(sorted(unique_matches)[:20])

    file_fields = [
        "source",
        "source_archive",
        "archive_sha256",
        "archive_root",
        "relative_path",
        "absolute_path",
        "file_name",
        "extension",
        "bytes",
        "role",
        "role_reason",
        "source_split",
        "external_split",
        "final_split",
        "raw_label",
        "harmonized_label",
        "is_cacao",
        "family_key",
        "family_type",
        "sha256",
        "pixel_sha256",
        "image_read_ok",
        "image_format",
        "width",
        "height",
        "mode",
        "exif_make",
        "exif_model",
        "exif_datetime_original",
        "file_error",
        "image_error",
    ]
    inspected_rows.sort(key=lambda row: (row["source_archive"], row["relative_path"]))
    write_tsv(output / "stage1_all_files.tsv", inspected_rows, file_fields)

    photo_rows = [
        row for row in inspected_rows if row["role"] in {"photo_candidate", "photo_non_cacao"}
    ]
    nonphoto_rows = [
        row for row in inspected_rows if row["role"] not in {"photo_candidate", "photo_non_cacao"}
    ]
    write_tsv(output / "stage1_photo_manifest.tsv", photo_rows, file_fields)
    write_tsv(output / "stage1_nonphoto_manifest.tsv", nonphoto_rows, file_fields)
    write_tsv(
        output / "stage1_external_split_entries.tsv",
        split_entries,
        [
            "split_file",
            "split",
            "line_number",
            "raw_entry",
            "basename",
            "normalized_basename",
        ],
    )
    write_tsv(
        output / "stage1_annotation_summary.tsv",
        annotation_summaries,
        [
            "source",
            "source_archive",
            "annotation_path",
            "annotation_format",
            "image_entries",
            "annotation_entries",
            "category_count",
            "categories",
            "parse_status",
            "message",
        ],
    )
    write_tsv(
        output / "stage1_coco_images.tsv",
        coco_image_rows,
        [
            "source",
            "source_archive",
            "annotation_path",
            "image_id",
            "file_name",
            "basename",
            "declared_width",
            "declared_height",
            "annotation_count",
            "category_ids",
            "category_names",
            "matched_photo_files",
            "matched_paths",
        ],
    )
    write_tsv(
        output / "stage1_yolo_annotations.tsv",
        yolo_rows,
        [
            "source",
            "source_archive",
            "annotation_path",
            "basename",
            "object_lines",
            "class_ids",
            "parse_status",
            "message",
            "matched_photo_files",
            "matched_paths",
        ],
    )

    # Reconcile split-list entries against extracted photo basenames.
    split_reconciliation_rows: list[dict[str, Any]] = []
    for basename, splits in sorted(split_by_basename.items()):
        matches = basename_photo_index.get(basename, [])
        split_reconciliation_rows.append(
            {
                "normalized_basename": basename,
                "splits": "|".join(sorted(splits)),
                "split_conflict": "YES" if len(splits) > 1 else "NO",
                "matched_photo_files": len(matches),
                "matched_archives": "|".join(
                    sorted({inspected_rows[index]["source_archive"] for index in matches})
                ),
                "matched_paths": "|".join(
                    inspected_rows[index]["relative_path"] for index in matches[:20]
                ),
            }
        )
    write_tsv(
        output / "stage1_split_reconciliation.tsv",
        split_reconciliation_rows,
        [
            "normalized_basename",
            "splits",
            "split_conflict",
            "matched_photo_files",
            "matched_archives",
            "matched_paths",
        ],
    )

    # Dataset-level summaries.
    group_counter: Counter[tuple[str, str, str, str, str]] = Counter()
    for row in inspected_rows:
        group_counter[
            (
                row["source_archive"],
                row["role"],
                row["raw_label"],
                row["harmonized_label"],
                row["final_split"] or "unspecified",
            )
        ] += 1
    dataset_rows = [
        {
            "source_archive": key[0],
            "role": key[1],
            "raw_label": key[2],
            "harmonized_label": key[3],
            "split": key[4],
            "file_count": count,
        }
        for key, count in sorted(group_counter.items())
    ]
    write_tsv(
        output / "stage1_dataset_counts.tsv",
        dataset_rows,
        [
            "source_archive",
            "role",
            "raw_label",
            "harmonized_label",
            "split",
            "file_count",
        ],
    )

    # Per-archive reconciliation combines physical files and annotation claims.
    archive_reconciliation_rows: list[dict[str, Any]] = []
    for task in archive_tasks:
        archive_name = task["archive_name"]
        rows = [row for row in inspected_rows if row["source_archive"] == archive_name]
        photos = [row for row in rows if row["role"] in {"photo_candidate", "photo_non_cacao"}]
        cacao_photos = [row for row in photos if row["is_cacao"] == "YES"]
        masks = [row for row in rows if row["role"] == "mask"]
        annotations = [row for row in rows if row["role"] == "annotation"]
        coco_summaries = [
            row for row in annotation_summaries if row["source_archive"] == archive_name
        ]
        archive_reconciliation_rows.append(
            {
                "source": task["source"],
                "source_archive": archive_name,
                "physical_photo_files": len(photos),
                "physical_cacao_photo_files": len(cacao_photos),
                "mask_files": len(masks),
                "annotation_files": len(annotations),
                "coco_image_entries": sum(
                    int(row.get("image_entries", 0) or 0) for row in coco_summaries
                ),
                "coco_annotation_entries": sum(
                    int(row.get("annotation_entries", 0) or 0) for row in coco_summaries
                ),
                "yolo_annotation_files": sum(
                    1 for row in yolo_rows if row["source_archive"] == archive_name
                ),
                "unique_file_sha256_photos": len(
                    {row["sha256"] for row in photos if row["sha256"]}
                ),
                "unique_pixel_sha256_photos": len(
                    {row["pixel_sha256"] for row in photos if row["pixel_sha256"]}
                ),
                "unique_filename_families_photos": len(
                    {row["family_key"] for row in photos if row["family_key"]}
                ),
                "unreadable_images": sum(
                    1
                    for row in rows
                    if row["role"] in {"photo_candidate", "photo_non_cacao", "mask"}
                    and row["image_read_ok"] == "NO"
                ),
            }
        )
    write_tsv(
        output / "stage1_archive_reconciliation.tsv",
        archive_reconciliation_rows,
        [
            "source",
            "source_archive",
            "physical_photo_files",
            "physical_cacao_photo_files",
            "mask_files",
            "annotation_files",
            "coco_image_entries",
            "coco_annotation_entries",
            "yolo_annotation_files",
            "unique_file_sha256_photos",
            "unique_pixel_sha256_photos",
            "unique_filename_families_photos",
            "unreadable_images",
        ],
    )

    # Exact-file, decoded-pixel, and filename-family lineage groups.
    exact_group_count, exact_cross_archive, exact_cross_split = group_members(
        photo_rows,
        "sha256",
        "EXACT",
        output / "stage1_exact_duplicate_groups.tsv",
    )
    pixel_group_count, pixel_cross_archive, pixel_cross_split = group_members(
        photo_rows,
        "pixel_sha256",
        "PIXEL",
        output / "stage1_pixel_duplicate_groups.tsv",
    )
    name_group_count, name_cross_archive, name_cross_split = group_members(
        photo_rows,
        "family_key",
        "NAME",
        output / "stage1_filename_lineage_groups.tsv",
    )

    ontology_rows = [
        {
            "raw_label": "healthy",
            "harmonized_label": "healthy",
            "causal_agent_or_damage": "none/asymptomatic",
            "biological_type": "healthy",
            "primary_use": "closed-set disease classification; external stage baseline",
            "default_include": "YES",
        },
        {
            "raw_label": "black_pod_rot|black_pod|phytophthora",
            "harmonized_label": "black_pod",
            "causal_agent_or_damage": "Phytophthora spp.",
            "biological_type": "oomycete disease",
            "primary_use": "disease classification",
            "default_include": "YES",
        },
        {
            "raw_label": "frosty_pod|monilia|moniliophthora",
            "harmonized_label": "frosty_pod",
            "causal_agent_or_damage": "Moniliophthora roreri",
            "biological_type": "fungal disease",
            "primary_use": "disease classification",
            "default_include": "YES",
        },
        {
            "raw_label": "monilia_stage_m1",
            "harmonized_label": "frosty_pod_stage_m1",
            "causal_agent_or_damage": "Moniliophthora roreri; hump stage",
            "biological_type": "disease stage",
            "primary_use": "external stage-aware validation",
            "default_include": "YES",
        },
        {
            "raw_label": "monilia_stage_m2",
            "harmonized_label": "frosty_pod_stage_m2",
            "causal_agent_or_damage": "Moniliophthora roreri; oily/brown-spot stage",
            "biological_type": "disease stage",
            "primary_use": "external stage-aware validation",
            "default_include": "YES",
        },
        {
            "raw_label": "monilia_stage_m3",
            "harmonized_label": "frosty_pod_stage_m3",
            "causal_agent_or_damage": "Moniliophthora roreri; sporulation stage",
            "biological_type": "disease stage",
            "primary_use": "external stage-aware validation",
            "default_include": "YES",
        },
        {
            "raw_label": "pod_borer",
            "harmonized_label": "pod_borer_damage",
            "causal_agent_or_damage": "insect damage",
            "biological_type": "non-disease damage",
            "primary_use": "open-set/out-of-scope challenge",
            "default_include": "NO",
        },
        {
            "raw_label": "mirid_damage",
            "harmonized_label": "mirid_damage",
            "causal_agent_or_damage": "mirid insect injury",
            "biological_type": "non-disease damage",
            "primary_use": "open-set/out-of-scope challenge",
            "default_include": "NO",
        },
        {
            "raw_label": "coffee_normal",
            "harmonized_label": "coffee_healthy",
            "causal_agent_or_damage": "none",
            "biological_type": "non-cacao image",
            "primary_use": "source-confounding negative control only",
            "default_include": "NO",
        },
        {
            "raw_label": "unknown",
            "harmonized_label": "unknown",
            "causal_agent_or_damage": "unresolved",
            "biological_type": "requires curation",
            "primary_use": "none until curated",
            "default_include": "NO",
        },
    ]
    write_tsv(
        output / "stage1_label_ontology_draft.tsv",
        ontology_rows,
        [
            "raw_label",
            "harmonized_label",
            "causal_agent_or_damage",
            "biological_type",
            "primary_use",
            "default_include",
        ],
    )

    role_counts = Counter(row["role"] for row in inspected_rows)
    label_counts = Counter(
        row["harmonized_label"]
        for row in photo_rows
        if row["is_cacao"] == "YES"
    )
    unreadable = sum(
        1
        for row in inspected_rows
        if row["role"] in {"photo_candidate", "photo_non_cacao", "mask"}
        and row["image_read_ok"] == "NO"
    )
    split_conflicts = sum(1 for row in split_reconciliation_rows if row["split_conflict"] == "YES")
    unmatched_split_entries = sum(
        1 for row in split_reconciliation_rows if int(row["matched_photo_files"]) == 0
    )

    summary_lines = [
        "# manifest reconstruction canonical manifest and exact-lineage audit",
        "",
        f"- Generated: {now()}",
        f"- Archive extraction tasks completed: {len(successful_status):,} / {len(archive_tasks):,}",
        f"- Extracted files inspected: {len(inspected_rows):,}",
        f"- Photograph candidates: {len(photo_rows):,}",
        f"- Cacao photograph candidates: {sum(1 for row in photo_rows if row['is_cacao'] == 'YES'):,}",
        f"- Non-cacao photograph candidates: {sum(1 for row in photo_rows if row['is_cacao'] == 'NO'):,}",
        f"- Mask/segmentation images: {role_counts.get('mask', 0):,}",
        f"- Annotation files: {role_counts.get('annotation', 0):,}",
        f"- Metadata files: {role_counts.get('metadata', 0):,}",
        f"- Unreadable image-like files: {unreadable:,}",
        f"- Exact-file duplicate groups: {exact_group_count:,} "
        f"(cross-archive {exact_cross_archive:,}; cross-split {exact_cross_split:,})",
        f"- Exact decoded-pixel duplicate groups: {pixel_group_count:,} "
        f"(cross-archive {pixel_cross_archive:,}; cross-split {pixel_cross_split:,})",
        f"- Filename-family lineage groups: {name_group_count:,} "
        f"(cross-archive {name_cross_archive:,}; cross-split {name_cross_split:,})",
        f"- External split basenames: {len(split_reconciliation_rows):,}",
        f"- External split conflicts: {split_conflicts:,}",
        f"- External split basenames unmatched to extracted photos: {unmatched_split_entries:,}",
        "",
        "## Cacao photograph counts by draft harmonized label",
        "",
    ]
    for label, count in sorted(label_counts.items(), key=lambda item: (-item[1], item[0])):
        summary_lines.append(f"- {label}: {count:,}")
    summary_lines.extend(
        [
            "",
            "## Claim boundary",
            "",
            "This stage can establish exact file reuse, exact decoded-pixel reuse, annotation/split",
            "inconsistencies, and filename-derived lineage families. It does not yet establish",
            "near-duplicate ancestry under crop, rotation, recompression, brightness change, or",
            "other transformations. Those require Stage 2 perceptual hashes and geometric checks.",
            "",
        ]
    )
    (output / "stage1_summary.md").write_text("\n".join(summary_lines), encoding="utf-8")
    print("\n".join(summary_lines))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"FATAL: {type(exc).__name__}: {exc}", file=sys.stderr)
        raise

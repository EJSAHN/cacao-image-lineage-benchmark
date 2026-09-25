#!/usr/bin/env python3
"""Inventory downloaded files and archive members without extracting payloads."""
from __future__ import annotations

import argparse
import csv
import hashlib
import os
import sys
import tarfile
import time
import zipfile
from collections import Counter
from pathlib import Path, PurePosixPath
from typing import Any, Iterable

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp", ".gif"}
ANNOTATION_EXTENSIONS = {".json", ".xml", ".txt", ".csv", ".tsv", ".yaml", ".yml", ".mat"}
ARCHIVE_SUFFIXES = {".zip", ".tar", ".tgz", ".gz", ".bz2", ".xz", ".7z", ".rar"}


def sha256(path: Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            h.update(chunk)
    return h.hexdigest()


def source_from_path(path: Path) -> str:
    for part in path.parts:
        if part.startswith("figshare_"):
            return part
        if part.startswith("zenodo_"):
            return part
    return "unknown"


def classify_member(name: str) -> tuple[str, str, str, str]:
    normalized_name = name.replace("\\", "/")
    suffix = PurePosixPath(normalized_name).suffix.lower()
    if suffix in IMAGE_EXTENSIONS:
        kind = "image"
    elif suffix in ANNOTATION_EXTENSIONS:
        kind = "annotation_or_metadata"
    elif suffix in ARCHIVE_SUFFIXES:
        kind = "nested_archive"
    else:
        kind = "other"
    parts = [p for p in PurePosixPath(normalized_name).parts if p not in ("/", "", ".")]
    prefix1 = parts[0] if parts else ""
    prefix2 = "/".join(parts[:2]) if parts else ""
    return suffix or "[no_extension]", kind, prefix1, prefix2


def zip_members(path: Path) -> Iterable[tuple[str, int, bool]]:
    with zipfile.ZipFile(path) as archive:
        for info in archive.infolist():
            yield info.filename, int(info.file_size), info.is_dir()


def tar_members(path: Path) -> Iterable[tuple[str, int, bool]]:
    with tarfile.open(path, mode="r:*") as archive:
        for member in archive:
            yield member.name, int(member.size), member.isdir()


def detect_archive(path: Path) -> str:
    if zipfile.is_zipfile(path):
        return "zip"
    try:
        if tarfile.is_tarfile(path):
            return "tar"
    except OSError:
        pass
    suffixes = "".join(path.suffixes).lower()
    if suffixes.endswith(".7z"):
        return "7z_unsupported"
    if suffixes.endswith(".rar"):
        return "rar_unsupported"
    if suffixes.endswith(".gz"):
        return "gzip_non_tar_or_unrecognized"
    return "not_archive"


def write_tsv(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, delimiter="\t", extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    os.replace(tmp, path)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True, type=Path)
    parser.add_argument("--raw", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()

    root = args.root.resolve()
    raw = args.raw.resolve()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)

    if not raw.is_dir():
        raise FileNotFoundError(f"Raw directory is missing: {raw}")

    downloaded_rows: list[dict[str, Any]] = []
    archive_rows: list[dict[str, Any]] = []
    member_rows: list[dict[str, Any]] = []
    prefix_counter: Counter[tuple[str, str, str]] = Counter()
    extension_counter: Counter[tuple[str, str, str]] = Counter()

    files = sorted(p for p in raw.rglob("*") if p.is_file() and not p.name.endswith(".part"))
    if not files:
        raise RuntimeError(f"No completed files found under {raw}")

    for file_path in files:
        source = source_from_path(file_path)
        archive_type = detect_archive(file_path)
        digest = sha256(file_path)
        downloaded_rows.append(
            {
                "source": source,
                "path": str(file_path),
                "name": file_path.name,
                "bytes": file_path.stat().st_size,
                "sha256": digest,
                "archive_type": archive_type,
                "mtime": time.strftime(
                    "%Y-%m-%dT%H:%M:%S%z", time.localtime(file_path.stat().st_mtime)
                ),
            }
        )

        member_count = image_count = annotation_count = nested_archive_count = directory_count = 0
        uncompressed_bytes = 0
        error = ""
        iterator: Iterable[tuple[str, int, bool]] | None = None
        if archive_type == "zip":
            iterator = zip_members(file_path)
        elif archive_type == "tar":
            iterator = tar_members(file_path)

        if iterator is not None:
            try:
                for member_name, member_size, is_dir in iterator:
                    if is_dir:
                        directory_count += 1
                        continue
                    member_count += 1
                    uncompressed_bytes += member_size
                    suffix, kind, prefix1, prefix2 = classify_member(member_name)
                    if kind == "image":
                        image_count += 1
                    elif kind == "annotation_or_metadata":
                        annotation_count += 1
                    elif kind == "nested_archive":
                        nested_archive_count += 1
                    prefix_counter[(source, file_path.name, prefix1)] += 1
                    if prefix2:
                        prefix_counter[(source, file_path.name, prefix2)] += 1
                    extension_counter[(source, file_path.name, suffix)] += 1
                    member_rows.append(
                        {
                            "source": source,
                            "archive_path": str(file_path),
                            "archive_name": file_path.name,
                            "member_path": member_name,
                            "member_bytes": member_size,
                            "extension": suffix,
                            "kind": kind,
                            "prefix_level1": prefix1,
                            "prefix_level2": prefix2,
                        }
                    )
            except Exception as exc:
                error = f"{type(exc).__name__}: {exc}"

        archive_rows.append(
            {
                "source": source,
                "archive_path": str(file_path),
                "archive_name": file_path.name,
                "archive_type": archive_type,
                "compressed_bytes": file_path.stat().st_size,
                "member_files": member_count,
                "member_directories": directory_count,
                "image_members": image_count,
                "annotation_or_metadata_members": annotation_count,
                "nested_archive_members": nested_archive_count,
                "uncompressed_member_bytes": uncompressed_bytes,
                "inventory_error": error,
            }
        )

    write_tsv(
        output / "downloaded_files.tsv",
        downloaded_rows,
        ["source", "path", "name", "bytes", "sha256", "archive_type", "mtime"],
    )
    write_tsv(
        output / "archive_summary.tsv",
        archive_rows,
        [
            "source",
            "archive_path",
            "archive_name",
            "archive_type",
            "compressed_bytes",
            "member_files",
            "member_directories",
            "image_members",
            "annotation_or_metadata_members",
            "nested_archive_members",
            "uncompressed_member_bytes",
            "inventory_error",
        ],
    )
    write_tsv(
        output / "archive_members.tsv",
        member_rows,
        [
            "source",
            "archive_path",
            "archive_name",
            "member_path",
            "member_bytes",
            "extension",
            "kind",
            "prefix_level1",
            "prefix_level2",
        ],
    )
    prefix_rows = [
        {"source": source, "archive_name": archive, "prefix": prefix, "member_count": count}
        for (source, archive, prefix), count in sorted(
            prefix_counter.items(), key=lambda x: (x[0][0], x[0][1], -x[1], x[0][2])
        )
    ]
    write_tsv(
        output / "path_prefix_counts.tsv",
        prefix_rows,
        ["source", "archive_name", "prefix", "member_count"],
    )
    extension_rows = [
        {"source": source, "archive_name": archive, "extension": ext, "member_count": count}
        for (source, archive, ext), count in sorted(
            extension_counter.items(), key=lambda x: (x[0][0], x[0][1], -x[1], x[0][2])
        )
    ]
    write_tsv(
        output / "extension_counts.tsv",
        extension_rows,
        ["source", "archive_name", "extension", "member_count"],
    )

    total_compressed = sum(int(row["bytes"]) for row in downloaded_rows)
    total_members = sum(int(row["member_files"]) for row in archive_rows)
    total_images = sum(int(row["image_members"]) for row in archive_rows)
    total_annotations = sum(
        int(row["annotation_or_metadata_members"]) for row in archive_rows
    )
    errors = [row for row in archive_rows if row["inventory_error"]]

    summary = f"""# Stage 0 archive screening summary

- Generated: {time.strftime('%Y-%m-%dT%H:%M:%S%z')}
- Project root: `{root}`
- Raw root: `{raw}`
- Completed downloaded files: {len(downloaded_rows):,}
- Compressed bytes: {total_compressed:,}
- Archive member files inventoried: {total_members:,}
- Image members recognized by extension: {total_images:,}
- Annotation/metadata-like members recognized by extension: {total_annotations:,}
- Archives with inventory errors: {len(errors):,}

## What must be decided from this bundle before manifest reconstruction

1. Actual archive nesting and whether source datasets remain separable.
2. Whether labels are encoded in directories, filenames, annotation files, or all three.
3. Whether reported counts refer to images, annotated objects, augmentation descendants, or mixed units.
4. Which archives can be extracted directly and which contain nested archives.
5. A canonical immutable source/label ontology before any deduplication or model benchmark.

No raw image has been extracted or modified in Stage 0.
"""
    (output / "stage0_summary.md").write_text(summary, encoding="utf-8")
    print(summary)
    # Per-archive errors are retained in the tables so the diagnostic bundle can
    # still be created. Catastrophic failures above still return nonzero.
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"FATAL: {type(exc).__name__}: {exc}", file=sys.stderr)
        raise

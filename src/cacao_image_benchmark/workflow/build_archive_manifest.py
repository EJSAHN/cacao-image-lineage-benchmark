#!/usr/bin/env python3
"""Build a deterministic archive-extraction task manifest from the archive inventory."""
from __future__ import annotations

import argparse
import csv
import re
import sys
from pathlib import Path


SUPPORTED_ARCHIVE_TYPES = {
    "zip",
    "tar",
    "rar_unsupported",
    "7z_unsupported",
    "gzip_non_tar_or_unrecognized",
}


def slugify(value: str) -> str:
    stem = value
    for suffix in (".tar.gz", ".tar.bz2", ".tar.xz", ".zip", ".rar", ".7z", ".tgz", ".tar", ".gz"):
        if stem.lower().endswith(suffix):
            stem = stem[: -len(suffix)]
            break
    stem = re.sub(r"[^A-Za-z0-9._-]+", "_", stem).strip("._-")
    return stem or "archive"


def read_tsv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle, delimiter="\t"))


def write_tsv(path: Path, rows: list[dict[str, object]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)
    tmp.replace(path)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--downloaded-files", required=True, type=Path)
    parser.add_argument("--extracted-root", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()

    rows = read_tsv(args.downloaded_files)
    tasks: list[dict[str, object]] = []
    seen_destinations: set[str] = set()

    for row in rows:
        archive_type = row.get("archive_type", "")
        if archive_type not in SUPPORTED_ARCHIVE_TYPES:
            continue
        source = row.get("source", "unknown")
        name = row.get("name", "")
        archive_path = Path(row.get("path", ""))
        if not archive_path.is_absolute():
            raise ValueError(f"Archive path must be absolute: {archive_path}")
        if not archive_path.is_file():
            raise FileNotFoundError(f"Archive is missing: {archive_path}")

        base_slug = slugify(name)
        destination = args.extracted_root / source / base_slug
        collision_index = 2
        while str(destination) in seen_destinations:
            destination = args.extracted_root / source / f"{base_slug}_{collision_index}"
            collision_index += 1
        seen_destinations.add(str(destination))

        tasks.append(
            {
                "task_id": len(tasks) + 1,
                "source": source,
                "archive_name": name,
                "archive_path": str(archive_path),
                "archive_type_stage0": archive_type,
                "archive_bytes": int(row.get("bytes", "0") or 0),
                "archive_sha256": row.get("sha256", ""),
                "destination_dir": str(destination),
            }
        )

    if not tasks:
        raise RuntimeError("No extractable archives were found in downloaded_files.tsv")

    write_tsv(
        args.output,
        tasks,
        [
            "task_id",
            "source",
            "archive_name",
            "archive_path",
            "archive_type_stage0",
            "archive_bytes",
            "archive_sha256",
            "destination_dir",
        ],
    )

    print(f"stage1_archive_tasks={len(tasks)}")
    for task in tasks:
        print(
            f"task={task['task_id']} source={task['source']} "
            f"type={task['archive_type_stage0']} archive={task['archive_name']}"
        )
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"FATAL: {type(exc).__name__}: {exc}", file=sys.stderr)
        raise

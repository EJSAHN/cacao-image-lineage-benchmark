#!/usr/bin/env python3
"""Extract one public archive into an isolated, restartable manifest reconstruction directory."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import shutil
import stat
import subprocess
import sys
import tarfile
import time
import zipfile
from pathlib import Path, PurePosixPath
from typing import Iterable


MARKER_NAME = ".stage1_extract_complete.json"


def now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%S%z")


def read_task(manifest: Path, task_id: int) -> dict[str, str]:
    with manifest.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle, delimiter="\t"))
    for row in rows:
        if int(row["task_id"]) == task_id:
            return row
    raise KeyError(f"Task ID {task_id} is absent from {manifest}")


def write_status(path: Path, row: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = [
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
    ]
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, delimiter="\t", extrasaction="ignore")
        writer.writeheader()
        writer.writerow(row)
    tmp.replace(path)


def ensure_descendant(path: Path, root: Path) -> None:
    root_resolved = root.resolve()
    path_resolved = path.resolve(strict=False)
    if path_resolved != root_resolved and root_resolved not in path_resolved.parents:
        raise RuntimeError(f"Unsafe path outside extraction root: {path_resolved}")


def safe_target(root: Path, member_name: str) -> Path:
    normalized = member_name.replace("\\", "/")
    pure = PurePosixPath(normalized)
    if pure.is_absolute() or ".." in pure.parts:
        raise RuntimeError(f"Unsafe archive member path: {member_name}")
    target = root.joinpath(*[part for part in pure.parts if part not in ("", ".")])
    ensure_descendant(target, root)
    return target


def extract_zip(archive_path: Path, destination: Path) -> str:
    with zipfile.ZipFile(archive_path) as archive:
        for info in archive.infolist():
            target = safe_target(destination, info.filename)
            if info.is_dir():
                target.mkdir(parents=True, exist_ok=True)
                continue
            # Reject symbolic links encoded in Unix mode bits.
            unix_mode = (info.external_attr >> 16) & 0xFFFF
            if stat.S_ISLNK(unix_mode):
                print(f"SKIP_SYMLINK member={info.filename}")
                continue
            target.parent.mkdir(parents=True, exist_ok=True)
            with archive.open(info, "r") as source, target.open("wb") as sink:
                shutil.copyfileobj(source, sink, length=8 * 1024 * 1024)
    return "python_zipfile"


def extract_tar(archive_path: Path, destination: Path) -> str:
    with tarfile.open(archive_path, mode="r:*") as archive:
        for member in archive:
            target = safe_target(destination, member.name)
            if member.isdir():
                target.mkdir(parents=True, exist_ok=True)
                continue
            if member.issym() or member.islnk() or not member.isfile():
                print(f"SKIP_NONREGULAR member={member.name} type={member.type!r}")
                continue
            target.parent.mkdir(parents=True, exist_ok=True)
            source = archive.extractfile(member)
            if source is None:
                raise RuntimeError(f"Could not read tar member: {member.name}")
            with source, target.open("wb") as sink:
                shutil.copyfileobj(source, sink, length=8 * 1024 * 1024)
    return "python_tarfile"


def find_external_extractor() -> tuple[str, str]:
    preferred = os.environ.get("CACAO_RAR_TOOL", "").strip()
    if preferred:
        preferred_path = Path(preferred)
        if preferred_path.is_file() and os.access(preferred_path, os.X_OK):
            return str(preferred_path), "7zip"
        raise RuntimeError(f"CACAO_RAR_TOOL is not executable: {preferred}")

    candidates = [
        ("7zz", "7zip"),
        ("7z", "7zip"),
        ("unrar", "unrar"),
        ("unar", "unar"),
        ("bsdtar", "bsdtar"),
    ]
    for executable, kind in candidates:
        found = shutil.which(executable)
        if found:
            return found, kind
    raise RuntimeError(
        "No RAR/7z-capable extractor found. Expected one of: 7zz, 7z, unrar, unar, bsdtar."
    )


def stream_command(command: list[str]) -> None:
    print("RUN:", " ".join(command))
    process = subprocess.Popen(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        errors="replace",
        bufsize=1,
    )
    assert process.stdout is not None
    output_tail: list[str] = []
    for line in process.stdout:
        clean = line.rstrip()
        print(clean)
        output_tail.append(clean)
        if len(output_tail) > 80:
            output_tail.pop(0)
    return_code = process.wait()
    if return_code != 0:
        compact_tail = " || ".join(output_tail[-40:])
        raise RuntimeError(
            f"Extractor exit code {return_code}; command={' '.join(command)}; output_tail={compact_tail}"
        )


def extract_external(archive_path: Path, destination: Path) -> str:
    executable, kind = find_external_extractor()
    if kind == "7zip":
        command = [executable, "x", "-y", "-aoa", f"-o{destination}", str(archive_path)]
    elif kind == "unrar":
        command = [executable, "x", "-o+", "-y", str(archive_path), str(destination) + os.sep]
    elif kind == "unar":
        command = [
            executable,
            "-force-overwrite",
            "-output-directory",
            str(destination),
            str(archive_path),
        ]
    else:
        command = [executable, "-xf", str(archive_path), "-C", str(destination)]
    stream_command(command)
    return f"{kind}:{executable}"


def detect_and_extract(archive_path: Path, stage0_type: str, destination: Path) -> str:
    if zipfile.is_zipfile(archive_path):
        return extract_zip(archive_path, destination)
    try:
        if tarfile.is_tarfile(archive_path):
            return extract_tar(archive_path, destination)
    except OSError:
        pass
    # RAR, 7z, and other recognized compressed containers are delegated to a
    # command-line extractor installed in the persistent conda environment.
    return extract_external(archive_path, destination)


def directory_stats(root: Path) -> tuple[int, int, int]:
    file_count = 0
    directory_count = 0
    total_bytes = 0
    for path in root.rglob("*"):
        if path.is_dir():
            directory_count += 1
        elif path.is_file():
            file_count += 1
            total_bytes += path.stat().st_size
    return file_count, directory_count, total_bytes


def marker_is_valid(marker: Path, expected_sha256: str, archive_bytes: int) -> bool:
    if not marker.is_file():
        return False
    try:
        data = json.loads(marker.read_text(encoding="utf-8"))
    except Exception:
        return False
    return (
        data.get("archive_sha256") == expected_sha256
        and int(data.get("archive_bytes", -1)) == archive_bytes
        and int(data.get("extracted_files", 0)) > 0
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--task-id", required=True, type=int)
    parser.add_argument("--extracted-root", required=True, type=Path)
    parser.add_argument("--status-dir", required=True, type=Path)
    args = parser.parse_args()

    task = read_task(args.manifest, args.task_id)
    archive_path = Path(task["archive_path"])
    destination = Path(task["destination_dir"])
    extracted_root = args.extracted_root
    expected_sha256 = task["archive_sha256"]
    expected_bytes = int(task["archive_bytes"])

    if not archive_path.is_file():
        raise FileNotFoundError(f"Archive is missing: {archive_path}")
    actual_bytes = archive_path.stat().st_size
    if actual_bytes != expected_bytes:
        raise RuntimeError(
            f"Archive size changed: expected={expected_bytes} actual={actual_bytes} path={archive_path}"
        )

    ensure_descendant(destination, extracted_root)
    status_path = args.status_dir / f"task_{args.task_id:04d}.tsv"
    started = now()
    base_status: dict[str, object] = {
        "task_id": args.task_id,
        "source": task["source"],
        "archive_name": task["archive_name"],
        "archive_path": str(archive_path),
        "destination_dir": str(destination),
        "status": "FAILED",
        "extractor": "",
        "started_at": started,
        "finished_at": "",
        "archive_bytes": actual_bytes,
        "extracted_files": 0,
        "extracted_directories": 0,
        "extracted_bytes": 0,
        "message": "",
    }

    marker = destination / MARKER_NAME
    if marker_is_valid(marker, expected_sha256, expected_bytes):
        file_count, directory_count, total_bytes = directory_stats(destination)
        base_status.update(
            {
                "status": "SKIPPED_VALID_EXISTING",
                "extractor": "existing_marker",
                "finished_at": now(),
                "extracted_files": file_count,
                "extracted_directories": directory_count,
                "extracted_bytes": total_bytes,
                "message": "Existing extraction marker matches the Stage 0 archive.",
            }
        )
        write_status(status_path, base_status)
        print(json.dumps(base_status, indent=2))
        return 0

    temp_name = f".stage1_tmp_task_{args.task_id:04d}_{os.environ.get('SLURM_JOB_ID', os.getpid())}"
    temp_dir = extracted_root / temp_name
    ensure_descendant(temp_dir, extracted_root)
    extracted_root.mkdir(parents=True, exist_ok=True)

    if temp_dir.exists():
        shutil.rmtree(temp_dir)
    temp_dir.mkdir(parents=True)

    try:
        extractor = detect_and_extract(archive_path, task["archive_type_stage0"], temp_dir)
        file_count, directory_count, total_bytes = directory_stats(temp_dir)
        if file_count == 0:
            raise RuntimeError(f"Extraction produced no files: {archive_path}")

        marker_data = {
            "stage": "stage1_extraction",
            "created_at": now(),
            "task_id": args.task_id,
            "source": task["source"],
            "archive_name": task["archive_name"],
            "archive_path": str(archive_path),
            "archive_bytes": actual_bytes,
            "archive_sha256": expected_sha256,
            "extractor": extractor,
            "extracted_files": file_count,
            "extracted_directories": directory_count,
            "extracted_bytes": total_bytes,
        }
        (temp_dir / MARKER_NAME).write_text(
            json.dumps(marker_data, indent=2, sort_keys=True), encoding="utf-8"
        )

        if destination.exists():
            ensure_descendant(destination, extracted_root)
            shutil.rmtree(destination)
        destination.parent.mkdir(parents=True, exist_ok=True)
        os.replace(temp_dir, destination)

        # Recount after marker and atomic move.
        file_count, directory_count, total_bytes = directory_stats(destination)
        base_status.update(
            {
                "status": "COMPLETED",
                "extractor": extractor,
                "finished_at": now(),
                "extracted_files": file_count,
                "extracted_directories": directory_count,
                "extracted_bytes": total_bytes,
                "message": "Archive extracted and completion marker written.",
            }
        )
        write_status(status_path, base_status)
        print(json.dumps(base_status, indent=2))
        return 0
    except Exception as exc:
        base_status.update(
            {
                "status": "FAILED",
                "finished_at": now(),
                "message": f"{type(exc).__name__}: {exc}",
            }
        )
        write_status(status_path, base_status)
        print(json.dumps(base_status, indent=2), file=sys.stderr)
        raise
    finally:
        if temp_dir.exists():
            shutil.rmtree(temp_dir, ignore_errors=True)


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"FATAL: {type(exc).__name__}: {exc}", file=sys.stderr)
        raise

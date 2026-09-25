#!/usr/bin/env python3
"""Create a deterministic short-side-256 JPEG cache for one end-to-end sensitivity array chunk."""
from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

import pandas as pd
from PIL import Image, ImageOps

from end_to_end_common import ProjectPaths, read_tsv, sha256_file, write_tsv

SHORT_SIDE = 256
JPEG_QUALITY = 92


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True, type=Path)
    parser.add_argument("--big", required=True, type=Path)
    parser.add_argument("--task-index", required=True, type=int)
    parser.add_argument("--expected-tasks", type=int, default=32)
    return parser.parse_args()


def resize_short_side(image: Image.Image, short_side: int = SHORT_SIDE) -> Image.Image:
    width, height = image.size
    if width <= 0 or height <= 0:
        raise ValueError(f"Invalid image dimensions: {image.size}")
    scale = short_side / min(width, height)
    new_width = max(short_side, int(round(width * scale)))
    new_height = max(short_side, int(round(height * scale)))
    return image.resize((new_width, new_height), Image.Resampling.LANCZOS)


def cache_valid(path: Path) -> bool:
    if not path.is_file() or path.stat().st_size <= 0:
        return False
    try:
        with Image.open(path) as image:
            image.verify()
        with Image.open(path) as image:
            width, height = image.size
            return min(width, height) == SHORT_SIDE and image.mode == "RGB"
    except Exception:
        return False


def main() -> int:
    args = parse_args()
    if not (1 <= args.task_index <= args.expected_tasks):
        raise ValueError(f"Invalid cache task index: {args.task_index}")
    paths = ProjectPaths(args.root, args.big)
    chunk_path = paths.cache_chunks / f"stage3d_cache_chunk_{args.task_index:03d}.tsv"
    if not chunk_path.is_file():
        raise RuntimeError(f"Cache chunk is missing: {chunk_path}")
    chunk = read_tsv(chunk_path)
    for column in ["sample_id", "absolute_path", "cache_path"]:
        if column not in chunk.columns:
            raise RuntimeError(f"Cache chunk lacks {column}")
    paths.cache_status.mkdir(parents=True, exist_ok=True)
    started = time.time()
    created = 0
    reused = 0
    rows: list[dict[str, object]] = []
    for index, record in enumerate(chunk.itertuples(index=False), start=1):
        source = Path(str(record.absolute_path))
        destination = Path(str(record.cache_path))
        if not source.is_file():
            raise RuntimeError(f"Source image is missing: {source}")
        destination.parent.mkdir(parents=True, exist_ok=True)
        if cache_valid(destination):
            reused += 1
            with Image.open(destination) as cached:
                cache_width, cache_height = cached.size
            rows.append(
                {
                    "sample_id": record.sample_id,
                    "source_path": str(source),
                    "cache_path": str(destination),
                    "cache_width": cache_width,
                    "cache_height": cache_height,
                    "cache_bytes": destination.stat().st_size,
                    "cache_sha256": sha256_file(destination),
                    "action": "REUSED_VALID_CACHE",
                }
            )
            continue
        temp = destination.with_name(destination.name + f".tmp.{os.getpid()}")
        if temp.exists():
            temp.unlink()
        with Image.open(source) as raw:
            image = ImageOps.exif_transpose(raw).convert("RGB")
            resized = resize_short_side(image)
            cache_width, cache_height = resized.size
            resized.save(
                temp,
                format="JPEG",
                quality=JPEG_QUALITY,
                subsampling=0,
                optimize=False,
                progressive=False,
            )
        os.replace(temp, destination)
        if not cache_valid(destination):
            raise RuntimeError(f"Cache validation failed after writing: {destination}")
        created += 1
        rows.append(
            {
                "sample_id": record.sample_id,
                "source_path": str(source),
                "cache_path": str(destination),
                "cache_width": cache_width,
                "cache_height": cache_height,
                "cache_bytes": destination.stat().st_size,
                "cache_sha256": sha256_file(destination),
                "action": "CREATED",
            }
        )
        if index % 100 == 0 or index == len(chunk):
            print(
                f"cache_task={args.task_index} progress={index}/{len(chunk)} created={created} reused={reused}",
                flush=True,
            )
    audit_path = paths.cache_status / f"stage3d_cache_task_{args.task_index:03d}_audit.tsv"
    status_path = paths.cache_status / f"stage3d_cache_task_{args.task_index:03d}_status.tsv"
    write_tsv(pd.DataFrame(rows), audit_path)
    elapsed = time.time() - started
    status = pd.DataFrame(
        [
            {
                "task_index": args.task_index,
                "tasks_expected": args.expected_tasks,
                "rows_expected": len(chunk),
                "rows_processed": len(rows),
                "created": created,
                "reused": reused,
                "failed": 0,
                "elapsed_seconds": elapsed,
                "audit_path": str(audit_path),
                "status": "COMPLETED",
            }
        ]
    )
    write_tsv(status, status_path)
    print(status.to_string(index=False))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"FATAL: {type(exc).__name__}: {exc}", file=sys.stderr, flush=True)
        raise

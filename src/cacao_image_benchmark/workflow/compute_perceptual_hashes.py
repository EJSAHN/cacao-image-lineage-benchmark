#!/usr/bin/env python3
"""perceptual-candidate discovery perceptual-hash extraction.

One row is computed per exact-file component representative. Exact duplicates are
therefore never reprocessed. The hash implementations are self-contained and use
Pillow, NumPy, and SciPy only.
"""
from __future__ import annotations

import argparse
import math
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from PIL import Image, ImageOps, ImageStat
from scipy.fft import dctn


def write_tsv(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, sep="\t", index=False, na_rep="")


def bits_to_hex(bits: np.ndarray) -> str:
    flat = np.asarray(bits, dtype=np.uint8).reshape(-1)
    value = 0
    for bit in flat:
        value = (value << 1) | int(bit)
    width = math.ceil(len(flat) / 4)
    return f"{value:0{width}x}"


def gray_array(image: Image.Image, size: tuple[int, int]) -> np.ndarray:
    return np.asarray(
        image.convert("L").resize(size, Image.Resampling.LANCZOS),
        dtype=np.float64,
    )


def phash(image: Image.Image, hash_size: int = 8, highfreq_factor: int = 4) -> str:
    size = hash_size * highfreq_factor
    pixels = gray_array(image, (size, size))
    coeff = dctn(pixels, type=2, norm="ortho")[:hash_size, :hash_size]
    median = float(np.median(coeff))
    return bits_to_hex(coeff > median)


def dhash(image: Image.Image, hash_size: int = 8) -> str:
    pixels = gray_array(image, (hash_size + 1, hash_size))
    return bits_to_hex(pixels[:, 1:] > pixels[:, :-1])


def ahash(image: Image.Image, hash_size: int = 8) -> str:
    pixels = gray_array(image, (hash_size, hash_size))
    return bits_to_hex(pixels > float(pixels.mean()))


def haar_whash(image: Image.Image, hash_size: int = 8) -> str:
    # A deterministic one-level Haar low-pass hash. Resize to 2x target, average
    # each 2x2 block, then threshold the low-pass coefficients at their median.
    pixels = gray_array(image, (hash_size * 2, hash_size * 2))
    low = pixels.reshape(hash_size, 2, hash_size, 2).mean(axis=(1, 3))
    return bits_to_hex(low > float(np.median(low)))


def variants(image: Image.Image) -> list[Image.Image]:
    mirrored = ImageOps.mirror(image)
    return [
        image,
        image.transpose(Image.Transpose.ROTATE_90),
        image.transpose(Image.Transpose.ROTATE_180),
        image.transpose(Image.Transpose.ROTATE_270),
        mirrored,
        mirrored.transpose(Image.Transpose.ROTATE_90),
        mirrored.transpose(Image.Transpose.ROTATE_180),
        mirrored.transpose(Image.Transpose.ROTATE_270),
    ]


def entropy_from_gray(gray: Image.Image) -> float:
    hist = np.asarray(gray.histogram(), dtype=np.float64)
    total = hist.sum()
    if total <= 0:
        return float("nan")
    p = hist[hist > 0] / total
    return float(-(p * np.log2(p)).sum())


def inspect(row: dict[str, Any]) -> dict[str, Any]:
    result = {
        "exact_component_id": row["exact_component_id"],
        "sha256": row["sha256"],
        "source_archive": row["source_archive"],
        "source_dataset_id": row["source_dataset_id"],
        "corrected_label": row["corrected_label"],
        "corrected_split": row["corrected_split"],
        "source_origin_key": row["source_origin_key"],
        "representative_relative_path": row["relative_path"],
        "representative_absolute_path": row["absolute_path"],
        "width": row["width"],
        "height": row["height"],
        "bytes": row["bytes"],
        "hash_status": "",
        "hash_error": "",
        "phash_algorithm": "dct_32x32_low8_median",
        "dhash_algorithm": "horizontal_gradient_9x8",
        "whash_algorithm": "single_level_haar_lowpass_median",
        "ahash_algorithm": "mean_threshold_8x8",
    }
    try:
        path = Path(row["absolute_path"])
        with Image.open(path) as raw:
            image = ImageOps.exif_transpose(raw).convert("RGB")
            image.thumbnail((512, 512), Image.Resampling.LANCZOS)
            transformed = variants(image)
            phashes = sorted({phash(v) for v in transformed})
            dhashes = sorted({dhash(v) for v in transformed})
            whashes = sorted({haar_whash(v) for v in transformed})
            ahashes = sorted({ahash(v) for v in transformed})
            stat = ImageStat.Stat(image)
            gray = image.convert("L")
            result.update(
                {
                    "thumbnail_width": image.width,
                    "thumbnail_height": image.height,
                    "phash_primary": phash(image),
                    "phash_variants": "|".join(phashes),
                    "dhash_primary": dhash(image),
                    "dhash_variants": "|".join(dhashes),
                    "whash_primary": haar_whash(image),
                    "whash_variants": "|".join(whashes),
                    "ahash_primary": ahash(image),
                    "ahash_variants": "|".join(ahashes),
                    "mean_r": stat.mean[0],
                    "mean_g": stat.mean[1],
                    "mean_b": stat.mean[2],
                    "std_r": stat.stddev[0],
                    "std_g": stat.stddev[1],
                    "std_b": stat.stddev[2],
                    "gray_entropy": entropy_from_gray(gray),
                    "hash_status": "OK",
                }
            )
    except Exception as exc:
        result["hash_status"] = "FAILED"
        result["hash_error"] = f"{type(exc).__name__}:{exc}"
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--task-id", required=True, type=int)
    parser.add_argument("--task-manifest", required=True, type=Path)
    parser.add_argument("--canonical-manifest", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()

    tasks = pd.read_csv(args.task_manifest, sep="\t", keep_default_na=False)
    selected = tasks[tasks["task_id"].eq(args.task_id)]
    if len(selected) != 1:
        raise RuntimeError(f"Task ID {args.task_id} not found exactly once")
    archive = selected.iloc[0]["source_archive"]

    canonical = pd.read_csv(args.canonical_manifest, sep="\t", low_memory=False, keep_default_na=False)
    subset = canonical[
        canonical["source_archive"].eq(archive)
        & canonical["exact_representative"].eq("YES")
        & canonical["image_read_ok"].eq("YES")
        & canonical["exact_component_id"].ne("NO_HASH")
    ].copy()
    subset = subset.sort_values(["exact_component_id", "relative_path"])

    args.output_dir.mkdir(parents=True, exist_ok=True)
    output = args.output_dir / f"task_{args.task_id:04d}_hashes.tsv"
    status_path = args.output_dir / f"task_{args.task_id:04d}_status.tsv"

    start = time.time()
    rows = subset.to_dict("records")
    print(f"task_id={args.task_id} archive={archive} representatives={len(rows)} workers={args.workers}")

    results: list[dict[str, Any]] = []
    with ThreadPoolExecutor(max_workers=max(1, args.workers)) as executor:
        for i, result in enumerate(executor.map(inspect, rows), 1):
            results.append(result)
            if i % 250 == 0 or i == len(rows):
                print(f"progress={i}/{len(rows)}")

    output_df = pd.DataFrame(results)
    write_tsv(output_df, output)
    ok = int(output_df["hash_status"].eq("OK").sum()) if len(output_df) else 0
    failed = int(output_df["hash_status"].eq("FAILED").sum()) if len(output_df) else 0
    elapsed = time.time() - start
    status = pd.DataFrame(
        [
            {
                "task_id": args.task_id,
                "source_archive": archive,
                "representatives_expected": len(rows),
                "rows_written": len(output_df),
                "hash_ok": ok,
                "hash_failed": failed,
                "elapsed_seconds": elapsed,
                "output_path": str(output),
                "status": "COMPLETED" if failed == 0 and len(output_df) == len(rows) else "FAILED",
            }
        ]
    )
    write_tsv(status, status_path)
    if failed or len(output_df) != len(rows):
        raise RuntimeError(f"Hash task incomplete: expected={len(rows)} rows={len(output_df)} failed={failed}")
    print(status.to_string(index=False))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"FATAL: {type(exc).__name__}: {exc}", file=sys.stderr)
        raise

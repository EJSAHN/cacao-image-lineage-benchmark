#!/usr/bin/env python3
"""Extract deterministic visual-shortcut features for one matched-removal and shortcut controls transformation."""
from __future__ import annotations

import argparse
import os
import platform
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image, ImageFilter, ImageOps
from torch import nn
from torch.utils.data import DataLoader, Dataset
import torchvision
from torchvision.models import EfficientNet_B0_Weights, efficientnet_b0

from control_analysis_common import VISUAL_FEATURE_TASKS, ProjectPaths, read_tsv, sha256_file, write_tsv

IMAGE_SIZE = 224
IMAGENET_MEAN = np.asarray([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.asarray([0.229, 0.224, 0.225], dtype=np.float32)
CANVAS_COLOR = (124, 116, 104)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True, type=Path)
    parser.add_argument("--big", required=True, type=Path)
    parser.add_argument("--task-index", required=True, type=int)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--threads", type=int, default=16)
    parser.add_argument("--expected-rows", type=int, default=13354)
    parser.add_argument("--weight-mode", choices=["default", "none"], default="default")
    return parser.parse_args()


def base_letterbox(path: str) -> Image.Image:
    with Image.open(path) as raw:
        image = ImageOps.exif_transpose(raw).convert("RGB")
        image.thumbnail((IMAGE_SIZE, IMAGE_SIZE), Image.Resampling.LANCZOS)
        canvas = Image.new("RGB", (IMAGE_SIZE, IMAGE_SIZE), CANVAS_COLOR)
        x = (IMAGE_SIZE - image.width) // 2
        y = (IMAGE_SIZE - image.height) // 2
        canvas.paste(image, (x, y))
    return canvas


def transformed_canvas(path: str, feature_name: str) -> Image.Image:
    canvas = base_letterbox(path)
    if feature_name == "blur32_efficientnet":
        small = canvas.resize((32, 32), Image.Resampling.LANCZOS)
        small = small.filter(ImageFilter.GaussianBlur(radius=1.25))
        return small.resize((IMAGE_SIZE, IMAGE_SIZE), Image.Resampling.BILINEAR)
    if feature_name == "grayscale_efficientnet":
        return ImageOps.grayscale(canvas).convert("RGB")
    array = np.asarray(canvas, dtype=np.uint8).copy()
    edge = int(round(IMAGE_SIZE * 0.20))
    x0, x1 = edge, IMAGE_SIZE - edge
    y0, y1 = edge, IMAGE_SIZE - edge
    if feature_name == "border_only_efficientnet":
        border_mask = np.ones((IMAGE_SIZE, IMAGE_SIZE), dtype=bool)
        border_mask[y0:y1, x0:x1] = False
        fill = np.median(array[border_mask], axis=0).astype(np.uint8)
        array[y0:y1, x0:x1, :] = fill
        return Image.fromarray(array, mode="RGB")
    if feature_name == "center_only_efficientnet":
        center = array[y0:y1, x0:x1, :]
        fill = np.median(center.reshape(-1, 3), axis=0).astype(np.uint8)
        masked = np.empty_like(array)
        masked[:, :, :] = fill
        masked[y0:y1, x0:x1, :] = center
        return Image.fromarray(masked, mode="RGB")
    raise ValueError(f"Unsupported transformed feature: {feature_name}")


def tensor_from_canvas(canvas: Image.Image) -> torch.Tensor:
    array = np.asarray(canvas, dtype=np.float32) / 255.0
    array = (array - IMAGENET_MEAN) / IMAGENET_STD
    return torch.from_numpy(np.transpose(array, (2, 0, 1))).contiguous()


def color_histogram(path: str) -> np.ndarray:
    with Image.open(path) as raw:
        image = ImageOps.exif_transpose(raw).convert("RGB")
        image.thumbnail((512, 512), Image.Resampling.LANCZOS)
        rgb = np.asarray(image, dtype=np.float32) / 255.0
        hsv = np.asarray(image.convert("HSV"), dtype=np.float32) / 255.0
        gray = np.asarray(ImageOps.grayscale(image), dtype=np.float32) / 255.0
    features: list[np.ndarray] = []
    for array, bins in [(rgb, 16), (hsv, 16)]:
        for channel in range(3):
            hist, _ = np.histogram(array[..., channel], bins=bins, range=(0.0, 1.0))
            hist = hist.astype(np.float32)
            hist /= max(float(hist.sum()), 1.0)
            features.append(hist)
        features.append(array.reshape(-1, 3).mean(axis=0).astype(np.float32))
        features.append(array.reshape(-1, 3).std(axis=0).astype(np.float32))
    hist, _ = np.histogram(gray, bins=16, range=(0.0, 1.0))
    hist = hist.astype(np.float32)
    hist /= max(float(hist.sum()), 1.0)
    features.append(hist)
    return np.concatenate(features).astype(np.float32, copy=False)


class ShortcutDataset(Dataset):
    def __init__(self, table: pd.DataFrame, feature_name: str) -> None:
        self.table = table.reset_index(drop=True)
        self.feature_name = feature_name

    def __len__(self) -> int:
        return len(self.table)

    def __getitem__(self, index: int) -> tuple[int, torch.Tensor]:
        path = str(self.table.iloc[index]["absolute_path"])
        canvas = transformed_canvas(path, self.feature_name)
        return index, tensor_from_canvas(canvas)


def extract_embedding_features(
    manifest: pd.DataFrame,
    feature_name: str,
    output_path: Path,
    model_cache: Path,
    batch_size: int,
    workers: int,
    threads: int,
    weight_mode: str,
) -> dict[str, object]:
    os.environ["TORCH_HOME"] = str(model_cache)
    torch.set_num_threads(max(1, threads))
    try:
        torch.set_num_interop_threads(max(1, min(4, threads)))
    except RuntimeError:
        pass
    weights = EfficientNet_B0_Weights.DEFAULT if weight_mode == "default" else None
    model = efficientnet_b0(weights=weights)
    model.classifier = nn.Identity()
    model.eval().to("cpu")
    dataset = ShortcutDataset(manifest, feature_name)
    loader_kwargs: dict[str, object] = {
        "dataset": dataset,
        "batch_size": max(1, batch_size),
        "shuffle": False,
        "num_workers": max(0, workers),
        "pin_memory": False,
    }
    if workers > 0:
        loader_kwargs.update({"persistent_workers": True, "prefetch_factor": 2})
    loader = DataLoader(**loader_kwargs)
    output = np.empty((len(manifest), 1280), dtype=np.float32)
    written = 0
    with torch.inference_mode():
        for indices, batch in loader:
            features = model(batch).reshape(batch.shape[0], -1)
            norms = torch.linalg.vector_norm(features, ord=2, dim=1, keepdim=True)
            if bool((norms <= torch.finfo(features.dtype).tiny).any()):
                raise RuntimeError("Zero-norm transformed embeddings")
            values = (features / norms).detach().cpu().numpy().astype(np.float32, copy=False)
            idx = indices.detach().cpu().numpy().astype(int, copy=False)
            output[idx] = values
            written += len(idx)
            if written % 512 < len(idx) or written == len(manifest):
                print(f"feature={feature_name} progress={written}/{len(manifest)}", flush=True)
    if written != len(manifest) or not np.isfinite(output).all():
        raise RuntimeError(f"Incomplete or non-finite transformed features: {written}/{len(manifest)}")
    norms = np.linalg.norm(output, axis=1)
    if not np.allclose(norms, 1.0, atol=2e-4):
        raise RuntimeError("Transformed embedding normalization failed")
    np.save(output_path, output)
    return {
        "feature_dim": output.shape[1],
        "l2_norm_min": float(norms.min()),
        "l2_norm_max": float(norms.max()),
        "weights": str(weights),
    }


def extract_hist_features(manifest: pd.DataFrame, output_path: Path) -> dict[str, object]:
    first = color_histogram(str(manifest.iloc[0]["absolute_path"]))
    output = np.empty((len(manifest), len(first)), dtype=np.float32)
    output[0] = first
    for index in range(1, len(manifest)):
        output[index] = color_histogram(str(manifest.iloc[index]["absolute_path"]))
        if (index + 1) % 500 == 0 or index + 1 == len(manifest):
            print(f"feature=color_histogram progress={index + 1}/{len(manifest)}", flush=True)
    if not np.isfinite(output).all():
        raise RuntimeError("Non-finite color-histogram features")
    np.save(output_path, output)
    return {
        "feature_dim": output.shape[1],
        "l2_norm_min": float("nan"),
        "l2_norm_max": float("nan"),
        "weights": "none",
    }


def main() -> int:
    args = parse_args()
    if args.task_index not in VISUAL_FEATURE_TASKS:
        raise ValueError(f"Unsupported task index: {args.task_index}")
    feature_name = VISUAL_FEATURE_TASKS[args.task_index]
    paths = ProjectPaths(args.root, args.big)
    paths.feature_dir.mkdir(parents=True, exist_ok=True)
    paths.feature_meta.mkdir(parents=True, exist_ok=True)
    manifest_path = paths.prepared / "stage3c_feature_manifest.tsv"
    if not manifest_path.is_file():
        raise RuntimeError("matched-removal and shortcut controls feature manifest is missing")
    manifest = read_tsv(manifest_path)
    if len(manifest) != args.expected_rows:
        raise RuntimeError(f"Unexpected feature rows: {len(manifest)} != {args.expected_rows}")
    missing = [path for path in manifest["absolute_path"] if not Path(str(path)).is_file()]
    if missing:
        raise RuntimeError(f"Missing image paths, first examples: {missing[:10]}")
    output_path = paths.feature_dir / f"stage3c_{feature_name}_features.npy"
    status_path = paths.feature_meta / f"stage3c_feature_task_{args.task_index:02d}_status.tsv"
    provenance_path = paths.feature_meta / f"stage3c_{feature_name}_provenance.tsv"
    started = time.time()
    if feature_name == "color_histogram":
        details = extract_hist_features(manifest, output_path)
    else:
        details = extract_embedding_features(
            manifest,
            feature_name,
            output_path,
            paths.big / "model_cache" / "stage2c",
            args.batch_size,
            args.workers,
            args.threads,
            args.weight_mode,
        )
    provenance = pd.DataFrame([{
        "task_index": args.task_index,
        "feature_name": feature_name,
        "rows": len(manifest),
        **details,
        "output_path": str(output_path),
        "output_sha256": sha256_file(output_path),
        "manifest_sha256": sha256_file(manifest_path),
        "elapsed_seconds": time.time() - started,
        "python": platform.python_version(),
        "torch": torch.__version__,
        "torchvision": torchvision.__version__,
        "platform": platform.platform(),
        "image_size": IMAGE_SIZE,
        "base_resize_policy": "preserve_aspect_ratio_then_center_letterbox",
    }])
    write_tsv(provenance, provenance_path)
    write_tsv(pd.DataFrame([{
        "task_index": args.task_index,
        "feature_name": feature_name,
        "rows_expected": len(manifest),
        "rows_written": len(manifest),
        "output_path": str(output_path),
        "elapsed_seconds": time.time() - started,
        "status": "COMPLETED",
    }]), status_path)
    print(provenance.to_string(index=False))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"FATAL: {type(exc).__name__}: {exc}", file=sys.stderr, flush=True)
        raise

#!/usr/bin/env python3
"""Extract deterministic frozen-image embeddings for the embedding retrieval all-image sweep."""
from __future__ import annotations

import argparse
import hashlib
import os
import platform
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image, ImageOps
from torch import nn
from torch.utils.data import DataLoader, Dataset
import torchvision
from torchvision.models import (
    EfficientNet_B0_Weights,
    ResNet18_Weights,
    efficientnet_b0,
    resnet18,
)

IMAGE_SIZE = 224
IMAGENET_MEAN = np.asarray([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.asarray([0.229, 0.224, 0.225], dtype=np.float32)


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(chunk_size)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def letterbox_tensor(path: str, size: int = IMAGE_SIZE) -> torch.Tensor:
    with Image.open(path) as raw:
        image = ImageOps.exif_transpose(raw).convert("RGB")
        image.thumbnail((size, size), Image.Resampling.LANCZOS)
        # Neutral brown-gray approximates cacao-image backgrounds without stretching the image.
        canvas = Image.new("RGB", (size, size), (124, 116, 104))
        x = (size - image.width) // 2
        y = (size - image.height) // 2
        canvas.paste(image, (x, y))
        array = np.asarray(canvas, dtype=np.float32) / 255.0
    array = (array - IMAGENET_MEAN) / IMAGENET_STD
    return torch.from_numpy(np.transpose(array, (2, 0, 1))).contiguous()


class ImageDataset(Dataset):
    def __init__(self, table: pd.DataFrame) -> None:
        self.table = table.reset_index(drop=True)

    def __len__(self) -> int:
        return len(self.table)

    def __getitem__(self, index: int) -> tuple[int, torch.Tensor]:
        row = self.table.iloc[index]
        return index, letterbox_tensor(str(row["absolute_path"]))


def make_model(name: str, weight_mode: str) -> tuple[nn.Module, str, int]:
    if weight_mode not in {"default", "none"}:
        raise ValueError(f"Unsupported weight mode: {weight_mode}")
    if name == "resnet18":
        weights = ResNet18_Weights.DEFAULT if weight_mode == "default" else None
        model = resnet18(weights=weights)
        model.fc = nn.Identity()
        return model, str(weights), 512
    if name == "efficientnet_b0":
        weights = EfficientNet_B0_Weights.DEFAULT if weight_mode == "default" else None
        model = efficientnet_b0(weights=weights)
        model.classifier = nn.Identity()
        return model, str(weights), 1280
    raise ValueError(f"Unsupported model: {name}")


def write_tsv(table: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(path, sep="\t", index=False, na_rep="")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--canonical-manifest", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--matrix-dir", required=True, type=Path)
    parser.add_argument("--model-cache", required=True, type=Path)
    parser.add_argument("--models", default="resnet18,efficientnet_b0")
    parser.add_argument("--weight-mode", choices=["default", "none"], default="default")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--threads", type=int, default=16)
    args = parser.parse_args()

    started = time.time()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.matrix_dir.mkdir(parents=True, exist_ok=True)
    args.model_cache.mkdir(parents=True, exist_ok=True)
    os.environ["TORCH_HOME"] = str(args.model_cache)

    torch.set_num_threads(max(1, args.threads))
    try:
        torch.set_num_interop_threads(max(1, min(4, args.threads)))
    except RuntimeError:
        pass

    canonical = pd.read_csv(
        args.canonical_manifest,
        sep="\t",
        keep_default_na=False,
        low_memory=False,
    )
    required = {
        "exact_component_id",
        "exact_representative",
        "image_read_ok",
        "absolute_path",
        "relative_path",
        "source_archive",
        "source_dataset_id",
        "corrected_label",
        "corrected_split",
        "source_origin_key",
        "width",
        "height",
        "sha256",
        "pixel_sha256",
    }
    missing = required - set(canonical.columns)
    if missing:
        raise RuntimeError(f"Canonical manifest is missing columns: {sorted(missing)}")

    reps = canonical[
        canonical["exact_representative"].eq("YES")
        & canonical["image_read_ok"].eq("YES")
        & canonical["exact_component_id"].ne("NO_HASH")
    ].copy()
    reps = (
        reps.sort_values("exact_component_id")
        .drop_duplicates("exact_component_id")
        .reset_index(drop=True)
    )
    if reps.empty:
        raise RuntimeError("No readable exact-component representatives found")

    missing_paths = [str(path) for path in reps["absolute_path"] if not Path(str(path)).is_file()]
    if missing_paths:
        raise RuntimeError(f"Representative paths missing, first examples: {missing_paths[:10]}")

    manifest_cols = [
        "exact_component_id",
        "sha256",
        "pixel_sha256",
        "source_archive",
        "source_dataset_id",
        "corrected_label",
        "corrected_split",
        "source_origin_key",
        "relative_path",
        "absolute_path",
        "width",
        "height",
    ]
    embedding_manifest = reps[manifest_cols].copy()
    embedding_manifest.insert(0, "embedding_row", np.arange(len(embedding_manifest), dtype=np.int64))
    write_tsv(embedding_manifest, args.output_dir / "stage2c_embedding_manifest.tsv")

    dataset = ImageDataset(reps)
    loader_kwargs: dict[str, object] = {
        "dataset": dataset,
        "batch_size": max(1, args.batch_size),
        "shuffle": False,
        "num_workers": max(0, args.workers),
        "pin_memory": False,
    }
    if args.workers > 0:
        loader_kwargs.update({"persistent_workers": True, "prefetch_factor": 2})
    loader = DataLoader(**loader_kwargs)

    models = [item.strip() for item in args.models.split(",") if item.strip()]
    provenance_rows: list[dict[str, object]] = []
    for model_name in models:
        model_started = time.time()
        model, weights_name, expected_dim = make_model(model_name, args.weight_mode)
        model.eval().to("cpu")

        output = np.empty((len(reps), expected_dim), dtype=np.float32)
        written = 0
        with torch.inference_mode():
            for indices, batch in loader:
                features = model(batch)
                if isinstance(features, (tuple, list)):
                    features = features[0]
                features = features.reshape(features.shape[0], -1)
                idx = indices.detach().cpu().numpy().astype(int, copy=False)
                feature_norms = torch.linalg.vector_norm(features, ord=2, dim=1, keepdim=True)
                zero_mask = feature_norms.squeeze(1) <= torch.finfo(features.dtype).tiny
                if bool(zero_mask.any()):
                    failed_rows = idx[zero_mask.detach().cpu().numpy()].tolist()
                    raise RuntimeError(
                        f"Zero-norm embeddings generated for {model_name}; rows={failed_rows[:20]}"
                    )
                features = features / feature_norms
                values = features.detach().cpu().numpy().astype(np.float32, copy=False)
                output[idx] = values
                written += len(idx)
                if written % 512 < len(idx) or written == len(reps):
                    print(f"model={model_name} progress={written}/{len(reps)}", flush=True)

        if written != len(reps):
            raise RuntimeError(f"Only {written}/{len(reps)} embeddings written for {model_name}")
        if not np.isfinite(output).all():
            raise RuntimeError(f"Non-finite embedding values generated for {model_name}")
        norms = np.linalg.norm(output, axis=1)
        if not np.allclose(norms, 1.0, atol=2e-4):
            raise RuntimeError(f"Embedding normalization check failed for {model_name}")

        path = args.matrix_dir / f"stage2c_{model_name}_embeddings.npy"
        np.save(path, output)
        provenance_rows.append(
            {
                "model": model_name,
                "weights": weights_name,
                "weight_mode": args.weight_mode,
                "embedding_rows": len(output),
                "embedding_dim": output.shape[1],
                "dtype": str(output.dtype),
                "l2_norm_min": float(norms.min()),
                "l2_norm_max": float(norms.max()),
                "output_path": str(path),
                "output_sha256": sha256_file(path),
                "elapsed_seconds": time.time() - model_started,
            }
        )
        del model
        del output

    provenance = pd.DataFrame(provenance_rows)
    provenance["python"] = platform.python_version()
    provenance["torch"] = torch.__version__
    provenance["torchvision"] = torchvision.__version__
    provenance["platform"] = platform.platform()
    provenance["image_size"] = IMAGE_SIZE
    provenance["resize_policy"] = "preserve_aspect_ratio_then_center_letterbox"
    provenance["normalization"] = "ImageNet mean/std"
    write_tsv(provenance, args.output_dir / "stage2c_embedding_provenance.tsv")

    summary = pd.DataFrame(
        [
            {"metric": "exact_representatives", "value": len(reps)},
            {"metric": "models", "value": "|".join(models)},
            {"metric": "weight_mode", "value": args.weight_mode},
            {
                "metric": "embedding_manifest_sha256",
                "value": sha256_file(args.output_dir / "stage2c_embedding_manifest.tsv"),
            },
            {"metric": "elapsed_seconds", "value": time.time() - started},
        ]
    )
    write_tsv(summary, args.output_dir / "stage2c_embedding_summary.tsv")
    (args.output_dir / ".stage2c_embeddings_ready").write_text("OK\n", encoding="utf-8")
    print(provenance.to_string(index=False), flush=True)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"FATAL: {type(exc).__name__}: {exc}", file=sys.stderr, flush=True)
        raise

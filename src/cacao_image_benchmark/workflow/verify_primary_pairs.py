#!/usr/bin/env python3
"""Geometric and photometric verification of candidate image-lineage pairs.

This script emits raw evidence only. Final relation classes are assigned by the
geometric verification aggregation script after control-pair diagnostics are available.
"""
from __future__ import annotations

import argparse
import hashlib
import math
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from functools import lru_cache
from pathlib import Path
from typing import Any, Iterable

import cv2
import numpy as np
import pandas as pd
from PIL import Image, ImageEnhance, ImageOps

cv2.setNumThreads(1)


def write_tsv(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, sep="\t", index=False, na_rep="")


def safe_float(value: Any, default: float = float("nan")) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def load_rgb(path: str) -> np.ndarray:
    with Image.open(path) as raw:
        image = ImageOps.exif_transpose(raw).convert("RGB")
        return np.asarray(image, dtype=np.uint8)


def encode_jpeg(rgb: np.ndarray, quality: int) -> np.ndarray:
    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    ok, encoded = cv2.imencode(".jpg", bgr, [int(cv2.IMWRITE_JPEG_QUALITY), int(quality)])
    if not ok:
        raise RuntimeError("JPEG encoding failed")
    decoded = cv2.imdecode(encoded, cv2.IMREAD_COLOR)
    if decoded is None:
        raise RuntimeError("JPEG decoding failed")
    return cv2.cvtColor(decoded, cv2.COLOR_BGR2RGB)


def apply_synthetic_transform(rgb: np.ndarray, transform: str) -> np.ndarray:
    transform = (transform or "").strip()
    if not transform:
        return rgb.copy()
    if transform == "jpeg_q45":
        return encode_jpeg(rgb, 45)
    if transform == "brightness_065":
        return np.clip(rgb.astype(np.float32) * 0.65, 0, 255).astype(np.uint8)
    if transform == "brightness_135":
        return np.clip(rgb.astype(np.float32) * 1.35, 0, 255).astype(np.uint8)
    if transform == "center_crop_15":
        h, w = rgb.shape[:2]
        y = max(1, int(round(h * 0.075)))
        x = max(1, int(round(w * 0.075)))
        return rgb[y : h - y, x : w - x].copy()
    if transform == "corner_crop_20":
        h, w = rgb.shape[:2]
        return rgb[: max(2, int(h * 0.80)), : max(2, int(w * 0.80))].copy()
    if transform == "rotate90":
        return np.ascontiguousarray(np.rot90(rgb, 1))
    if transform == "mirror_horizontal":
        return np.ascontiguousarray(np.fliplr(rgb))
    if transform == "resize_045_jpeg60":
        h, w = rgb.shape[:2]
        down = cv2.resize(rgb, (max(32, int(w * 0.45)), max(32, int(h * 0.45))), interpolation=cv2.INTER_AREA)
        return encode_jpeg(down, 60)
    if transform == "combined_crop_brightness_jpeg":
        h, w = rgb.shape[:2]
        y = max(1, int(round(h * 0.10)))
        x = max(1, int(round(w * 0.10)))
        cropped = rgb[y : h - y, x : w - x]
        bright = np.clip(cropped.astype(np.float32) * 0.78 + 12.0, 0, 255).astype(np.uint8)
        return encode_jpeg(bright, 55)
    raise ValueError(f"Unknown synthetic transform: {transform}")


def resize_max(rgb: np.ndarray, max_dim: int) -> tuple[np.ndarray, float]:
    h, w = rgb.shape[:2]
    if max(h, w) <= max_dim:
        return np.ascontiguousarray(rgb), 1.0
    scale = max_dim / float(max(h, w))
    out = cv2.resize(rgb, (max(2, int(round(w * scale))), max(2, int(round(h * scale)))), interpolation=cv2.INTER_AREA)
    return np.ascontiguousarray(out), scale


def d4_variants(rgb: np.ndarray) -> list[tuple[str, np.ndarray]]:
    mirrored = np.ascontiguousarray(np.fliplr(rgb))
    variants = [
        ("identity", rgb),
        ("rot90", np.rot90(rgb, 1)),
        ("rot180", np.rot90(rgb, 2)),
        ("rot270", np.rot90(rgb, 3)),
        ("mirror", mirrored),
        ("mirror_rot90", np.rot90(mirrored, 1)),
        ("mirror_rot180", np.rot90(mirrored, 2)),
        ("mirror_rot270", np.rot90(mirrored, 3)),
    ]
    return [(name, np.ascontiguousarray(image)) for name, image in variants]


def gray_u8(rgb: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)


def normalized_corr(a: np.ndarray, b: np.ndarray, mask: np.ndarray | None = None) -> float:
    if mask is None:
        x = a.astype(np.float64).reshape(-1)
        y = b.astype(np.float64).reshape(-1)
    else:
        valid = mask.astype(bool)
        if valid.sum() < 64:
            return float("nan")
        x = a[valid].astype(np.float64)
        y = b[valid].astype(np.float64)
    x -= x.mean()
    y -= y.mean()
    denom = math.sqrt(float(np.dot(x, x)) * float(np.dot(y, y)))
    if denom <= 1e-12:
        return 1.0 if np.allclose(x, y) else 0.0
    return float(np.dot(x, y) / denom)


def gradient_magnitude(gray: np.ndarray) -> np.ndarray:
    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    return cv2.magnitude(gx, gy)


def masked_ssim(a: np.ndarray, b: np.ndarray, mask: np.ndarray | None = None) -> float:
    # Standard Gaussian-window SSIM, averaged only over the valid overlap mask.
    a = a.astype(np.float32)
    b = b.astype(np.float32)
    c1 = (0.01 * 255.0) ** 2
    c2 = (0.03 * 255.0) ** 2
    mu_a = cv2.GaussianBlur(a, (11, 11), 1.5)
    mu_b = cv2.GaussianBlur(b, (11, 11), 1.5)
    mu_a2 = mu_a * mu_a
    mu_b2 = mu_b * mu_b
    mu_ab = mu_a * mu_b
    sigma_a2 = cv2.GaussianBlur(a * a, (11, 11), 1.5) - mu_a2
    sigma_b2 = cv2.GaussianBlur(b * b, (11, 11), 1.5) - mu_b2
    sigma_ab = cv2.GaussianBlur(a * b, (11, 11), 1.5) - mu_ab
    numerator = (2.0 * mu_ab + c1) * (2.0 * sigma_ab + c2)
    denominator = (mu_a2 + mu_b2 + c1) * (sigma_a2 + sigma_b2 + c2)
    score_map = numerator / np.maximum(denominator, 1e-12)
    if mask is None:
        return float(np.nanmean(score_map))
    valid = mask.astype(bool)
    if valid.sum() < 64:
        return float("nan")
    return float(np.nanmean(score_map[valid]))


def linear_match_b_to_a(a: np.ndarray, b: np.ndarray, mask: np.ndarray) -> np.ndarray:
    valid = mask.astype(bool)
    if valid.sum() < 64:
        return b
    x = b[valid].astype(np.float64)
    y = a[valid].astype(np.float64)
    vx = float(np.var(x))
    if vx < 1e-8:
        return b
    alpha = float(np.cov(x, y, ddof=0)[0, 1] / vx)
    beta = float(y.mean() - alpha * x.mean())
    adjusted = alpha * b.astype(np.float64) + beta
    return np.clip(adjusted, 0, 255).astype(np.uint8)


def direct_similarity(a_rgb: np.ndarray, b_rgb: np.ndarray, size: int = 384) -> dict[str, Any]:
    a_ratio = a_rgb.shape[1] / max(1.0, float(a_rgb.shape[0]))
    best: dict[str, Any] = {
        "direct_orientation": "",
        "direct_ncc": float("nan"),
        "direct_gradient_ncc": float("nan"),
        "direct_ssim": float("nan"),
        "direct_score": -999.0,
    }
    a_small = cv2.resize(a_rgb, (size, size), interpolation=cv2.INTER_AREA)
    a_gray = gray_u8(a_small)
    a_grad = gradient_magnitude(a_gray)
    for orientation, variant in d4_variants(b_rgb):
        b_ratio = variant.shape[1] / max(1.0, float(variant.shape[0]))
        ratio_difference = abs(math.log(max(a_ratio, 1e-6) / max(b_ratio, 1e-6)))
        if ratio_difference > 0.12:
            continue
        b_small = cv2.resize(variant, (size, size), interpolation=cv2.INTER_AREA)
        b_gray = gray_u8(b_small)
        ncc = normalized_corr(a_gray, b_gray)
        grad = normalized_corr(a_grad, gradient_magnitude(b_gray))
        adjusted = linear_match_b_to_a(a_gray, b_gray, np.ones_like(a_gray, dtype=np.uint8))
        ssim = masked_ssim(a_gray, adjusted)
        score = 0.40 * ncc + 0.30 * grad + 0.30 * ssim
        if score > best["direct_score"]:
            best = {
                "direct_orientation": orientation,
                "direct_ncc": ncc,
                "direct_gradient_ncc": grad,
                "direct_ssim": ssim,
                "direct_score": score,
            }
    return best


def clahe_gray(rgb: np.ndarray) -> np.ndarray:
    gray = gray_u8(rgb)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    return clahe.apply(gray)


def ratio_matches(desc_src: np.ndarray, desc_dst: np.ndarray, norm: int, ratio: float) -> list[cv2.DMatch]:
    matcher = cv2.BFMatcher(norm)
    raw = matcher.knnMatch(desc_src, desc_dst, k=2)
    good: list[cv2.DMatch] = []
    for pair in raw:
        if len(pair) < 2:
            continue
        first, second = pair
        if first.distance < ratio * second.distance:
            good.append(first)
    return good


def mutual_ratio_matches(desc_b: np.ndarray, desc_a: np.ndarray, norm: int, ratio: float) -> list[cv2.DMatch]:
    ba = ratio_matches(desc_b, desc_a, norm, ratio)
    ab = ratio_matches(desc_a, desc_b, norm, ratio)
    reverse = {(m.trainIdx, m.queryIdx) for m in ab}
    mutual = [m for m in ba if (m.queryIdx, m.trainIdx) in reverse]
    if len(mutual) >= 6:
        return mutual
    return ba


def hull_coverage(points: np.ndarray, width: int, height: int) -> float:
    if len(points) < 3 or width <= 0 or height <= 0:
        return 0.0
    hull = cv2.convexHull(points.astype(np.float32))
    area = abs(float(cv2.contourArea(hull)))
    return min(1.0, area / float(width * height))


def reprojection_errors(matrix: np.ndarray, src: np.ndarray, dst: np.ndarray) -> np.ndarray:
    src_h = np.concatenate([src.astype(np.float64), np.ones((len(src), 1), dtype=np.float64)], axis=1)
    projected = (matrix.astype(np.float64) @ src_h.T).T
    denom = projected[:, 2:3]
    denom[np.abs(denom) < 1e-12] = np.nan
    xy = projected[:, :2] / denom
    return np.linalg.norm(xy - dst.astype(np.float64), axis=1)


def estimate_models(src_b: np.ndarray, dst_a: np.ndarray, shape_b: tuple[int, int], shape_a: tuple[int, int]) -> list[dict[str, Any]]:
    candidates: list[dict[str, Any]] = []
    if len(src_b) < 4:
        return candidates
    h_a, w_a = shape_a
    h_b, w_b = shape_b

    # Affine partial model.
    affine, affine_mask = cv2.estimateAffinePartial2D(
        src_b,
        dst_a,
        method=cv2.RANSAC,
        ransacReprojThreshold=4.0,
        maxIters=5000,
        confidence=0.999,
        refineIters=20,
    )
    if affine is not None and affine_mask is not None:
        matrix = np.vstack([affine, [0.0, 0.0, 1.0]])
        inliers = affine_mask.reshape(-1).astype(bool)
        errors = reprojection_errors(matrix, src_b, dst_a)
        candidates.append(
            {
                "model_type": "affine_partial",
                "matrix": matrix,
                "inlier_mask": inliers,
                "inliers": int(inliers.sum()),
                "median_reprojection_error": float(np.nanmedian(errors[inliers])) if inliers.any() else float("nan"),
                "coverage_a": hull_coverage(dst_a[inliers], w_a, h_a),
                "coverage_b": hull_coverage(src_b[inliers], w_b, h_b),
            }
        )

    method = getattr(cv2, "USAC_MAGSAC", cv2.RANSAC)
    homography, homography_mask = cv2.findHomography(
        src_b,
        dst_a,
        method,
        ransacReprojThreshold=4.0,
        maxIters=10000,
        confidence=0.999,
    )
    if homography is not None and homography_mask is not None:
        inliers = homography_mask.reshape(-1).astype(bool)
        errors = reprojection_errors(homography, src_b, dst_a)
        candidates.append(
            {
                "model_type": "homography",
                "matrix": homography,
                "inlier_mask": inliers,
                "inliers": int(inliers.sum()),
                "median_reprojection_error": float(np.nanmedian(errors[inliers])) if inliers.any() else float("nan"),
                "coverage_a": hull_coverage(dst_a[inliers], w_a, h_a),
                "coverage_b": hull_coverage(src_b[inliers], w_b, h_b),
            }
        )
    return candidates


def projected_polygon(matrix: np.ndarray, width: int, height: int) -> np.ndarray:
    corners = np.array([[[0.0, 0.0]], [[width - 1.0, 0.0]], [[width - 1.0, height - 1.0]], [[0.0, height - 1.0]]], dtype=np.float32)
    return cv2.perspectiveTransform(corners, matrix.astype(np.float64)).reshape(-1, 2)


def photometric_after_warp(a_rgb: np.ndarray, b_rgb: np.ndarray, matrix: np.ndarray) -> dict[str, float]:
    h_a, w_a = a_rgb.shape[:2]
    h_b, w_b = b_rgb.shape[:2]
    warped_b = cv2.warpPerspective(b_rgb, matrix, (w_a, h_a), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
    mask_b = cv2.warpPerspective(np.ones((h_b, w_b), dtype=np.uint8), matrix, (w_a, h_a), flags=cv2.INTER_NEAREST)
    valid = mask_b > 0
    if valid.sum() < 256:
        return {
            "overlap_pixels": int(valid.sum()),
            "overlap_fraction_a": 0.0,
            "overlap_fraction_b": 0.0,
            "overlap_fraction_smaller": 0.0,
            "aligned_ncc": float("nan"),
            "aligned_gradient_ncc": float("nan"),
            "aligned_ssim": float("nan"),
        }
    valid_u8 = valid.astype(np.uint8)
    valid_u8 = cv2.erode(valid_u8, np.ones((5, 5), np.uint8), iterations=1)
    valid = valid_u8.astype(bool)
    overlap = int(valid.sum())
    polygon = projected_polygon(matrix, w_b, h_b)
    projected_area = max(1.0, abs(float(cv2.contourArea(polygon.astype(np.float32)))))
    area_a = float(w_a * h_a)
    gray_a = gray_u8(a_rgb)
    gray_b = gray_u8(warped_b)
    adjusted_b = linear_match_b_to_a(gray_a, gray_b, valid_u8)
    return {
        "overlap_pixels": overlap,
        "overlap_fraction_a": min(1.0, overlap / area_a),
        "overlap_fraction_b": min(1.0, overlap / projected_area),
        "overlap_fraction_smaller": min(1.0, overlap / min(area_a, projected_area)),
        "aligned_ncc": normalized_corr(gray_a, gray_b, valid_u8),
        "aligned_gradient_ncc": normalized_corr(gradient_magnitude(gray_a), gradient_magnitude(gray_b), valid_u8),
        "aligned_ssim": masked_ssim(gray_a, adjusted_b, valid_u8),
    }


def feature_detector(method: str):
    if method == "SIFT":
        return cv2.SIFT_create(nfeatures=5000, contrastThreshold=0.015, edgeThreshold=12, sigma=1.6), cv2.NORM_L2, 0.75
    if method == "ORB":
        return cv2.ORB_create(nfeatures=6000, scaleFactor=1.2, nlevels=8, edgeThreshold=15, fastThreshold=7), cv2.NORM_HAMMING, 0.82
    raise ValueError(method)


def feature_result(a_rgb: np.ndarray, b_rgb: np.ndarray, method: str, parity: str) -> dict[str, Any]:
    b_variant = np.ascontiguousarray(np.fliplr(b_rgb)) if parity == "mirror" else b_rgb
    gray_a = clahe_gray(a_rgb)
    gray_b = clahe_gray(b_variant)
    detector, norm, ratio = feature_detector(method)
    kp_a, desc_a = detector.detectAndCompute(gray_a, None)
    kp_b, desc_b = detector.detectAndCompute(gray_b, None)
    result: dict[str, Any] = {
        "method": method,
        "parity": parity,
        "keypoints_a": len(kp_a) if kp_a is not None else 0,
        "keypoints_b": len(kp_b) if kp_b is not None else 0,
        "good_matches": 0,
        "inliers": 0,
        "inlier_ratio": 0.0,
        "model_type": "",
        "median_reprojection_error": float("nan"),
        "coverage_a": 0.0,
        "coverage_b": 0.0,
        "overlap_pixels": 0,
        "overlap_fraction_a": 0.0,
        "overlap_fraction_b": 0.0,
        "overlap_fraction_smaller": 0.0,
        "aligned_ncc": float("nan"),
        "aligned_gradient_ncc": float("nan"),
        "aligned_ssim": float("nan"),
        "matrix": None,
    }
    if desc_a is None or desc_b is None or len(kp_a) < 4 or len(kp_b) < 4:
        return result
    matches = mutual_ratio_matches(desc_b, desc_a, norm, ratio)
    result["good_matches"] = len(matches)
    if len(matches) < 4:
        return result
    src_b = np.float32([kp_b[m.queryIdx].pt for m in matches])
    dst_a = np.float32([kp_a[m.trainIdx].pt for m in matches])
    models = estimate_models(src_b, dst_a, gray_b.shape, gray_a.shape)
    if not models:
        return result
    for model in models:
        model["inlier_ratio"] = model["inliers"] / max(1, len(matches))
        # Conservative geometric ranking: favor spatial support and low error.
        error = model["median_reprojection_error"]
        error_term = 0.0 if not np.isfinite(error) else max(0.0, 1.0 - error / 12.0)
        model["rank_score"] = (
            model["inliers"]
            + 12.0 * model["inlier_ratio"]
            + 25.0 * min(model["coverage_a"], 0.25)
            + 25.0 * min(model["coverage_b"], 0.25)
            + 5.0 * error_term
        )
    best = max(models, key=lambda x: (x["rank_score"], x["inliers"], x["inlier_ratio"]))
    photo = photometric_after_warp(a_rgb, b_variant, best["matrix"])
    result.update(
        {
            "inliers": best["inliers"],
            "inlier_ratio": best["inlier_ratio"],
            "model_type": best["model_type"],
            "median_reprojection_error": best["median_reprojection_error"],
            "coverage_a": best["coverage_a"],
            "coverage_b": best["coverage_b"],
            "matrix": best["matrix"],
            **photo,
        }
    )
    return result


def choose_best_feature(results: Iterable[dict[str, Any]]) -> dict[str, Any]:
    def value(x: Any, default: float = -999.0) -> float:
        try:
            x = float(x)
            return x if math.isfinite(x) else default
        except Exception:
            return default

    def key(row: dict[str, Any]):
        return (
            int(row.get("inliers", 0)),
            value(row.get("inlier_ratio")),
            value(row.get("aligned_ncc")),
            value(row.get("aligned_ssim")),
            value(row.get("overlap_fraction_smaller")),
            -value(row.get("median_reprojection_error"), 999.0),
        )
    return max(list(results), key=key)


def verify_images(a_rgb_original: np.ndarray, b_rgb_original: np.ndarray, max_dim: int) -> dict[str, Any]:
    direct = direct_similarity(a_rgb_original, b_rgb_original)
    a_rgb, scale_a = resize_max(a_rgb_original, max_dim)
    b_rgb, scale_b = resize_max(b_rgb_original, max_dim)

    feature_results: list[dict[str, Any]] = []
    for parity in ("identity", "mirror"):
        feature_results.append(feature_result(a_rgb, b_rgb, "SIFT", parity))
    best_sift = choose_best_feature(feature_results)

    # ORB is a complementary fallback and confirmation layer.
    orb_results: list[dict[str, Any]] = []
    for parity in ("identity", "mirror"):
        orb_results.append(feature_result(a_rgb, b_rgb, "ORB", parity))
    best_orb = choose_best_feature(orb_results)
    best = choose_best_feature([best_sift, best_orb])

    matrix = best.pop("matrix", None)
    matrix_text = ""
    if matrix is not None:
        matrix_text = ";".join(f"{float(v):.8g}" for v in np.asarray(matrix).reshape(-1))

    output = {
        **direct,
        "resize_scale_a": scale_a,
        "resize_scale_b": scale_b,
        "best_method": best.get("method", ""),
        "best_parity": best.get("parity", ""),
        "best_keypoints_a": best.get("keypoints_a", 0),
        "best_keypoints_b": best.get("keypoints_b", 0),
        "best_good_matches": best.get("good_matches", 0),
        "best_inliers": best.get("inliers", 0),
        "best_inlier_ratio": best.get("inlier_ratio", 0.0),
        "best_model_type": best.get("model_type", ""),
        "best_median_reprojection_error": best.get("median_reprojection_error", float("nan")),
        "best_coverage_a": best.get("coverage_a", 0.0),
        "best_coverage_b": best.get("coverage_b", 0.0),
        "best_overlap_pixels": best.get("overlap_pixels", 0),
        "best_overlap_fraction_a": best.get("overlap_fraction_a", 0.0),
        "best_overlap_fraction_b": best.get("overlap_fraction_b", 0.0),
        "best_overlap_fraction_smaller": best.get("overlap_fraction_smaller", 0.0),
        "best_aligned_ncc": best.get("aligned_ncc", float("nan")),
        "best_aligned_gradient_ncc": best.get("aligned_gradient_ncc", float("nan")),
        "best_aligned_ssim": best.get("aligned_ssim", float("nan")),
        "best_transform_matrix": matrix_text,
        "sift_inliers": best_sift.get("inliers", 0),
        "sift_inlier_ratio": best_sift.get("inlier_ratio", 0.0),
        "sift_aligned_ncc": best_sift.get("aligned_ncc", float("nan")),
        "sift_aligned_ssim": best_sift.get("aligned_ssim", float("nan")),
        "orb_inliers": best_orb.get("inliers", 0),
        "orb_inlier_ratio": best_orb.get("inlier_ratio", 0.0),
        "orb_aligned_ncc": best_orb.get("aligned_ncc", float("nan")),
        "orb_aligned_ssim": best_orb.get("aligned_ssim", float("nan")),
    }
    return output


def process_row(row: dict[str, Any], max_dim: int) -> dict[str, Any]:
    start = time.time()
    base = dict(row)
    base.update({"verification_status": "", "verification_error": "", "elapsed_seconds": float("nan")})
    try:
        path_a = str(row.get("absolute_path_a") or row.get("path_a") or "")
        path_b = str(row.get("absolute_path_b") or row.get("path_b") or "")
        synthetic_transform = str(row.get("synthetic_transform") or "")
        if not path_a:
            raise FileNotFoundError("path_a is empty")
        a_rgb = load_rgb(path_a)
        if synthetic_transform:
            b_rgb = apply_synthetic_transform(a_rgb, synthetic_transform)
        else:
            if not path_b:
                raise FileNotFoundError("path_b is empty")
            b_rgb = load_rgb(path_b)
        metrics = verify_images(a_rgb, b_rgb, max_dim=max_dim)
        base.update(metrics)
        base["verification_status"] = "COMPLETED"
    except Exception as exc:
        base["verification_status"] = "FAILED"
        base["verification_error"] = f"{type(exc).__name__}:{exc}"
    base["elapsed_seconds"] = time.time() - start
    return base


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pair-table", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--status-output", required=True, type=Path)
    parser.add_argument("--task-id", required=True, type=int)
    parser.add_argument("--task-count", required=True, type=int)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--max-dim", type=int, default=1400)
    args = parser.parse_args()

    table = pd.read_csv(args.pair_table, sep="\t", keep_default_na=False, low_memory=False)
    if "pair_id" not in table.columns:
        raise RuntimeError("pair table lacks pair_id")
    table = table.sort_values("pair_id").reset_index(drop=True)
    subset = table[table.index % args.task_count == (args.task_id - 1)].copy()
    records = subset.to_dict("records")
    print(f"task_id={args.task_id} task_count={args.task_count} pairs={len(records)} workers={args.workers} max_dim={args.max_dim}")
    started = time.time()
    results: list[dict[str, Any]] = []
    with ThreadPoolExecutor(max_workers=max(1, args.workers)) as executor:
        for index, result in enumerate(executor.map(lambda row: process_row(row, args.max_dim), records), 1):
            results.append(result)
            if index % 25 == 0 or index == len(records):
                print(f"progress={index}/{len(records)}")
    output = pd.DataFrame(results)
    write_tsv(output, args.output)
    completed = int(output.get("verification_status", pd.Series(dtype=str)).eq("COMPLETED").sum())
    failed = int(output.get("verification_status", pd.Series(dtype=str)).eq("FAILED").sum())
    status = pd.DataFrame([
        {
            "task_id": args.task_id,
            "task_count": args.task_count,
            "pairs_expected": len(records),
            "rows_written": len(output),
            "completed": completed,
            "failed": failed,
            "elapsed_seconds": time.time() - started,
            "output_path": str(args.output),
            "status": "COMPLETED" if failed == 0 and len(output) == len(records) else "FAILED",
        }
    ])
    write_tsv(status, args.status_output)
    print(status.to_string(index=False))
    if failed or len(output) != len(records):
        raise RuntimeError(f"verification task incomplete: expected={len(records)} rows={len(output)} failed={failed}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"FATAL: {type(exc).__name__}: {exc}", file=sys.stderr)
        raise

from pathlib import Path
import subprocess
import sys


def test_cli_lists_workflow_commands():
    result = subprocess.run([sys.executable, "-m", "cacao_image_benchmark.cli", "list"], capture_output=True, text=True, check=True)
    commands = set(result.stdout.splitlines())
    assert "download_public_data" in commands
    assert "freeze_lineages_and_split_blocks" in commands
    assert "train_end_to_end_sensitivity" in commands


def test_public_release_validator():
    root = Path(__file__).resolve().parents[1]
    subprocess.run([sys.executable, str(root / "scripts" / "validate_release.py")], check=True)


def test_frozen_verifier_detects_transformation_and_rejects_unrelated_image():
    import cv2
    import numpy as np
    import pandas as pd
    from cacao_image_benchmark.workflow.verify_image_pairs import apply_synthetic_transform, verify_images
    from cacao_image_benchmark.workflow.build_initial_lineages import classify

    height, width = 300, 420
    y, x = np.mgrid[:height, :width]
    base = np.zeros((height, width, 3), dtype=np.uint8)
    base[..., 0] = (x * 255 // width).astype(np.uint8)
    base[..., 1] = (y * 255 // height).astype(np.uint8)
    base[..., 2] = ((x + y) % 256).astype(np.uint8)
    for index in range(12):
        center = (30 + (index * 61) % 350, 25 + (index * 43) % 240)
        cv2.circle(base, center, 8 + (index % 4) * 4, (230 - index * 8, index * 17 % 255, 120), -1)
    cv2.putText(base, "CACAO", (85, 165), cv2.FONT_HERSHEY_SIMPLEX, 1.4, (255, 255, 255), 4, cv2.LINE_AA)

    transformed = apply_synthetic_transform(base, "combined_crop_brightness_jpeg")
    transformed_relation = classify(pd.Series(verify_images(base, transformed, 1024)))[0]
    assert transformed_relation in {"VERIFIED_DERIVATIVE_STRONG", "VERIFIED_DERIVATIVE_MODERATE"}

    unrelated = np.random.default_rng(7).integers(0, 256, size=base.shape, dtype=np.uint8)
    unrelated_relation = classify(pd.Series(verify_images(base, unrelated, 1024)))[0]
    assert unrelated_relation == "REJECTED_NOT_DERIVATIVE"

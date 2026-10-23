"""
tests/test_metrics.py
---------------------
src/evaluation/metrics.py icin pytest birim testleri.
"""

from __future__ import annotations

import time
import numpy as np
import pytest

from src.evaluation.metrics import (
    psnr,
    ssim,
    sharpness_laplacian_var,
    foreground_ratio,
    precision,
    recall,
    f1_score,
    iou_score,
    rectification_ssim,
    timed,
    timed_median,
)


@pytest.fixture
def clean_image() -> np.ndarray:
    rng = np.random.default_rng(42)
    return rng.integers(50, 200, size=(50, 50), dtype=np.uint8)


def test_psnr_identical_images(clean_image: np.ndarray) -> None:
    score = psnr(clean_image, clean_image)
    assert score == 100.0


def test_psnr_noisy_image_is_lower(clean_image: np.ndarray) -> None:
    noisy = clean_image.copy()
    noisy[0:10, 0:10] = 0
    score = psnr(clean_image, noisy)
    assert score < 100.0
    assert score > 0.0


def test_psnr_shape_mismatch(clean_image: np.ndarray) -> None:
    diff_shape = np.zeros((30, 30), dtype=np.uint8)
    with pytest.raises(ValueError, match=r"Şekil uyuşmazlığı|ref="):
        psnr(clean_image, diff_shape)


def test_psnr_dtype_mismatch(clean_image: np.ndarray) -> None:
    diff_dtype = clean_image.astype(np.float32)
    with pytest.raises(ValueError, match="Veri tipi"):
        psnr(clean_image, diff_dtype)


def test_ssim_identical_images(clean_image: np.ndarray) -> None:
    score = ssim(clean_image, clean_image)
    assert pytest.approx(score, rel=1e-4) == 1.0


def test_ssim_noisy_image_is_lower(clean_image: np.ndarray) -> None:
    noisy = clean_image.copy()
    noisy[0:20, 0:20] = 0
    score = ssim(clean_image, noisy)
    assert score < 1.0


def test_sharpness_laplacian_var() -> None:
    sharp = np.zeros((50, 50), dtype=np.uint8)
    sharp[10:40, 10:40] = 255
    var_sharp = sharpness_laplacian_var(sharp)

    uniform = np.full((50, 50), 128, dtype=np.uint8)
    var_uniform = sharpness_laplacian_var(uniform)

    assert var_sharp > var_uniform
    assert var_uniform == 0.0


def test_foreground_ratio() -> None:
    empty = np.zeros((20, 20), dtype=np.uint8)
    assert foreground_ratio(empty) == 0.0

    full = np.full((20, 20), 255, dtype=np.uint8)
    assert foreground_ratio(full) == 1.0

    half = np.zeros((20, 20), dtype=np.uint8)
    half[:10, :] = 255
    assert foreground_ratio(half) == 0.5


def test_segmentation_metrics_identical_masks() -> None:
    gt = np.zeros((50, 50), dtype=np.uint8)
    gt[10:30, 10:30] = 255
    pred = gt.copy()

    assert precision(gt, pred) == 1.0
    assert recall(gt, pred) == 1.0
    assert f1_score(gt, pred) == 1.0
    assert iou_score(gt, pred) == 1.0


def test_segmentation_metrics_partial_overlap() -> None:
    gt = np.zeros((20, 20), dtype=np.uint8)
    gt[0:10, 0:10] = 255  # 100 piksel TP

    pred = np.zeros((20, 20), dtype=np.uint8)
    pred[0:10, 0:5] = 255   # 50 piksel TP
    pred[0:10, 10:15] = 255 # 50 piksel FP

    # TP = 50, FP = 50, FN = 50
    # Precision = 50 / 100 = 0.5
    # Recall = 50 / 100 = 0.5
    # F1 = 0.5
    # IoU = 50 / (50 + 50 + 50) = 50 / 150 = 1/3
    assert precision(gt, pred) == 0.5
    assert recall(gt, pred) == 0.5
    assert f1_score(gt, pred) == 0.5
    assert pytest.approx(iou_score(gt, pred), rel=1e-4) == 1.0 / 3.0


def test_rectification_ssim_checks_rotated_non_square_reference() -> None:
    image = np.random.default_rng(17).integers(0, 256, size=(30, 50), dtype=np.uint8)
    rotated = np.ascontiguousarray(np.rot90(image))
    assert rectification_ssim(image, rotated) > 0.999


def test_timed_and_timed_median() -> None:
    def dummy_task(x: int) -> int:
        return x * 2

    res, duration_ms = timed(dummy_task, 5)
    assert res == 10
    assert duration_ms >= 0.0

    res_med, med_ms = timed_median(dummy_task, 7, repeats=5, warmup=2)
    assert res_med == 14
    assert med_ms >= 0.0

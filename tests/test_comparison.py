"""
tests/test_comparison.py
------------------------
src/evaluation/comparison.py icin pytest birim testleri.
"""

from __future__ import annotations

import numpy as np
import pytest

from src.evaluation.comparison import compare_enhancements, compare_filters, compare_thresholds
from src.evaluation.noise import add_gaussian_noise


@pytest.fixture
def clean_image() -> np.ndarray:
    img = np.full((64, 64), 100, dtype=np.uint8)
    img[20:44, 20:44] = 200
    return img


def test_compare_filters_keys_and_metrics(clean_image: np.ndarray) -> None:
    noisy = add_gaussian_noise(clean_image, sigma=15.0, seed=42)
    res = compare_filters(clean_image, noisy, ksize=3)

    for method in ("gaussian", "median", "bilateral"):
        assert method in res
        entry = res[method]
        assert "image" in entry
        assert "psnr" in entry
        assert "ssim" in entry
        assert "duration_ms" in entry
        assert entry["image"].shape == clean_image.shape
        assert entry["image"].dtype == np.uint8
        assert entry["duration_ms"] >= 0.0


def test_compare_filters_mismatch_raises(clean_image: np.ndarray) -> None:
    diff = np.zeros((32, 32), dtype=np.uint8)
    with pytest.raises(ValueError, match=r"Boyut uyuşmazlığı"):
        compare_filters(clean_image, diff)


def test_compare_thresholds_keys(clean_image: np.ndarray) -> None:
    res = compare_thresholds(clean_image, invert=False)

    for method in ("global", "otsu", "adaptive_mean", "adaptive_gaussian"):
        assert method in res
        entry = res[method]
        assert "image" in entry
        assert "duration_ms" in entry
        assert "foreground_ratio" in entry
        assert entry["image"].shape == clean_image.shape
        assert set(np.unique(entry["image"])).issubset({0, 255})


def test_compare_enhancements_returns_all_methods_without_reference_metrics(clean_image: np.ndarray) -> None:
    results = compare_enhancements(clean_image)
    expected = {
        "none", "contrast_stretch", "gamma_auto", "gamma_0.6", "gamma_0.8",
        "gamma_1.2", "gamma_1.6", "gamma_2.0", "clahe_1.0", "clahe_2.0",
        "clahe_4.0", "histogram_equalization",
    }
    assert set(results) == expected
    for result in results.values():
        assert result["image"].shape == clean_image.shape
        assert result["image"].dtype == np.uint8
        assert result["duration_ms"] >= 0.0
        assert "psnr" not in result and "ssim" not in result
    assert results["gamma_auto"]["parameters"]["selected_method"] == "gamma"
    assert results["gamma_auto"]["parameters"]["gamma"] > 0

    bright = np.full_like(clean_image, 245)
    bright_result = compare_enhancements(bright)["gamma_auto"]
    assert bright_result["parameters"]["gamma"] < 1.0

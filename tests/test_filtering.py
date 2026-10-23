"""
tests/test_filtering.py
-----------------------
src/preprocessing/filtering.py icin pytest birim testleri.
"""

from __future__ import annotations

import numpy as np
import pytest

from src.preprocessing.filtering import gaussian_blur, median_blur, bilateral_filter
from src.evaluation.noise import add_salt_pepper_noise


@pytest.fixture
def sample_gray() -> np.ndarray:
    rng = np.random.default_rng(42)
    return rng.integers(50, 200, size=(64, 64), dtype=np.uint8)


def test_gaussian_blur_shape_and_dtype(sample_gray: np.ndarray) -> None:
    blurred = gaussian_blur(sample_gray, ksize=5, sigma=1.0)
    assert blurred.shape == sample_gray.shape
    assert blurred.dtype == np.uint8


def test_gaussian_blur_even_ksize_raises(sample_gray: np.ndarray) -> None:
    with pytest.raises(ValueError, match="ksize"):
        gaussian_blur(sample_gray, ksize=4)


def test_gaussian_blur_invalid_dim() -> None:
    rgb = np.zeros((32, 32, 3), dtype=np.uint8)
    with pytest.raises(ValueError):
        gaussian_blur(rgb, ksize=3)


def test_median_blur_shape_and_dtype(sample_gray: np.ndarray) -> None:
    blurred = median_blur(sample_gray, ksize=3)
    assert blurred.shape == sample_gray.shape
    assert blurred.dtype == np.uint8


def test_median_blur_even_ksize_raises(sample_gray: np.ndarray) -> None:
    with pytest.raises(ValueError, match="ksize"):
        median_blur(sample_gray, ksize=6)


def test_median_blur_reduces_salt_pepper_noise() -> None:
    clean = np.full((50, 50), 128, dtype=np.uint8)
    noisy = add_salt_pepper_noise(clean, amount=0.1, seed=123)

    noisy_pixels_before = int(np.count_nonzero(noisy != 128))
    assert noisy_pixels_before > 0

    filtered = median_blur(noisy, ksize=3)
    noisy_pixels_after = int(np.count_nonzero(filtered != 128))

    assert noisy_pixels_after < noisy_pixels_before


def test_bilateral_filter_shape_and_dtype(sample_gray: np.ndarray) -> None:
    filtered = bilateral_filter(sample_gray, d=5, sigma_color=50, sigma_space=50)
    assert filtered.shape == sample_gray.shape
    assert filtered.dtype == np.uint8


def test_bilateral_filter_invalid_dim() -> None:
    rgb = np.zeros((32, 32, 3), dtype=np.uint8)
    with pytest.raises(ValueError):
        bilateral_filter(rgb)

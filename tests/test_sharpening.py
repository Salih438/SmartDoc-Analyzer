"""
tests/test_sharpening.py
------------------------
src/preprocessing/sharpening.py icin pytest birim testleri.
"""

from __future__ import annotations

import numpy as np
import pytest

from src.preprocessing.sharpening import unsharp_mask, laplacian_sharpen
from src.evaluation.metrics import sharpness_laplacian_var


@pytest.fixture
def sample_gray() -> np.ndarray:
    # Keskin kenarlar iceren sentetik gri goruntu
    img = np.full((64, 64), 100, dtype=np.uint8)
    img[20:44, 20:44] = 200
    return img


def test_unsharp_mask_zero_amount_returns_equal(sample_gray: np.ndarray) -> None:
    res = unsharp_mask(sample_gray, ksize=5, sigma=1.0, amount=0.0)
    np.testing.assert_array_equal(res, sample_gray)


def test_unsharp_mask_shape_and_dtype(sample_gray: np.ndarray) -> None:
    res = unsharp_mask(sample_gray, ksize=5, sigma=1.0, amount=1.0)
    assert res.shape == sample_gray.shape
    assert res.dtype == np.uint8


def test_unsharp_mask_even_ksize_raises(sample_gray: np.ndarray) -> None:
    with pytest.raises(ValueError, match="ksize"):
        unsharp_mask(sample_gray, ksize=4)


def test_unsharp_mask_invalid_dim() -> None:
    rgb = np.zeros((32, 32, 3), dtype=np.uint8)
    with pytest.raises(ValueError):
        unsharp_mask(rgb)


def test_laplacian_sharpen_zero_weight_returns_equal(sample_gray: np.ndarray) -> None:
    res = laplacian_sharpen(sample_gray, weight=0.0)
    np.testing.assert_array_equal(res, sample_gray)


def test_laplacian_sharpen_shape_and_dtype(sample_gray: np.ndarray) -> None:
    res = laplacian_sharpen(sample_gray, weight=1.0)
    assert res.shape == sample_gray.shape
    assert res.dtype == np.uint8


def test_laplacian_sharpen_increases_sharpness(sample_gray: np.ndarray) -> None:
    # Hafif bulaniklastirilmis bir goruntude keskinlestirme yapildiginda laplacian varyansi artmali
    from src.preprocessing.filtering import gaussian_blur
    blurred = gaussian_blur(sample_gray, ksize=5, sigma=1.0)
    sharp = laplacian_sharpen(blurred, weight=1.0)

    var_blurred = sharpness_laplacian_var(blurred)
    var_sharp = sharpness_laplacian_var(sharp)
    assert var_sharp > var_blurred

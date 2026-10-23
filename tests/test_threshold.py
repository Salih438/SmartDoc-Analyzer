"""
tests/test_threshold.py
-----------------------
src/segmentation/threshold.py icin pytest birim testleri.
"""

from __future__ import annotations

import numpy as np
import pytest

from src.segmentation.threshold import global_threshold, otsu_threshold, adaptive_threshold


@pytest.fixture
def bimodal_image() -> np.ndarray:
    # Sol yari 50, sag yari 200
    img = np.zeros((64, 64), dtype=np.uint8)
    img[:, :32] = 50
    img[:, 32:] = 200
    return img


def test_global_threshold_binary_values(bimodal_image: np.ndarray) -> None:
    bin_img = global_threshold(bimodal_image, thresh=127)
    unique_vals = set(np.unique(bin_img))
    assert unique_vals.issubset({0, 255})


def test_global_threshold_invert(bimodal_image: np.ndarray) -> None:
    normal = global_threshold(bimodal_image, thresh=127, invert=False)
    inverted = global_threshold(bimodal_image, thresh=127, invert=True)

    # Birbirini tam 255'e tamamlamali
    combined = normal.astype(np.int32) + inverted.astype(np.int32)
    assert np.all(combined == 255)


def test_otsu_threshold_finds_between_modes(bimodal_image: np.ndarray) -> None:
    bin_img, thresh_val = otsu_threshold(bimodal_image)
    unique_vals = set(np.unique(bin_img))
    assert unique_vals.issubset({0, 255})

    # OpenCV Otsu uygulaması ayrık iki seviyeli (50 ve 200) görüntüde
    # varyans platosunun ilk elemani olan 50 degerini secer.
    # Bu esik (<=50 -> 0, >50 -> 255) iki seviyeyi kusursuz ayirir.
    assert 50 <= thresh_val <= 200
    assert np.all(bin_img[:, :32] == 0)
    assert np.all(bin_img[:, 32:] == 255)


def test_otsu_threshold_bimodal_distribution() -> None:
    # 50 ve 200 etrafinda iki ayrik Gauss tepe dagilimi
    rng = np.random.default_rng(42)
    c1 = rng.normal(50, 5, size=(64, 32)).clip(0, 255).astype(np.uint8)
    c2 = rng.normal(200, 5, size=(64, 32)).clip(0, 255).astype(np.uint8)
    img = np.hstack([c1, c2])

    bin_img, thresh_val = otsu_threshold(img)
    assert 50 < thresh_val < 200
    assert set(np.unique(bin_img)).issubset({0, 255})


def test_otsu_threshold_invert(bimodal_image: np.ndarray) -> None:
    normal, _ = otsu_threshold(bimodal_image, invert=False)
    inverted, _ = otsu_threshold(bimodal_image, invert=True)
    combined = normal.astype(np.int32) + inverted.astype(np.int32)
    assert np.all(combined == 255)


def test_adaptive_threshold_binary_values(bimodal_image: np.ndarray) -> None:
    bin_img = adaptive_threshold(bimodal_image, method="gaussian", block_size=11, c=2)
    unique_vals = set(np.unique(bin_img))
    assert unique_vals.issubset({0, 255})


def test_adaptive_threshold_methods(bimodal_image: np.ndarray) -> None:
    gauss = adaptive_threshold(bimodal_image, method="gaussian", block_size=11, c=2)
    mean = adaptive_threshold(bimodal_image, method="mean", block_size=11, c=2)
    assert set(np.unique(gauss)).issubset({0, 255})
    assert set(np.unique(mean)).issubset({0, 255})


def test_adaptive_threshold_even_blocksize_raises(bimodal_image: np.ndarray) -> None:
    with pytest.raises(ValueError, match="block_size"):
        adaptive_threshold(bimodal_image, block_size=10)


def test_adaptive_threshold_small_blocksize_raises(bimodal_image: np.ndarray) -> None:
    with pytest.raises(ValueError, match="block_size"):
        adaptive_threshold(bimodal_image, block_size=1)


def test_adaptive_threshold_invalid_method(bimodal_image: np.ndarray) -> None:
    with pytest.raises(ValueError, match="method"):
        adaptive_threshold(bimodal_image, method="invalid_method")

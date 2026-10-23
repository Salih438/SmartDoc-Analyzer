"""
tests/test_morphology.py
------------------------
src/segmentation/morphology.py icin pytest birim testleri.
"""

from __future__ import annotations

import numpy as np
import pytest

from src.segmentation.morphology import erode, dilate, opening, closing


def test_opening_removes_single_pixel_noise() -> None:
    # Siyah arka plan uzerinde tek bir izole beyaz piksel (gurultu)
    img = np.zeros((30, 30), dtype=np.uint8)
    img[15, 15] = 255
    assert np.count_nonzero(img) == 1

    cleaned = opening(img, ksize=3)
    assert np.count_nonzero(cleaned) == 0


def test_closing_fills_single_pixel_hole() -> None:
    # Beyaz on plan (255) blogu icinde tek piksellik siyah delik (0)
    img = np.zeros((30, 30), dtype=np.uint8)
    img[5:25, 5:25] = 255
    img[15, 15] = 0

    assert img[15, 15] == 0
    filled = closing(img, ksize=3)
    assert filled[15, 15] == 255


def test_erode_decreases_white_pixels() -> None:
    img = np.zeros((30, 30), dtype=np.uint8)
    img[5:25, 5:25] = 255
    count_before = int(np.count_nonzero(img == 255))

    eroded = erode(img, ksize=3, iterations=1)
    count_after = int(np.count_nonzero(eroded == 255))

    assert count_after < count_before


def test_dilate_increases_white_pixels() -> None:
    img = np.zeros((30, 30), dtype=np.uint8)
    img[10:20, 10:20] = 255
    count_before = int(np.count_nonzero(img == 255))

    dilated = dilate(img, ksize=3, iterations=1)
    count_after = int(np.count_nonzero(dilated == 255))

    assert count_after > count_before


def test_morphology_shapes() -> None:
    img = np.zeros((30, 30), dtype=np.uint8)
    img[10:20, 10:20] = 255

    for shape in ("rect", "ellipse", "cross"):
        e = erode(img, ksize=3, shape=shape)
        d = dilate(img, ksize=3, shape=shape)
        assert e.shape == img.shape
        assert d.shape == img.shape


def test_morphology_invalid_shape() -> None:
    img = np.zeros((30, 30), dtype=np.uint8)
    with pytest.raises(ValueError, match=r"Geçersiz şekil"):
        erode(img, shape="star")


def test_morphology_even_ksize_raises() -> None:
    img = np.zeros((30, 30), dtype=np.uint8)
    with pytest.raises(ValueError, match="ksize"):
        dilate(img, ksize=4)

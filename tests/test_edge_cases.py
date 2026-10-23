"""
tests/test_edge_cases.py
------------------------
Adim 6: Tum algoritmalar icin kenar durumlari (edge cases) birim testleri:
  1. Kucuk goruntude buyuk ksize (sinir dolgulama davranisi)
  2. uint8 disi giris (float32, int32)
  3. Renkli giris (3 kanalli BGR goruntu)
  4. Ters polarite (siyah zemin uzerine beyaz yazi vs beyaz zemin uzerine siyah yazi)
  5. Ekstrem degerler (tamamen sifir veya tamamen 255)
"""

from __future__ import annotations

import numpy as np
import pytest

from src.preprocessing.filtering import gaussian_blur, median_blur, bilateral_filter
from src.preprocessing.sharpening import unsharp_mask, laplacian_sharpen
from src.segmentation.edges import canny_edges, auto_canny
from src.segmentation.threshold import global_threshold, otsu_threshold, adaptive_threshold
from src.segmentation.morphology import erode, dilate, opening, closing
from src.evaluation.metrics import psnr, ssim, precision, recall, f1_score, iou_score, foreground_ratio


def test_small_image_large_ksize_filtering() -> None:
    small = np.full((5, 5), 100, dtype=np.uint8)
    small[2, 2] = 200

    g = gaussian_blur(small, ksize=7, sigma=1.0)
    assert g.shape == (5, 5)
    assert g.dtype == np.uint8

    m = median_blur(small, ksize=5)
    assert m.shape == (5, 5)
    assert m.dtype == np.uint8

    e = erode(small, ksize=7)
    assert e.shape == (5, 5)


@pytest.mark.parametrize(
    "fn",
    [
        lambda im: gaussian_blur(im, ksize=3),
        lambda im: median_blur(im, ksize=3),
        lambda im: bilateral_filter(im),
        lambda im: unsharp_mask(im),
        lambda im: laplacian_sharpen(im),
        lambda im: canny_edges(im, 50, 150),
        lambda im: auto_canny(im),
        lambda im: global_threshold(im),
        lambda im: otsu_threshold(im),
        lambda im: adaptive_threshold(im),
        lambda im: erode(im),
        lambda im: dilate(im),
    ],
)
def test_color_3d_input_raises_value_error(fn) -> None:
    rgb = np.zeros((32, 32, 3), dtype=np.uint8)
    with pytest.raises(ValueError):
        fn(rgb)


def test_float_input_in_metrics() -> None:
    f1 = np.ones((32, 32), dtype=np.float32) * 0.5
    f2 = np.ones((32, 32), dtype=np.float32) * 0.5
    score = psnr(f1, f2)
    assert score == 100.0


def test_polarity_reversal_threshold() -> None:
    dark_text = np.full((30, 30), 255, dtype=np.uint8)
    dark_text[10:20, 10:20] = 0

    bin_inv = global_threshold(dark_text, thresh=127, invert=True)
    assert np.all(bin_inv[10:20, 10:20] == 255)
    assert np.all(bin_inv[:10, :10] == 0)

    bin_norm = global_threshold(dark_text, thresh=127, invert=False)
    assert np.all(bin_norm[10:20, 10:20] == 0)
    assert np.all(bin_norm[:10, :10] == 255)


def test_extreme_all_zeros_and_all_ones() -> None:
    zeros = np.zeros((30, 30), dtype=np.uint8)
    ones = np.full((30, 30), 255, dtype=np.uint8)

    assert foreground_ratio(zeros) == 0.0
    assert foreground_ratio(ones) == 1.0

    assert precision(zeros, zeros) == 1.0
    assert recall(zeros, zeros) == 1.0
    assert f1_score(zeros, zeros) == 1.0
    assert iou_score(zeros, zeros) == 1.0

    assert iou_score(zeros, ones) == 0.0
    assert f1_score(zeros, ones) == 0.0

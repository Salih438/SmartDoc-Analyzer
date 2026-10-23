"""Aydınlatma düzleştirme testleri."""

from __future__ import annotations

import cv2
import numpy as np
import pytest

from src.preprocessing.illumination import _scaled_odd_ksize, flatten_background


def test_flatten_background_reduces_sloped_blank_background_std() -> None:
    height, width = 500, 700
    gradient = np.tile(np.linspace(120, 235, width, dtype=np.float32), (height, 1))
    gray = np.clip(gradient, 0, 255).astype(np.uint8)
    ink_mask = np.zeros((height, width), dtype=np.uint8)
    cv2.putText(ink_mask, "SYNTHETIC DOCUMENT", (45, 260), cv2.FONT_HERSHEY_SIMPLEX, 1.2, 255, 3)
    gray[ink_mask > 0] = 25

    flattened = flatten_background(gray, ksize=301)
    blank = ink_mask == 0

    assert flattened.dtype == np.uint8
    assert flattened.shape == gray.shape
    assert float(flattened[blank].std()) < float(gray[blank].std())


def test_flatten_background_supports_bgr_and_preserves_color_channel_relationship() -> None:
    height, width = 180, 240
    light = np.tile(np.linspace(0.55, 1.0, width, dtype=np.float32), (height, 1))
    color = np.array([90.0, 150.0, 210.0], dtype=np.float32)
    image = np.clip(light[:, :, None] * color, 0, 255).astype(np.uint8)

    flattened = flatten_background(image)
    original_hue = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)[:, :, 0].astype(np.int16)
    output_hue = cv2.cvtColor(flattened, cv2.COLOR_BGR2HSV)[:, :, 0].astype(np.int16)
    hue_distance = np.abs(original_hue - output_hue)
    hue_distance = np.minimum(hue_distance, 180 - hue_distance)

    assert flattened.shape == image.shape
    assert flattened.dtype == np.uint8
    assert float(flattened[:, :, 0].std()) < float(image[:, :, 0].std())
    assert float(np.median(hue_distance)) <= 2


def test_flatten_background_none_kernel_handles_small_images() -> None:
    small = np.full((18, 24), 190, dtype=np.uint8)
    flattened = flatten_background(small, ksize=None)

    assert flattened.shape == small.shape
    assert flattened.dtype == np.uint8


def test_small_image_matches_full_resolution_closing_reference() -> None:
    height, width = 520, 580
    ramp = np.tile(np.linspace(100, 240, width, dtype=np.float32), (height, 1))
    gray = np.clip(ramp, 0, 255).astype(np.uint8)
    kernel_size = _scaled_odd_ksize(None, gray.shape)
    kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (kernel_size, kernel_size)
    )
    background = cv2.morphologyEx(gray, cv2.MORPH_CLOSE, kernel)
    expected = cv2.divide(gray, background, scale=255)

    actual = flatten_background(gray)

    np.testing.assert_allclose(actual, expected, atol=0)


@pytest.mark.parametrize("ksize", [0, 2, 4, 2.5])
def test_flatten_background_rejects_invalid_kernel(ksize: int) -> None:
    with pytest.raises(ValueError, match="ksize"):
        flatten_background(np.full((80, 90), 200, dtype=np.uint8), ksize)  # type: ignore[arg-type]


def test_flatten_background_rejects_non_uint8() -> None:
    with pytest.raises(TypeError, match="uint8"):
        flatten_background(np.full((50, 60), 200, dtype=np.float32), 31)


def test_flatten_background_rejects_unsupported_channel_count() -> None:
    with pytest.raises(ValueError, match="BGR"):
        flatten_background(np.zeros((40, 50, 4), dtype=np.uint8))

"""Kullanıcıya dönük tarama modlarının renk ve polarite testleri."""

from __future__ import annotations

import cv2
import numpy as np
import pytest

from src.scan import make_scan


def _document() -> np.ndarray:
    height, width = 420, 340
    light = np.tile(np.linspace(190, 245, width, dtype=np.float32), (height, 1))
    image = np.repeat(light[:, :, None], 3, axis=2).astype(np.uint8)
    cv2.putText(image, "SMART DOC", (35, 210), cv2.FONT_HERSHEY_SIMPLEX, 1.3, (15, 15, 15), 3, cv2.LINE_AA)
    return image


@pytest.mark.parametrize("mode", ["color", "gray", "bw", "original"])
def test_make_scan_modes_return_uint8_with_expected_dimensions(mode: str) -> None:
    image = _document()
    result = make_scan(image, mode)  # type: ignore[arg-type]

    assert result.dtype == np.uint8
    inset_y = round(image.shape[0] * 0.01)
    inset_x = round(image.shape[1] * 0.01)
    assert result.shape[:2] == (
        image.shape[0] - 2 * inset_y,
        image.shape[1] - 2 * inset_x,
    )
    if mode == "color":
        assert result.ndim == 3 and result.shape[2] == 3
    elif mode in {"gray", "bw"}:
        assert result.ndim == 2
    else:
        assert result.ndim == 3 and result.shape[2] == 3


def test_bw_scan_is_white_background_black_ink_not_ocr_polarity() -> None:
    image = _document()
    scan = make_scan(image, "bw")
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    assert set(np.unique(scan)) <= {0, 255}
    assert scan[15, 15] == 255
    gray_inset = gray[4:-4, 3:-3]
    assert float(np.mean(scan[gray_inset < 80] == 0)) > 0.95


def test_original_scan_is_an_unchanged_copy() -> None:
    image = _document()
    before = image.copy()
    result = make_scan(image, "original")

    assert result is not image
    np.testing.assert_array_equal(result, before[4:-4, 3:-3])


def test_scan_can_disable_inset_and_parameterize_sauvola_window() -> None:
    image = _document()
    unchanged_size = make_scan(image, "gray", inset_fraction=0.0)
    binary = make_scan(image, "bw", inset_fraction=0.0, sauvola_window_size=61)

    assert unchanged_size.shape == image.shape[:2]
    assert binary.shape == image.shape[:2]
    assert set(np.unique(binary)) <= {0, 255}


def test_default_sauvola_window_is_the_measured_99_pixel_setting() -> None:
    image = _document()

    default = make_scan(image, "bw", inset_fraction=0.0)
    explicit = make_scan(image, "bw", inset_fraction=0.0, sauvola_window_size=99)

    np.testing.assert_array_equal(default, explicit)


def test_scan_rejects_invalid_inset_and_sauvola_window() -> None:
    image = _document()
    with pytest.raises(ValueError, match="inset_fraction"):
        make_scan(image, "gray", inset_fraction=0.5)
    with pytest.raises(ValueError, match="Sauvola"):
        make_scan(image, "bw", sauvola_window_size=30)


def test_make_scan_rejects_invalid_mode_dtype_and_channels() -> None:
    image = _document()
    with pytest.raises(ValueError, match="tarama modu"):
        make_scan(image, "invalid")  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="uint8"):
        make_scan(image.astype(np.float32), "gray")
    with pytest.raises(ValueError, match="BGR"):
        make_scan(np.zeros((30, 40, 4), dtype=np.uint8), "gray")

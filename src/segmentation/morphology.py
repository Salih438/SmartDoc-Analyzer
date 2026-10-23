"""
src/segmentation/morphology.py
------------------------------
İkili (binary) belge görüntüleri için morfolojik filtreleme modülü.

Aşınma (erosion), genişleme (dilation), açılma (opening) ve kapanma (closing)
işlemlerini sunar.

Önemli Konvansiyon
------------------
Bu modüldeki tüm fonksiyonlar beyaz pikselleri (255) ön plan (foreground),
siyah pikselleri (0) ise arka plan (background) olarak kabul eder.
"""

from __future__ import annotations

import cv2
import numpy as np

from src.utils.validation import validate_binary, validate_ksize, validate_uint8

_validate_binary = validate_binary


_SHAPE_MAP = {
    "rect": cv2.MORPH_RECT,
    "ellipse": cv2.MORPH_ELLIPSE,
    "cross": cv2.MORPH_CROSS,
}


def _get_structuring_element(ksize: int, shape: str, func_name: str) -> np.ndarray:
    validate_ksize(ksize, func_name)
    shape_lower = shape.lower().strip()
    if shape_lower not in _SHAPE_MAP:
        raise ValueError(
            f"{func_name}: Geçersiz şekil '{shape}'. "
            f"Desteklenenler: {list(_SHAPE_MAP.keys())}"
        )
    return cv2.getStructuringElement(_SHAPE_MAP[shape_lower], (ksize, ksize))


def erode(
    binary: np.ndarray,
    ksize: int = 3,
    iterations: int = 1,
    shape: str = "rect",
) -> np.ndarray:
    """Beyaz ön plan bölgelerini aşınmaya (erosion) uğratır."""
    validate_binary(binary, "erode")
    validate_uint8(binary, "erode")
    kernel = _get_structuring_element(ksize, shape, "erode")
    return cv2.erode(binary, kernel, iterations=iterations)


def dilate(
    binary: np.ndarray,
    ksize: int = 3,
    iterations: int = 1,
    shape: str = "rect",
) -> np.ndarray:
    """Beyaz ön plan bölgelerini genişletir (dilation)."""
    validate_binary(binary, "dilate")
    validate_uint8(binary, "dilate")
    kernel = _get_structuring_element(ksize, shape, "dilate")
    return cv2.dilate(binary, kernel, iterations=iterations)


def opening(
    binary: np.ndarray,
    ksize: int = 3,
    iterations: int = 1,
    shape: str = "rect",
) -> np.ndarray:
    """Önce aşınma sonra genişleme uygulayarak küçük beyaz gürültüleri siler."""
    validate_binary(binary, "opening")
    validate_uint8(binary, "opening")
    kernel = _get_structuring_element(ksize, shape, "opening")
    return cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel, iterations=iterations)


def closing(
    binary: np.ndarray,
    ksize: int = 3,
    iterations: int = 1,
    shape: str = "rect",
) -> np.ndarray:
    """Önce genişleme sonra aşınma uygulayarak küçük siyah delikleri kapatır."""
    validate_binary(binary, "closing")
    validate_uint8(binary, "closing")
    kernel = _get_structuring_element(ksize, shape, "closing")
    return cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel, iterations=iterations)

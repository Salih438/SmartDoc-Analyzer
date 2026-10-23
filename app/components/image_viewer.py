"""Streamlit görüntü önizlemesi; analiz girdisinin boyutunu değiştirmez."""

from __future__ import annotations

import cv2
import numpy as np


def preview_image(image: np.ndarray, max_dimension: int = 1400) -> np.ndarray:
    """Return a display-only copy scaled to fit, preserving grayscale/RGB values."""
    if image.ndim not in (2, 3):
        raise ValueError("Önizleme için 2B gri veya 3B renkli görüntü bekleniyor.")
    height, width = image.shape[:2]
    if max(height, width) <= max_dimension:
        return image.copy()
    scale = max_dimension / max(height, width)
    size = (max(1, round(width * scale)), max(1, round(height * scale)))
    return cv2.resize(image, size, interpolation=cv2.INTER_AREA)


def display_image(st, image: np.ndarray, caption: str, max_dimension: int = 1400) -> None:
    """Render an OpenCV image as RGB without changing the processing source."""
    shown = preview_image(image, max_dimension)
    if shown.ndim == 2:
        shown = cv2.cvtColor(shown, cv2.COLOR_GRAY2RGB)
    elif shown.shape[2] == 3:
        shown = cv2.cvtColor(shown, cv2.COLOR_BGR2RGB)
    elif shown.shape[2] == 4:
        shown = cv2.cvtColor(shown, cv2.COLOR_BGRA2RGBA)
    st.image(shown, caption=caption, width="stretch")

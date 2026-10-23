"""Render existing threshold/filter comparisons without winner labels."""

from __future__ import annotations

import numpy as np

from app.components.image_viewer import display_image
from src.evaluation.comparison import compare_filters, compare_thresholds


def render_comparisons(st, gray: np.ndarray) -> None:
    """Show threshold outputs and filter timings for one grayscale source."""
    st.subheader("Eşikleme karşılaştırması")
    thresholds = compare_thresholds(gray, invert=True)
    cols = st.columns(4)
    for column, (name, result) in zip(cols, thresholds.items()):
        with column:
            display_image(st, result["image"], name)
            st.caption(f"Süre: {result['duration_ms']:.3f} ms · Önplan: {result['foreground_ratio']:.3f}")

    st.subheader("Filtre çıktıları ve süreleri")
    # Same-source filtering is used only for output/timing; no clean reference exists.
    filters = compare_filters(gray, gray, ksize=5)
    cols = st.columns(3)
    for column, (name, result) in zip(cols, filters.items()):
        with column:
            display_image(st, result["image"], name)
            st.caption(f"Süre: {result['duration_ms']:.3f} ms · Kalite metriği: referans yok")

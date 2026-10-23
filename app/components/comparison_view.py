"""Render existing threshold/filter comparisons without winner labels."""

from __future__ import annotations

import numpy as np

from app.components.image_viewer import display_image
from app.i18n import t
from src.evaluation.comparison import compare_enhancements, compare_filters, compare_thresholds


def render_comparisons(st, gray: np.ndarray) -> None:
    """Aynı gri kaynakta eşik, filtre ve iyileştirme çıktılarını gösterir."""
    st.subheader(t("threshold_comparison"))
    thresholds = compare_thresholds(gray, invert=True)
    cols = st.columns(4)
    for column, (name, result) in zip(cols, thresholds.items()):
        with column:
            display_image(st, result["image"], t(f"method_{name}"))
            st.caption(t("duration_foreground", duration=result["duration_ms"], ratio=result["foreground_ratio"]))

    st.subheader(t("filter_comparison"))
    # Same-source filtering is used only for output/timing; no clean reference exists.
    filters = compare_filters(gray, gray, ksize=5)
    cols = st.columns(3)
    for column, (name, result) in zip(cols, filters.items()):
        with column:
            display_image(st, result["image"], t(f"method_{name}"))
            st.caption(t("duration_only", duration=result["duration_ms"]))

    st.subheader(t("enhancement_comparison"))
    enhancements = compare_enhancements(gray)
    columns = st.columns(4)
    for index, (name, result) in enumerate(enhancements.items()):
        with columns[index % 4]:
            display_image(st, result["image"], t(f"method_{name}"))
            st.caption(t(
                "enhancement_caption",
                duration=result["duration_ms"],
                parameters=result["parameters"],
            ))

"""
tests/test_edges.py
-------------------
src/segmentation/edges.py icin pytest birim testleri.
"""

from __future__ import annotations

import numpy as np
import pytest

from src.segmentation.edges import canny_edges, auto_canny


def test_canny_edges_uniform_image_has_no_edges() -> None:
    uniform = np.full((64, 64), 128, dtype=np.uint8)
    edges = canny_edges(uniform, low=50, high=150)
    assert np.count_nonzero(edges) == 0


def test_canny_edges_sharp_vertical_boundary() -> None:
    # Sol yarisi siyah (0), sag yarisi beyaz (255)
    img = np.zeros((64, 64), dtype=np.uint8)
    img[:, 32:] = 255
    edges = canny_edges(img, low=50, high=150)

    # Ortadaki kenarda (x=31 veya 32 civari) kenar pikselleri olmali
    assert np.count_nonzero(edges[:, 30:34]) > 0
    # Kenardan uzak bolgelerde kenar olmamali
    assert np.count_nonzero(edges[:, :20]) == 0
    assert np.count_nonzero(edges[:, 40:]) == 0


def test_canny_edges_invalid_thresholds() -> None:
    img = np.zeros((32, 32), dtype=np.uint8)
    with pytest.raises(ValueError, match="low"):
        canny_edges(img, low=200, high=100)


def test_canny_edges_invalid_dim() -> None:
    rgb = np.zeros((32, 32, 3), dtype=np.uint8)
    with pytest.raises(ValueError):
        canny_edges(rgb, 50, 150)


def test_auto_canny_output_binary() -> None:
    img = np.zeros((64, 64), dtype=np.uint8)
    img[:, 32:] = 255
    edges = auto_canny(img, sigma=0.33)

    unique_vals = set(np.unique(edges))
    assert unique_vals.issubset({0, 255})
    assert np.count_nonzero(edges) > 0


def test_auto_canny_uniform_image() -> None:
    uniform = np.full((64, 64), 128, dtype=np.uint8)
    edges = auto_canny(uniform)
    assert np.count_nonzero(edges) == 0


def test_auto_canny_negative_sigma_raises() -> None:
    img = np.zeros((32, 32), dtype=np.uint8)
    with pytest.raises(ValueError, match="sigma negatif olamaz"):
        auto_canny(img, sigma=-0.1)

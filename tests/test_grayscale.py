"""
tests/test_grayscale.py
------------------------
to_grayscale() fonksiyonu için pytest testleri.
"""

from __future__ import annotations

import numpy as np
import pytest

from src.preprocessing.grayscale import to_grayscale


class TestToGrayscale:
    def test_already_gray_returns_same_array(self):
        """2-D gri görüntü olduğu gibi döndürülmeli (kopya oluşturulmamalı)."""
        gray = np.zeros((50, 60), dtype=np.uint8)
        result = to_grayscale(gray)
        assert result is gray  # Aynı nesne olmalı

    def test_bgr_to_gray_shape(self):
        """BGR→Gri dönüşüm sonucu 2-D olmalı."""
        bgr = np.random.randint(0, 256, (100, 80, 3), dtype=np.uint8)
        result = to_grayscale(bgr)
        assert result.ndim == 2
        assert result.shape == (100, 80)

    def test_bgra_to_gray_shape(self):
        """BGRA→Gri dönüşüm sonucu 2-D olmalı."""
        bgra = np.random.randint(0, 256, (60, 90, 4), dtype=np.uint8)
        result = to_grayscale(bgra)
        assert result.ndim == 2
        assert result.shape == (60, 90)

    def test_single_channel_3d(self):
        """(H, W, 1) görüntü 2-D dizi olarak döndürülmeli."""
        img = np.zeros((30, 40, 1), dtype=np.uint8)
        result = to_grayscale(img)
        assert result.ndim == 2

    def test_bgr_gray_output_dtype(self):
        """Çıktı dtype uint8 olmalı."""
        bgr = np.random.randint(0, 256, (50, 50, 3), dtype=np.uint8)
        result = to_grayscale(bgr)
        assert result.dtype == np.uint8

    def test_pure_red_bgr_darker_than_pure_green(self):
        """ITU-R BT.601 gereği yeşil kanal kırmızıdan daha parlak görünmeli.
        
        BT.601: Y = 0.114·B + 0.587·G + 0.299·R
        Saf yeşil: 0.587 → ~150 / Saf kırmızı: 0.299 → ~76
        """
        red_bgr = np.zeros((10, 10, 3), dtype=np.uint8)
        red_bgr[:, :, 2] = 255  # R kanalı (BGR'de index 2)
        green_bgr = np.zeros((10, 10, 3), dtype=np.uint8)
        green_bgr[:, :, 1] = 255  # G kanalı (BGR'de index 1)

        gray_red = to_grayscale(red_bgr)
        gray_green = to_grayscale(green_bgr)

        assert gray_green[0, 0] > gray_red[0, 0]

    def test_invalid_ndim_raises(self):
        """1-D dizi geçirilince ValueError vermeli."""
        arr_1d = np.arange(10, dtype=np.uint8)
        with pytest.raises(ValueError):
            to_grayscale(arr_1d)

    def test_unsupported_channel_count_raises(self):
        """5-kanallı görüntü geçirilince ValueError vermeli."""
        img_5ch = np.zeros((10, 10, 5), dtype=np.uint8)
        with pytest.raises(ValueError, match="Desteklenmeyen"):
            to_grayscale(img_5ch)

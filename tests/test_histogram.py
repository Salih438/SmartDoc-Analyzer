"""
tests/test_histogram.py
------------------------
compute_histogram() ve plot_histogram() için pytest testleri.
"""

from __future__ import annotations

import numpy as np
import pytest

from src.analysis.histogram import compute_histogram, plot_histogram


class TestComputeHistogram:
    def test_output_shape(self):
        """Histogram çıktısı her zaman 256 eleman içermeli."""
        gray = np.random.randint(0, 256, (100, 100), dtype=np.uint8)
        hist = compute_histogram(gray)
        assert hist.shape == (256,)

    def test_output_dtype(self):
        """Histogram çıktısı float32 olmalı."""
        gray = np.zeros((50, 50), dtype=np.uint8)
        hist = compute_histogram(gray)
        assert hist.dtype == np.float32

    def test_total_count_equals_pixel_count(self):
        """Histogram toplamı toplam piksel sayısına eşit olmalı."""
        gray = np.random.randint(0, 256, (80, 120), dtype=np.uint8)
        hist = compute_histogram(gray)
        assert hist.sum() == pytest.approx(80 * 120)

    def test_black_image_only_zero_bin(self):
        """Tamamen siyah görüntüde yalnızca bin[0] dolu olmalı."""
        black = np.zeros((40, 40), dtype=np.uint8)
        hist = compute_histogram(black)
        assert hist[0] == pytest.approx(40 * 40)
        assert hist[1:].sum() == pytest.approx(0.0)

    def test_white_image_only_255_bin(self):
        """Tamamen beyaz görüntüde yalnızca bin[255] dolu olmalı."""
        white = np.full((30, 30), 255, dtype=np.uint8)
        hist = compute_histogram(white)
        assert hist[255] == pytest.approx(30 * 30)
        assert hist[:255].sum() == pytest.approx(0.0)

    def test_non_2d_raises(self):
        """3-D görüntü geçirilince ValueError vermeli."""
        bgr = np.zeros((50, 50, 3), dtype=np.uint8)
        with pytest.raises(ValueError, match="2-D"):
            compute_histogram(bgr)

    def test_gradient_uniform_ish_distribution(self):
        """0→255 gradyanında her bin yaklaşık eşit sayıda piksel içermeli."""
        # 256 sütun, her sütun tek bir renk değeri → her bin tam olarak dolu
        gray = np.tile(np.arange(256, dtype=np.uint8), (10, 1))
        hist = compute_histogram(gray)
        # Her değer 10 kez geçmeli
        assert hist.min() == pytest.approx(10.0)
        assert hist.max() == pytest.approx(10.0)


class TestPlotHistogram:
    def test_returns_figure_and_axes(self):
        """Fonksiyon (fig, ax) çifti döndürmeli."""
        import matplotlib.pyplot as plt
        gray = np.random.randint(0, 256, (50, 50), dtype=np.uint8)
        fig, ax = plot_histogram(gray)
        assert fig is not None
        assert ax is not None
        plt.close(fig)

    def test_existing_axes_used(self):
        """Varolan ax geçirilince yeni figür oluşturulmamalı."""
        import matplotlib.pyplot as plt
        fig_pre, ax_pre = plt.subplots()
        gray = np.zeros((20, 20), dtype=np.uint8)
        fig_ret, ax_ret = plot_histogram(gray, ax=ax_pre)
        assert ax_ret is ax_pre
        assert fig_ret is fig_pre
        plt.close(fig_pre)

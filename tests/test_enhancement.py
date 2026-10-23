"""
tests/test_enhancement.py
--------------------------
Tum enhancement fonksiyonlari icin pytest testleri.
Duzeltmeler:
  - contrast_stretch tekduze goruntu -> kopya dondurur (siyah dizi degil)
  - gamma_correction bilinen deger testi eklendi
  - histogram_equalization tekduze goruntu davranis testi duzeltildi
"""

from __future__ import annotations

import numpy as np
import pytest

from src.preprocessing.enhancement import (
    apply_clahe,
    contrast_stretch,
    document_stretch,
    gamma_correction,
    histogram_equalization,
)


# ---------------------------------------------------------------------------
# Yardimcilar
# ---------------------------------------------------------------------------

def _gradient_gray(height: int = 50, width: int = 256) -> np.ndarray:
    """0->255 yatay gradyan gri goruntu olusturur."""
    return np.tile(np.arange(256, dtype=np.uint8), (height, 1))


def _narrow_range_gray(low: int = 100, high: int = 150, size: int = 100) -> np.ndarray:
    """Dar [low, high] araliginda rastgele gri goruntu olusturur."""
    return np.random.randint(low, high + 1, (size, size), dtype=np.uint8)


# ---------------------------------------------------------------------------
# contrast_stretch
# ---------------------------------------------------------------------------

class TestContrastStretch:
    def test_output_shape_preserved(self):
        gray = _narrow_range_gray()
        out = contrast_stretch(gray)
        assert out.shape == gray.shape

    def test_output_dtype_uint8(self):
        gray = _narrow_range_gray()
        assert contrast_stretch(gray).dtype == np.uint8

    def test_range_expands(self):
        """Dar aralikli goruntude gerdirme sonrasi aralik genislemeli."""
        gray = _narrow_range_gray(100, 150)
        stretched = contrast_stretch(gray, low_pct=2, high_pct=98)
        original_range = int(gray.max()) - int(gray.min())
        stretched_range = int(stretched.max()) - int(stretched.min())
        assert stretched_range > original_range

    def test_full_range_image_nearly_unchanged(self):
        gray = _gradient_gray()
        stretched = contrast_stretch(gray, low_pct=0, high_pct=100)
        assert stretched.min() == 0
        assert stretched.max() == 255

    def test_uniform_image_returns_copy(self):
        """Tekduze goruntu -> siyah dizi degil, girisin kopyasi donmeli."""
        uniform = np.full((50, 50), 128, dtype=np.uint8)
        out = contrast_stretch(uniform)
        # Kopya olmali (ayni nesne degil) ama degerler korunmali
        assert out is not uniform
        assert np.all(out == 128)

    def test_3d_input_raises(self):
        bgr = np.zeros((50, 50, 3), dtype=np.uint8)
        with pytest.raises(ValueError):
            contrast_stretch(bgr)


class TestDocumentStretch:
    def test_white_point_stretch_handles_sparse_ink(self):
        gray = np.full((100, 100), 150, dtype=np.uint8)
        gray[:1, :] = 100

        generic = contrast_stretch(gray)
        document = document_stretch(gray)

        np.testing.assert_array_equal(generic, gray)
        assert document.min() == 0
        assert document.max() == 255

    def test_uniform_document_returns_copy(self):
        gray = np.full((20, 30), 180, dtype=np.uint8)
        result = document_stretch(gray)
        assert result is not gray
        np.testing.assert_array_equal(result, gray)

    def test_invalid_percentiles_raise(self):
        gray = np.full((20, 30), 180, dtype=np.uint8)
        with pytest.raises(ValueError, match="low_pct"):
            document_stretch(gray, low_pct=95, high_pct=2)


# ---------------------------------------------------------------------------
# gamma_correction
# ---------------------------------------------------------------------------

class TestGammaCorrection:
    def test_gamma_less_than_1_darkens(self):
        """gamma < 1 -> us > 1 -> goruntu koyulastirmali."""
        gray = _narrow_range_gray(80, 120)
        darkened = gamma_correction(gray, gamma=0.5)
        assert darkened.mean() < gray.mean()

    def test_gamma_greater_than_1_brightens(self):
        """gamma > 1 -> us < 1 -> goruntu aydinlatmali."""
        gray = _narrow_range_gray(100, 180)
        brightened = gamma_correction(gray, gamma=2.5)
        assert brightened.mean() > gray.mean()

    def test_gamma_1_identity(self):
        """gamma=1 goruntuyu degistirmemeli."""
        gray = _gradient_gray()
        result = gamma_correction(gray, gamma=1.0)
        np.testing.assert_array_equal(result, gray)

    def test_known_value_gamma2(self):
        """gamma=2 icin bilinen degeri dogrula: out=255*(64/255)^0.5 ~= 128."""
        # 64^(1/2)*255/sqrt(255) = sqrt(64/255)*255 = sqrt(64*255) ~ 127.5
        # Formul: out = 255*(in/255)^(1/gamma) = 255*(64/255)^0.5
        # = 255 * 8/sqrt(255) = 255 * 8/15.97 = 127.5 -> uint8: 127 veya 128
        single = np.array([[64]], dtype=np.uint8)
        result = gamma_correction(single, gamma=2.0)
        expected = int(round(255.0 * (64.0 / 255.0) ** 0.5))
        assert abs(int(result[0, 0]) - expected) <= 1  # LUT yuvarlamasi icin +-1

    def test_output_shape_preserved(self):
        gray = np.random.randint(0, 256, (60, 80), dtype=np.uint8)
        result = gamma_correction(gray, gamma=1.5)
        assert result.shape == gray.shape

    def test_output_dtype_uint8(self):
        gray = np.zeros((30, 30), dtype=np.uint8)
        assert gamma_correction(gray, gamma=0.5).dtype == np.uint8

    def test_zero_gamma_raises(self):
        gray = np.zeros((10, 10), dtype=np.uint8)
        with pytest.raises(ValueError, match="pozitif"):
            gamma_correction(gray, gamma=0)

    def test_negative_gamma_raises(self):
        gray = np.zeros((10, 10), dtype=np.uint8)
        with pytest.raises(ValueError):
            gamma_correction(gray, gamma=-1.0)

    def test_3d_input_raises(self):
        bgr = np.zeros((50, 50, 3), dtype=np.uint8)
        with pytest.raises(ValueError):
            gamma_correction(bgr, gamma=1.0)


# ---------------------------------------------------------------------------
# apply_clahe
# ---------------------------------------------------------------------------

class TestApplyCLAHE:
    def test_output_shape_same(self):
        gray = np.random.randint(0, 256, (100, 100), dtype=np.uint8)
        out = apply_clahe(gray)
        assert out.shape == gray.shape

    def test_output_dtype_uint8(self):
        gray = np.random.randint(0, 256, (60, 60), dtype=np.uint8)
        assert apply_clahe(gray).dtype == np.uint8

    def test_different_clip_limits(self):
        gray = _narrow_range_gray(80, 180, size=64)
        out_low = apply_clahe(gray, clip_limit=1.0)
        out_high = apply_clahe(gray, clip_limit=8.0)
        assert not np.array_equal(out_low, out_high)

    def test_3d_input_raises(self):
        bgr = np.zeros((50, 50, 3), dtype=np.uint8)
        with pytest.raises(ValueError):
            apply_clahe(bgr)


# ---------------------------------------------------------------------------
# histogram_equalization
# ---------------------------------------------------------------------------

class TestHistogramEqualization:
    def test_output_shape_preserved(self):
        gray = np.random.randint(0, 256, (80, 80), dtype=np.uint8)
        out = histogram_equalization(gray)
        assert out.shape == gray.shape

    def test_output_dtype_uint8(self):
        gray = np.random.randint(0, 256, (50, 50), dtype=np.uint8)
        assert histogram_equalization(gray).dtype == np.uint8

    def test_narrow_range_expands(self):
        """Dar aralikli giris, esitleme sonrasi daha genis araligi yayilmali."""
        gray = _narrow_range_gray(100, 150)
        out = histogram_equalization(gray)
        original_range = int(gray.max()) - int(gray.min())
        out_range = int(out.max()) - int(out.min())
        assert out_range > original_range

    def test_uniform_image_preserves_value(self):
        """Tekduze goruntude cv2.equalizeHist degeri DEGISTIRMEZ (255'e donmez).
        
        OpenCV'nin gercek davranisi: tekduze goruntude CDF tek noktada
        toplanir ve pikseller ayni deger uzerinde kalir.
        """
        uniform = np.full((50, 50), 128, dtype=np.uint8)
        out = histogram_equalization(uniform)
        # Deger 128 olmali (255 degil)
        assert out[0, 0] == 128
        assert np.all(out == out[0, 0])

    def test_3d_input_raises(self):
        bgr = np.zeros((50, 50, 3), dtype=np.uint8)
        with pytest.raises(ValueError):
            histogram_equalization(bgr)

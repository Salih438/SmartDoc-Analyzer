"""
tests/test_image_info.py
------------------------
load_image(), save_image() ve get_image_info() fonksiyonlari icin pytest testleri.
"""

from __future__ import annotations

import os
import tempfile
from pathlib import Path

import cv2
import numpy as np
import pytest

from src.analysis.image_info import get_image_info, load_image, save_image


# ---------------------------------------------------------------------------
# Yardimci: gecici goruntu dosyasi
# ---------------------------------------------------------------------------

def _write_temp_image(img: np.ndarray, suffix: str = ".png") -> Path:
    """Gecici bir PNG/JPG dosyasina goruntu yazar ve yolunu dondurur."""
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as f:
        path = Path(f.name)
    cv2.imwrite(str(path), img)
    return path


# ---------------------------------------------------------------------------
# load_image testleri
# ---------------------------------------------------------------------------

class TestLoadImage:
    def test_load_valid_png(self):
        """Gecerli PNG dosyasi basariyla yuklenmeli."""
        img = np.random.randint(0, 256, (50, 60, 3), dtype=np.uint8)
        path = _write_temp_image(img, ".png")
        try:
            loaded = load_image(path)
            assert isinstance(loaded, np.ndarray)
            assert loaded.shape[0] == 50
            assert loaded.shape[1] == 60
        finally:
            path.unlink(missing_ok=True)

    def test_load_valid_jpg(self):
        """Gecerli JPG dosyasi basariyla yuklenmeli."""
        img = np.random.randint(0, 256, (40, 80, 3), dtype=np.uint8)
        path = _write_temp_image(img, ".jpg")
        try:
            loaded = load_image(path)
            assert isinstance(loaded, np.ndarray)
        finally:
            path.unlink(missing_ok=True)

    def test_load_valid_jpeg(self):
        """Gecerli .jpeg uzantili dosya yuklenmeli."""
        img = np.ones((30, 30, 3), dtype=np.uint8) * 128
        path = _write_temp_image(img, ".jpeg")
        try:
            loaded = load_image(path)
            assert loaded is not None
        finally:
            path.unlink(missing_ok=True)

    def test_file_not_found_raises(self):
        """Var olmayan dosya yolu FileNotFoundError vermeli."""
        with pytest.raises(FileNotFoundError, match=r"Görüntü dosyası bulunamadı"):
            load_image("/nonexistent/path/image.png")

    def test_unsupported_extension_raises(self):
        """Desteklenmeyen uzanti ValueError vermeli."""
        with tempfile.NamedTemporaryFile(suffix=".bmp", delete=False) as f:
            bmp_path = f.name
        try:
            with pytest.raises(ValueError, match=r"Desteklenmeyen dosya uzantısı"):
                load_image(bmp_path)
        finally:
            os.unlink(bmp_path)

    def test_string_path_accepted(self):
        """str turunde yol da kabul edilmeli."""
        img = np.zeros((10, 10, 3), dtype=np.uint8)
        path = _write_temp_image(img, ".png")
        try:
            loaded = load_image(str(path))
            assert loaded is not None
        finally:
            path.unlink(missing_ok=True)

    def test_unicode_directory_path(self, tmp_path):
        """Turkce/Unicode karakterli dizin yolunda yukleme basariyla calisimali.
        
        Bu test Windows'ta cv2.imread yerine np.fromfile+imdecode
        kullanimindan faydalanir.
        """
        unicode_dir = tmp_path / "görüntü_ışleme_şü"
        unicode_dir.mkdir()
        img = np.random.randint(0, 256, (40, 40, 3), dtype=np.uint8)
        img_path = unicode_dir / "test_goruntu.png"
        save_image(img_path, img)
        
        loaded = load_image(img_path)
        assert isinstance(loaded, np.ndarray)
        assert loaded.shape[:2] == (40, 40)


# ---------------------------------------------------------------------------
# save_image testleri
# ---------------------------------------------------------------------------

class TestSaveImage:
    def test_save_and_reload_png(self, tmp_path):
        """Kaydedilen PNG dosyasi tekrar yuklenebilmeli."""
        img = np.random.randint(0, 256, (30, 30, 3), dtype=np.uint8)
        path = tmp_path / "output.png"
        save_image(path, img)
        assert path.exists()
        loaded = load_image(path)
        assert loaded.shape == img.shape

    def test_save_unicode_path(self, tmp_path):
        """Unicode karakterli hedef yoluna kaydetme basarili olmali."""
        unicode_dir = tmp_path / "kayıt_görüntü_şü"
        img = np.ones((20, 20, 3), dtype=np.uint8) * 200
        out_path = unicode_dir / "cikti.png"
        save_image(out_path, img)
        assert out_path.exists()

    def test_save_unsupported_extension_raises(self, tmp_path):
        """Desteklenmeyen uzantida ValueError vermeli."""
        img = np.zeros((10, 10), dtype=np.uint8)
        with pytest.raises(ValueError, match="Desteklenmeyen"):
            save_image(tmp_path / "out.bmp", img)

    def test_save_creates_parent_dirs(self, tmp_path):
        """Hedef ust dizinler yoksa otomatik olusturulmali."""
        deep_path = tmp_path / "a" / "b" / "c" / "img.png"
        img = np.zeros((10, 10, 3), dtype=np.uint8)
        save_image(deep_path, img)
        assert deep_path.exists()


# ---------------------------------------------------------------------------
# get_image_info testleri
# ---------------------------------------------------------------------------

class TestGetImageInfo:
    def test_grayscale_image_channels(self):
        """2-D gri goruntu icin channels=1 dondurulmeli."""
        gray = np.zeros((100, 200), dtype=np.uint8)
        info = get_image_info(gray)
        assert info["channels"] == 1
        assert info["width"] == 200
        assert info["height"] == 100

    def test_bgr_image_channels(self):
        """3-kanalli BGR goruntu icin channels=3 dondurulmeli."""
        img = np.zeros((80, 120, 3), dtype=np.uint8)
        info = get_image_info(img)
        assert info["channels"] == 3

    def test_bgra_image_channels(self):
        """4-kanalli BGRA goruntu icin channels=4 dondurulmeli."""
        img = np.zeros((50, 50, 4), dtype=np.uint8)
        info = get_image_info(img)
        assert info["channels"] == 4

    def test_dtype_reported_correctly(self):
        """dtype alani goruntunun gercek dtype'ini string olarak icermeli."""
        img = np.zeros((10, 10), dtype=np.uint8)
        assert get_image_info(img)["dtype"] == "uint8"

    def test_black_image_stats(self):
        """Tamamen siyah goruntude min=max=mean=0, kontrast=0 olmali."""
        black = np.zeros((50, 50), dtype=np.uint8)
        info = get_image_info(black)
        assert info["min_pixel"] == 0.0
        assert info["max_pixel"] == 0.0
        assert info["mean_brightness"] == pytest.approx(0.0)
        assert info["std_contrast"] == pytest.approx(0.0)
        assert info["range_contrast"] == pytest.approx(0.0)

    def test_white_image_stats(self):
        """Tamamen beyaz goruntude min=max=mean=255, range_contrast=0 olmali."""
        white = np.full((50, 50), 255, dtype=np.uint8)
        info = get_image_info(white)
        assert info["min_pixel"] == 255.0
        assert info["max_pixel"] == 255.0
        assert info["mean_brightness"] == pytest.approx(255.0)
        assert info["range_contrast"] == pytest.approx(0.0)

    def test_gradient_image_range(self):
        """0->255 gradyaninda range_contrast 255'e yakin olmali."""
        grad = np.tile(np.arange(256, dtype=np.uint8), (10, 1))
        info = get_image_info(grad)
        assert info["range_contrast"] == pytest.approx(255.0)

    def test_invalid_input_raises(self):
        """np.ndarray olmayan giris TypeError vermeli."""
        with pytest.raises(TypeError):
            get_image_info([[1, 2], [3, 4]])

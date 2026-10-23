"""Pillow tabanlı çok sayfalı PDF dışa aktarma işlevleri."""

from __future__ import annotations

from io import BytesIO
import math
from pathlib import Path
from typing import Literal

import numpy as np
from PIL import Image, features


PageSize = Literal["a4", "fit"]
A4Orientation = Literal["auto", "portrait", "landscape"]
PDF_DPI = 300
A4_WIDTH_MM = 210.0
A4_HEIGHT_MM = 297.0


def _a4_dimensions_px(dpi: int) -> tuple[int, int]:
    """A4 sayfa genişlik ve yüksekliğini belirtilen DPI'da hesaplar."""
    return (
        round(A4_WIDTH_MM * dpi / 25.4),
        round(A4_HEIGHT_MM * dpi / 25.4),
    )


def _to_pdf_image(
    image: np.ndarray, index: int, lossless_binary: bool = False
) -> Image.Image:
    """Gri ve ikili kipleri korur; BGR'yi RGB'ye dönüştürür."""
    if not isinstance(image, np.ndarray):
        raise TypeError(f"PDF sayfa {index}: NumPy görüntüsü bekleniyor")
    if image.dtype != np.uint8:
        raise TypeError(f"PDF sayfa {index}: uint8 görüntü bekleniyor")
    if image.ndim == 2:
        if min(image.shape) < 1:
            raise ValueError(f"PDF sayfa {index}: boş görüntü işlenemez")
        if set(np.unique(image).tolist()) <= {0, 255}:
            page = Image.fromarray(image)
            return page.convert("1") if lossless_binary else page
        page = Image.fromarray(image)
        return page
    if image.ndim != 3 or image.shape[2] != 3 or min(image.shape[:2]) < 1:
        raise ValueError(f"PDF sayfa {index}: gri veya üç kanallı BGR görüntü bekleniyor")
    rgb = image[:, :, ::-1].copy()
    return Image.fromarray(rgb)


def _page_canvas(
    image: Image.Image,
    page_size: PageSize,
    dpi: int,
    margin_mm: float,
    a4_orientation: A4Orientation = "auto",
) -> Image.Image:
    """Bir görüntüyü A4 veya görüntü ölçülü beyaz sayfa üzerine ortalar."""
    margin_px = round(margin_mm * dpi / 25.4)
    if page_size == "fit":
        white = 1 if image.mode == "1" else 255 if image.mode == "L" else "white"
        canvas = Image.new(image.mode, (image.width + 2 * margin_px, image.height + 2 * margin_px), white)
        canvas.paste(image, (margin_px, margin_px))
        return canvas

    if a4_orientation not in ("auto", "portrait", "landscape"):
        raise ValueError("a4_orientation 'auto', 'portrait' veya 'landscape' olmalıdır.")
    page_width, page_height = _a4_dimensions_px(dpi)
    landscape = a4_orientation == "landscape" or (
        a4_orientation == "auto" and image.width / image.height > 1
    )
    if landscape:
        page_width, page_height = page_height, page_width
    available_width = page_width - 2 * margin_px
    available_height = page_height - 2 * margin_px
    if available_width < 1 or available_height < 1:
        raise ValueError("PDF kenar boşluğu A4 sayfadan büyük olamaz")
    scale = min(available_width / image.width, available_height / image.height)
    resized_size = (
        max(1, round(image.width * scale)),
        max(1, round(image.height * scale)),
    )
    if image.mode == "1":
        # Antialias 1-bit artwork in grayscale, threshold it back to 1-bit,
        # and retain Pillow's CCITT-capable page mode for PDF embedding.
        gray = image.convert("L").resize(resized_size, Image.Resampling.LANCZOS)
        resized = gray.point(lambda value: 255 if value >= 128 else 0).convert("1")
    else:
        resized = image.resize(resized_size, Image.Resampling.LANCZOS)
    white = 1 if image.mode == "1" else 255 if image.mode == "L" else "white"
    canvas = Image.new(image.mode, (page_width, page_height), white)
    offset = ((page_width - resized.width) // 2, (page_height - resized.height) // 2)
    canvas.paste(resized, offset)
    return canvas


def images_to_pdf(
    images: list[np.ndarray],
    output: Path | BytesIO,
    page_size: PageSize = "a4",
    dpi: int = PDF_DPI,
    margin_mm: float = 0,
    a4_orientation: A4Orientation = "auto",
) -> None:
    """Gri/BGR görüntüleri bir veya çok sayfalı PDF olarak kaydeder.

    Problem
    -------
    Tarama görüntülerini yazdırılabilir, sabit ölçülü A4 veya kayıpsız piksel
    ölçülü sayfalara dönüştürmek.

    Algoritma
    ---------
    Gri ve renkli sayfalar JPEG kalite 95 ile gömülür. Yalnızca ikili sayfalardan
    oluşan PDF'lerde libtiff varsa 1-bit CCITT Group 4 kayıpsız kodlama kullanılır;
    karma PDF'lerde kalite 95 uygulanır. BGR sayfalar RGB'ye çevrilir. İkili A4
    küçültme gri tonlamada LANCZOS, ardından 128 eşiğiyle yeniden 1-bit yapılır;
    böylece çizgi yumuşatması ölçülürken CCITT gömme biçimi korunur. A4'te
    oran korunarak sığdırılıp ortalanır,
    ``fit`` modunda görüntü boyutunda sayfa oluşturulur. Pillow PDF kodlayıcısı
    tüm sayfaları ``save_all`` ve belirtilen çözünürlükle yazar.

    Parametreler
    ------------
    images : list[np.ndarray]
        uint8 gri veya BGR sayfa görüntüleri.
    output : Path | BytesIO
        PDF dosya yolu veya bellekte yazılabilir bayt akışı.
    page_size : Literal["a4", "fit"]
        A4 sayfası veya görüntü ölçüsünde sayfa.
    dpi : int
        PDF fiziksel çözünürlüğü; pozitif tam sayı.
    margin_mm : float
        Sayfa kenar boşluğu milimetre cinsinden.
    a4_orientation : A4Orientation
        A4 sayfası yönü; auto yatay görüntülerde yatay, diğerlerinde dikey kullanır.

    Başarısız Olabilir
    ------------------
    Boş sayfa listesi, bozuk görüntü boyutu/kanal/dtype, geçersiz sayfa boyutu,
    DPI veya sayfaya sığmayan kenar boşluğu için anlaşılır hata verir.
    """
    if not images:
        raise ValueError("PDF oluşturmak için en az bir görüntü gerekli.")
    if page_size not in ("a4", "fit"):
        raise ValueError("page_size 'a4' veya 'fit' olmalıdır.")
    if isinstance(dpi, bool) or not isinstance(dpi, int) or dpi <= 0:
        raise ValueError("dpi pozitif bir tam sayı olmalıdır.")
    if not math.isfinite(margin_mm) or margin_mm < 0:
        raise ValueError("margin_mm sıfır veya pozitif sonlu bir sayı olmalıdır.")
    if a4_orientation not in ("auto", "portrait", "landscape"):
        raise ValueError("a4_orientation 'auto', 'portrait' veya 'landscape' olmalıdır.")

    binary_pages = all(
        image.ndim == 2 and set(np.unique(image).tolist()) <= {0, 255}
        for image in images
    )
    lossless_binary = binary_pages and features.check("libtiff")
    pages = [
        _page_canvas(
            _to_pdf_image(image, index, lossless_binary), page_size, dpi, margin_mm,
            a4_orientation,
        )
        for index, image in enumerate(images, start=1)
    ]
    if isinstance(output, BytesIO):
        output.seek(0)
        output.truncate(0)
        target = output
    elif isinstance(output, Path):
        output.parent.mkdir(parents=True, exist_ok=True)
        target = output
    else:
        raise TypeError("output Path veya BytesIO olmalıdır.")
    save_options = {
        "format": "PDF",
        "save_all": True,
        "append_images": pages[1:],
        "resolution": float(dpi),
    }
    if not lossless_binary:
        save_options["quality"] = 95
    pages[0].save(target, **save_options)

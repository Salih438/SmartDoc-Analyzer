"""Tarama görünümü üretimi; OCR için kullanılan ters ikili çıktıyı değiştirmez."""

from __future__ import annotations

from typing import Literal

import cv2
import numpy as np
from skimage.filters import threshold_sauvola

from src.pipeline import scale_odd_parameter
from src.preprocessing.enhancement import document_stretch
from src.preprocessing.illumination import flatten_background
from src.preprocessing.grayscale import to_grayscale


ScanMode = Literal["color", "gray", "bw", "original"]
SCAN_MODES: tuple[ScanMode, ...] = ("color", "gray", "bw", "original")


def make_scan(
    rectified_bgr: np.ndarray,
    mode: ScanMode,
    *,
    ksize: int | None = None,
    cleanup: bool = False,
    inset_fraction: float = 0.01,
    sauvola_window_size: int = 99,
    background_flattened: bool = False,
) -> np.ndarray:
    """Perspektifi düzeltilmiş görüntüden kullanıcıya dönük tarama çıktısı üretir.

    Problem
    -------
    Pipeline'ın ters ikili görüntüsü OCR'da yazıyı beyaz ön plan yapar. Tarama
    çıktısı ise yaygın belge biçiminde beyaz kâğıt ve siyah yazı sağlamalıdır.

    Algoritma
    ---------
    ``original`` kopya döndürür. Diğer modlar önce arka plan düzleştirir. Renkli
    mod HSV parlaklık kanalını normalize edip beyaz nokta gerdirmesi uygular;
    gri mod düzleştirilmiş griyi döndürür; bw modu Sauvola yerel eşiğiyle koyu
    mürekkebi 0, açık zemini 255 yapar.

    Parametreler
    ------------
    rectified_bgr : np.ndarray
        uint8 gri veya BGR, perspektifi düzeltilmiş görüntü.
    mode : ScanMode
        color, gray, bw veya original çıktı modu.
    ksize : int | None
        Arka plan kestirim kernel çapı; None ise görüntü çözünürlüğüne göre seçilir.
    cleanup : bool
        True ise bw çıktıda küçük beyaz gürültü bileşenlerine opening uygular.
    inset_fraction : float
        Her kenardan kırpılacak görüntü oranı; varsayılan 0.01.
    sauvola_window_size : int
        Referans çözünürlükte ölçeklenecek tek sayılı pencere; varsayılan 99.
    background_flattened : bool
        True ise girdi arka plan düzleştirmesinden geçmiştir ve işlem yinelenmez.

    Başarısız Olabilir
    ------------------
    Closing kernel'i yazı boyutundan küçükse yazıyı arka plan kestirimine dahil
    edebilir. Sauvola sabitleri çok düşük kontrastta veya çok küçük yazıda uygun
    eşik üretmeyebilir. Geçersiz mod, dtype veya kanal düzeni açık hata verir.
    """
    if not isinstance(rectified_bgr, np.ndarray):
        raise TypeError("make_scan: NumPy görüntüsü bekleniyor")
    if rectified_bgr.dtype != np.uint8:
        raise TypeError("make_scan: uint8 görüntü bekleniyor")
    if mode not in SCAN_MODES:
        raise ValueError(f"Desteklenmeyen tarama modu: {mode}")
    if rectified_bgr.ndim not in (2, 3) or (
        rectified_bgr.ndim == 3 and rectified_bgr.shape[2] != 3
    ):
        raise ValueError("make_scan: gri veya üç kanallı BGR görüntü bekleniyor")
    if min(rectified_bgr.shape[:2]) < 3:
        raise ValueError("make_scan: görüntü en az 3x3 piksel olmalıdır")
    if not isinstance(cleanup, bool):
        raise TypeError("make_scan: cleanup bool olmalıdır")
    if not isinstance(inset_fraction, (int, float)) or isinstance(inset_fraction, bool):
        raise TypeError("make_scan: inset_fraction sayısal olmalıdır")
    if not 0 <= inset_fraction < 0.5:
        raise ValueError("make_scan: inset_fraction 0 ile 0.5 arasında olmalıdır")
    if isinstance(sauvola_window_size, bool) or not isinstance(sauvola_window_size, int):
        raise TypeError("make_scan: Sauvola penceresi tam sayı olmalıdır")
    if sauvola_window_size < 3 or sauvola_window_size % 2 == 0:
        raise ValueError("make_scan: Sauvola penceresi en az 3 olan tek sayı olmalıdır")
    if not isinstance(background_flattened, bool):
        raise TypeError("make_scan: background_flattened bool olmalıdır")

    height, width = rectified_bgr.shape[:2]
    inset_y = round(height * inset_fraction)
    inset_x = round(width * inset_fraction)
    cropped = rectified_bgr[
        inset_y : height - inset_y if inset_y else height,
        inset_x : width - inset_x if inset_x else width,
    ]
    if min(cropped.shape[:2]) < 3:
        raise ValueError("make_scan: kenar kırpmasından sonra görüntü en az 3x3 olmalıdır")

    if mode == "original":
        return cropped.copy()

    gray = to_grayscale(cropped)
    if mode == "gray":
        return gray.copy() if background_flattened else flatten_background(gray, ksize=ksize)
    if mode == "color":
        if cropped.ndim == 2:
            bgr = cv2.cvtColor(cropped, cv2.COLOR_GRAY2BGR)
        else:
            bgr = cropped.copy()
        hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
        value = hsv[:, :, 2] if background_flattened else flatten_background(hsv[:, :, 2], ksize=ksize)
        hsv[:, :, 2] = document_stretch(value)
        return cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

    flattened = gray.copy() if background_flattened else flatten_background(gray, ksize=ksize)
    window_size = scale_odd_parameter(sauvola_window_size, flattened.shape, 3, 99)
    threshold = threshold_sauvola(flattened, window_size=window_size, k=0.2)
    binary = np.where(flattened > threshold, 255, 0).astype(np.uint8)
    if cleanup:
        ink_mask = cv2.bitwise_not(binary)
        kernel_size = scale_odd_parameter(3, binary.shape, 3, 9)
        kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (kernel_size, kernel_size)
        )
        ink_mask = cv2.morphologyEx(ink_mask, cv2.MORPH_OPEN, kernel)
        binary = cv2.bitwise_not(ink_mask)
    return binary

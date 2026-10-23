"""
src/segmentation/edges.py
-------------------------
Belge görüntüleri için kenar tespit algoritmalarını barındıran modül.

Canny kenar tespiti ve medyan tabanlı otomatik eşik belirleme (auto_canny)
fonksiyonlarını sunar. Belge sınırlarının, tabloların ve metin bloklarının
tespitinde temel segmentasyon adımı olarak kullanılır.
"""

from __future__ import annotations

import cv2
import numpy as np

from src.utils.validation import validate_gray, validate_uint8

_validate_gray = validate_gray


# ---------------------------------------------------------------------------
# 1. Canny Kenar Tespiti
# ---------------------------------------------------------------------------

def canny_edges(
    gray: np.ndarray,
    low: float,
    high: float,
) -> np.ndarray:
    """Canny algoritması ile kenarları tespit eder.

    Problem
    -------
    Görüntüdeki önemli yapısal sınırları (metin satırları, belge kenarları,
    çizgiler) tespit etmek; gürültüden kaynaklanan zayıf geçişleri filtrelemek.

    Parametreler
    ------------
    gray : np.ndarray
        2 boyutlu gri tonlamalı görüntü dizisi (uint8).
    low : float
        Histerezis için alt eşik değeri.
    high : float
        Histerezis için üst eşik değeri.

    Döndürür
    --------
    np.ndarray
        uint8 formatında ikili (binary) görüntü. Kenar pikselleri 255, arka plan 0'dır.

    Başarısız Olabilir
    ------------------
    - ValueError : gray 2 boyutlu değilse veya low > high ise.
    """
    validate_gray(gray, "canny_edges")
    validate_uint8(gray, "canny_edges")
    if low > high:
        raise ValueError(
            f"canny_edges: low eşiği ({low}) high eşiğinden ({high}) büyük olamaz."
        )

    return cv2.Canny(gray, threshold1=float(low), threshold2=float(high))


# ---------------------------------------------------------------------------
# 2. Otomatik Canny (Auto Canny)
# ---------------------------------------------------------------------------

def auto_canny(
    gray: np.ndarray,
    sigma: float = 0.33,
) -> np.ndarray:
    """Medyan parlaklığa dayalı otomatik eşik hesaplamalı Canny kenar tespiti.

    Problem
    -------
    Manuel Canny eşiklerinin her belgenin aydınlatma ve kontrast seviyesine
    göre elle ayarlanması ihtiyacını ortadan kaldırmak.

    Parametreler
    ------------
    gray : np.ndarray
        2 boyutlu gri tonlamalı görüntü dizisi (uint8).
    sigma : float, varsayılan 0.33
        Medyan etrafındaki eşik penceresi genişliği katsayısı.

    Döndürür
    --------
    np.ndarray
        uint8 formatında ikili görüntü.
    """
    validate_gray(gray, "auto_canny")
    validate_uint8(gray, "auto_canny")
    if sigma < 0:
        raise ValueError(f"auto_canny: sigma negatif olamaz, alınan: {sigma}")

    v = float(np.median(gray))
    lower = int(max(0, (1.0 - sigma) * v))
    upper = int(min(255, (1.0 + sigma) * v))

    return canny_edges(gray, lower, upper)

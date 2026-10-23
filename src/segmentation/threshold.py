"""
src/segmentation/threshold.py
-----------------------------
Belge görüntüleri için eşikleme (binarizasyon) algoritmalarını barındıran modül.

Global eşikleme, Otsu otomatik eşikleme ve uyarlanabilir (adaptive) eşikleme
yöntemlerini sunar.

Önemli Belge Konvansiyonu
-------------------------
Belge görsellerinde yazı koyu (düşük piksel değeri), kâğıt zemin ise açık
(yüksek piksel değeri) renktedir. Morfolojik işlemler ve nesne analizi
için "yazı = beyaz ön plan (255)" ve "zemin = siyah arka plan (0)" konvansiyonu
kullanılır. `invert=True` seçeneği, koyu metni beyaz ön plana (255), açık kâğıdı
ise siyah arka plana (0) dönüştürmek amacıyla kullanılır.
"""

from __future__ import annotations

import cv2
import numpy as np

from src.utils.validation import validate_gray, validate_uint8

_validate_gray = validate_gray


# ---------------------------------------------------------------------------
# 1. Sabit Global Eşikleme
# ---------------------------------------------------------------------------

def global_threshold(
    gray: np.ndarray,
    thresh: float = 127,
    invert: bool = False,
) -> np.ndarray:
    """Sabit bir eşik değeri ile ikili (binary) görüntü üretir.

    Konvansiyon
    -----------
    - invert=False: piksel > thresh ise 255 (açık zemin), aksi halde 0 (koyu yazı).
    - invert=True : piksel <= thresh ise 255 (beyaz yazı ön plan), zemin 0 olur.

    Parametreler
    ------------
    gray : np.ndarray
        2 boyutlu gri tonlamalı görüntü dizisi (uint8).
    thresh : float, varsayılan 127
        Eşik değeri [0, 255].
    invert : bool, varsayılan False
        True ise çıktı tersine çevrilir (THRESH_BINARY_INV).
    """
    validate_gray(gray, "global_threshold")
    validate_uint8(gray, "global_threshold")
    thresh_type = cv2.THRESH_BINARY_INV if invert else cv2.THRESH_BINARY
    _, binary = cv2.threshold(gray, float(thresh), 255, thresh_type)
    return binary


# ---------------------------------------------------------------------------
# 2. Otsu Otomatik Eşikleme
# ---------------------------------------------------------------------------

def otsu_threshold(
    gray: np.ndarray,
    invert: bool = False,
) -> tuple[np.ndarray, float]:
    """Otsu algoritması ile sınıflar arası varyansı maksimize eden eşiği bulur.

    Parametreler
    ------------
    gray : np.ndarray
        2 boyutlu gri tonlamalı görüntü dizisi (uint8).
    invert : bool, varsayılan False
        True ise yazı ön planı 255 (THRESH_BINARY_INV + THRESH_OTSU).

    Döndürür
    --------
    tuple[np.ndarray, float]
        (binary_görüntü, bulunan_eşik_değeri)
    """
    validate_gray(gray, "otsu_threshold")
    validate_uint8(gray, "otsu_threshold")
    thresh_type = (cv2.THRESH_BINARY_INV if invert else cv2.THRESH_BINARY) | cv2.THRESH_OTSU
    thresh_val, binary = cv2.threshold(gray, 0, 255, thresh_type)
    return binary, float(thresh_val)


# ---------------------------------------------------------------------------
# 3. Uyarlanabilir (Adaptive) Eşikleme
# ---------------------------------------------------------------------------

def adaptive_threshold(
    gray: np.ndarray,
    method: str = "gaussian",
    block_size: int = 11,
    c: float = 2.0,
    invert: bool = False,
) -> np.ndarray:
    """Yerel komşuluklara göre dinamik eşik hesaplayan uyarlanabilir eşikleme.

    Problem
    -------
    Dengesiz aydınlatma, gölgeler veya parlamalar bulunan belge görüntülerinde
    her piksel için komşuluğuna göre yerel eşik belirler.

    Parametreler
    ------------
    gray : np.ndarray
        2 boyutlu gri tonlamalı görüntü dizisi (uint8).
    method : str, varsayılan 'gaussian'
        Yerel ağırlıklandırma yöntemi: 'gaussian' veya 'mean'.
    block_size : int, varsayılan 11
        Yerel pencere boyutu. Tek sayı ve > 1 olmalıdır.
    c : float, varsayılan 2.0
        Hesaplanan ortalamadan çıkarılacak sabit değer.
    invert : bool, varsayılan False
        True ise yazı ön planı 255 yapılır.
    """
    validate_gray(gray, "adaptive_threshold")
    validate_uint8(gray, "adaptive_threshold")

    if block_size <= 1 or block_size % 2 == 0:
        raise ValueError(
            f"adaptive_threshold: block_size tek sayı ve > 1 olmalı, alınan: {block_size}"
        )

    method_lower = method.lower().strip()
    if method_lower == "gaussian":
        adaptive_method = cv2.ADAPTIVE_THRESH_GAUSSIAN_C
    elif method_lower == "mean":
        adaptive_method = cv2.ADAPTIVE_THRESH_MEAN_C
    else:
        raise ValueError(
            f"adaptive_threshold: method 'gaussian' veya 'mean' olmalı, alınan: '{method}'"
        )

    thresh_type = cv2.THRESH_BINARY_INV if invert else cv2.THRESH_BINARY

    return cv2.adaptiveThreshold(
        gray,
        maxValue=255,
        adaptiveMethod=adaptive_method,
        thresholdType=thresh_type,
        blockSize=block_size,
        C=float(c),
    )

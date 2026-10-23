"""
src/preprocessing/sharpening.py
--------------------------------
Görüntü keskinleştirme algoritmalarını barındıran modül.

Unsharp masking ve Laplacian tabanlı keskinleştirme fonksiyonlarını sunar.
Bu yöntemler metin belgelerindeki silik karakter kenarlarını güçlendirmek
amacıyla kullanılabilir.
"""

from __future__ import annotations

import cv2
import numpy as np

from src.utils.validation import validate_gray, validate_ksize, validate_uint8

_validate_gray = validate_gray
_validate_ksize = validate_ksize


# ---------------------------------------------------------------------------
# 1. Unsharp Masking
# ---------------------------------------------------------------------------

def unsharp_mask(
    gray: np.ndarray,
    ksize: int = 5,
    sigma: float = 1.0,
    amount: float = 1.0,
) -> np.ndarray:
    """Unsharp masking yöntemi ile görüntüyü keskinleştirir.

    Problem
    -------
    Düşük çözünürlüklü taramalar veya odak dışı çekimler nedeniyle belgedeki
    harf kenarlarında oluşan bulanıklığı gidermek ve kenar kontrastını artırmak.

    Algoritma
    ---------
    1. Görüntüye Gaussian blur uygulanarak alçak frekans bileşeni (blurred) elde edilir.
    2. Detay maskesi oluşturulur: mask = orig - blurred (yüksek frekans bileşenleri).
    3. Ölçeklenmiş detaylar orijinal görüntüye eklenir:
       out = orig + amount * (orig - blurred)
    4. Piksel değerleri [0, 255] aralığına kırpılır (clip) ve uint8'e dönüştürülür.

    Parametreler
    ------------
    gray : np.ndarray
        2 boyutlu gri tonlamalı görüntü dizisi (uint8).
    ksize : int, varsayılan 5
        Gaussian filtresi çekirdek boyutu (tek sayı ve >= 1 olmalıdır).
    sigma : float, varsayılan 1.0
        Gaussian standart sapması.
    amount : float, varsayılan 1.0
        Keskinleştirme gücü katsayısı. amount=0 iken çıktı girdinin kopyasıdır.

    Döndürür
    --------
    np.ndarray
        uint8 formatında keskinleştirilmiş görüntü.

    Başarısız Olabilir
    ------------------
    - ValueError : gray 2 boyutlu değilse veya ksize çift sayıysa.
    - Çok yüksek amount (> 2.5) gürültüyü büyütür ve harf etrafında hale (halo) oluşturur.
    """
    validate_gray(gray, "unsharp_mask")
    validate_uint8(gray, "unsharp_mask")
    validate_ksize(ksize, "unsharp_mask")

    if amount == 0.0:
        return gray.copy()

    orig_float = gray.astype(np.float64)
    blurred = cv2.GaussianBlur(gray, (ksize, ksize), sigmaX=sigma, sigmaY=sigma).astype(np.float64)
    detail = orig_float - blurred
    sharpened = orig_float + amount * detail

    return np.clip(sharpened, 0, 255).astype(np.uint8)


# ---------------------------------------------------------------------------
# 2. Laplacian Sharpening
# ---------------------------------------------------------------------------

def laplacian_sharpen(
    gray: np.ndarray,
    weight: float = 1.0,
) -> np.ndarray:
    """Laplacian ikinci derece türev filtresi ile görüntüyü keskinleştirir.

    Problem
    -------
    İkinci derece uzamsal türev (Laplacian), pikseller arasındaki ani ton
    değişimlerini (kenarlar ve köşe noktaları) belirler. Bu türev bileşeni
    orijinal görüntüye eklenerek keskinleştirme sağlanır.

    Parametreler
    ------------
    gray : np.ndarray
        2 boyutlu gri tonlamalı görüntü dizisi (uint8).
    weight : float, varsayılan 1.0
        Laplacian katsayısının ağırlığı. weight=0 iken çıktı girdinin kopyasıdır.

    Döndürür
    --------
    np.ndarray
        uint8 formatında keskinleştirilmiş görüntü.

    Başarısız Olabilir
    ------------------
    - ValueError : gray 2 boyutlu değilse.
    """
    validate_gray(gray, "laplacian_sharpen")
    validate_uint8(gray, "laplacian_sharpen")

    if weight == 0.0:
        return gray.copy()

    orig_float = gray.astype(np.float64)
    lap = cv2.Laplacian(gray, cv2.CV_64F, ksize=1)
    sharpened = orig_float - weight * lap

    return np.clip(sharpened, 0, 255).astype(np.uint8)

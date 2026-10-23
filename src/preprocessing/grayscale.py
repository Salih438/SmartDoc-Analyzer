"""
src/preprocessing/grayscale.py
-------------------------------
Renk uzayından gri tonlamaya dönüşüm modülü.
"""

from __future__ import annotations

import cv2
import numpy as np


def to_grayscale(img: np.ndarray) -> np.ndarray:
    """BGR veya BGRA görüntüyü gri tonlamalı tek-kanallı görüntüye dönüştürür.

    Problem
    -------
    Renk bilgisi içermeyen belge görüntüleri üzerinde çalışan algoritmalar
    (eşikleme, kenar tespiti vb.) tek kanallı giriş gerektirir. Bu fonksiyon,
    zaten gri olan görüntüleri kopyalamadan geçirir; çok-kanallı olanları ise
    ITU-R BT.601 ağırlıklı dönüşümle gri tonlamaya çevirir.

    ITU-R BT.601 formülü:
        Y = 0.114·B + 0.587·G + 0.299·R

    Parametreler
    ------------
    img : np.ndarray
        Giriş görüntüsü.  2-D (gri), 3-D BGR veya 3-D BGRA dizileri
        kabul edilir.  dtype'ın uint8 olması önerilir; float görüntüler
        OpenCV'nin dönüşüm fonksiyonuna olduğu gibi iletilir.

    Döndürür
    --------
    np.ndarray
        2-D gri tonlamalı görüntü dizisi.  Giriş zaten 2-D ise **aynı
        dizi** döndürülür (kopya oluşturulmaz).

    Başarısız Olabilir
    ------------------
    - ValueError : img 2-D veya 3-D değilse (ör. 1-D vektör geçirilirse).
    - 4 kanaldan fazla kanal içeren görüntülerde ValueError yükseltilir;
      bu durum standart görüntü dosyalarında oluşmaz.
    """
    if img.ndim == 2:
        return img  # Zaten gri tonlamalı

    if img.ndim == 3:
        channels = img.shape[2]
        if channels == 1:
            return img[:, :, 0]
        if channels == 3:
            return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        if channels == 4:
            return cv2.cvtColor(img, cv2.COLOR_BGRA2GRAY)
        raise ValueError(
            f"Desteklenmeyen kanal sayısı: {channels}. "
            "Beklenen: 1, 3 veya 4."
        )

    raise ValueError(
        f"img 2-D veya 3-D olmalıdır, alınan boyut sayısı: {img.ndim}"
    )

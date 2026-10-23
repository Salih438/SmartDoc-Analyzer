"""
src/preprocessing/filtering.py
--------------------------------
Görüntü yumuşatma ve gürültü azaltma filtreleri modülü.

Üç farklı filtre sunulur: Gaussian, Median ve Bilateral.
Her biri farklı gürültü türlerine karşı farklı davranır;
ölçüm sonuçları deneylerle belirlenir.
"""

from __future__ import annotations

import cv2
import numpy as np

from src.utils.validation import validate_gray, validate_ksize, validate_uint8

_validate_gray = validate_gray
_validate_ksize = validate_ksize


# ---------------------------------------------------------------------------
# A1. Gaussian Blur
# ---------------------------------------------------------------------------

def gaussian_blur(
    gray: np.ndarray,
    ksize: int = 5,
    sigma: float = 0,
) -> np.ndarray:
    """Gaussian yumuşatma filtresi uygular.

    Problem
    -------
    Görüntüde bulunan Gaussian (istatistiksel) gürültüyü azaltmak.
    Gaussian filtre, her pikseli komşuluğunun ağırlıklı ortalamasıyla
    değiştirir; ağırlıklar merkeze yakın piksellerde daha yüksektir.

    Parametreler
    ------------
    gray : np.ndarray
        2 boyutlu gri tonlamalı görüntü dizisi (uint8).
    ksize : int, varsayılan 5
        Filtre pencere boyutu. Tek sayı ve >= 1 olmalıdır (1, 3, 5, 7, ...).
        Çift sayı verilirse ValueError yükseltilir.
    sigma : float, varsayılan 0
        Gaussian standart sapması. 0 verilirse OpenCV ksize'dan otomatik hesaplar.

    Döndürür
    --------
    np.ndarray
        uint8, girdiyle aynı şekil.

    Başarısız Olabilir
    ------------------
    - ValueError : gray 2 boyutlu değilse veya ksize çift sayıysa.
    """
    validate_gray(gray, "gaussian_blur")
    validate_uint8(gray, "gaussian_blur")
    validate_ksize(ksize, "gaussian_blur")
    return cv2.GaussianBlur(gray, (ksize, ksize), sigma)


# ---------------------------------------------------------------------------
# A2. Median Blur
# ---------------------------------------------------------------------------

def median_blur(
    gray: np.ndarray,
    ksize: int = 5,
) -> np.ndarray:
    """Median filtresi uygular (tuz-biber gürültüsüne karşı etkilidir).

    Problem
    -------
    Tuz-biber (salt-and-pepper) gürültüsü, görüntüde rastgele siyah-beyaz
    noktalara yol açar. Median filtre, her pikseli penceresindeki medyan
    değerle değiştirerek kenarları korur ve aykırı impuls değerlerini temizler.

    Parametreler
    ------------
    gray : np.ndarray
        2 boyutlu gri tonlamalı görüntü dizisi (uint8).
    ksize : int, varsayılan 5
        Filtre pencere boyutu. Tek sayı ve >= 1 olmalıdır.

    Döndürür
    --------
    np.ndarray
        uint8, girdiyle aynı şekil.

    Başarısız Olabilir
    ------------------
    - ValueError : gray 2 boyutlu değilse veya ksize çift sayıysa.
    """
    validate_gray(gray, "median_blur")
    validate_uint8(gray, "median_blur")
    validate_ksize(ksize, "median_blur")
    return cv2.medianBlur(gray, ksize)


# ---------------------------------------------------------------------------
# A3. Bilateral Filter
# ---------------------------------------------------------------------------

def bilateral_filter(
    gray: np.ndarray,
    d: int = 9,
    sigma_color: float = 75,
    sigma_space: float = 75,
) -> np.ndarray:
    """Bilateral filtre uygular (kenar koruyucu yumuşatma).

    Problem
    -------
    Gaussian filtre hem gürültüyü hem de kenarları bulanıklaştırır.
    Bilateral filtre, hem konumsal yakınlığı hem de ton benzerliğini
    birlikte kullanarak kenarları korurken homojen alanları yumuşatır.

    Parametreler
    ------------
    gray : np.ndarray
        2 boyutlu gri tonlamalı görüntü dizisi (uint8).
    d : int, varsayılan 9
        Piksel komşuluk çapı.
    sigma_color : float, varsayılan 75
        Renk/ton uzayındaki sigma değeri.
    sigma_space : float, varsayılan 75
        Koordinat uzayındaki sigma değeri.

    Döndürür
    --------
    np.ndarray
        uint8, girdiyle aynı şekil.

    Başarısız Olabilir
    ------------------
    - ValueError : gray 2 boyutlu değilse.
    """
    validate_gray(gray, "bilateral_filter")
    validate_uint8(gray, "bilateral_filter")
    return cv2.bilateralFilter(gray, d, sigma_color, sigma_space)

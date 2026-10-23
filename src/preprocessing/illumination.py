"""Dengesiz aydınlatma için arka plan kestirimi ve düzleştirme yardımcıları."""

from __future__ import annotations

import cv2
import numpy as np

from src.utils.validation import validate_gray, validate_uint8


REFERENCE_HEIGHT = 2200
REFERENCE_WIDTH = 1650
BACKGROUND_MAX_DIMENSION = 600


def _scaled_odd_ksize(ksize: int | None, shape: tuple[int, int]) -> int:
    """Referans çözünürlüğüne göre kernel'i sınırlandırıp tek sayıya çevirir."""
    if ksize is None:
        ksize = 301
    if isinstance(ksize, bool) or not isinstance(ksize, int) or ksize < 3:
        raise ValueError("flatten_background: ksize en az 3 olan tek tam sayı olmalıdır")
    if ksize % 2 == 0:
        raise ValueError("flatten_background: ksize tek sayı olmalıdır")
    height, width = shape
    scale = min(height / REFERENCE_HEIGHT, width / REFERENCE_WIDTH)
    scaled = max(3, int(round((ksize - 1) * scale)) + 1)
    if scaled % 2 == 0:
        scaled += 1
    limit = min(height, width)
    if limit % 2 == 0:
        limit -= 1
    return max(1, min(scaled, limit))


def _flatten_gray(gray: np.ndarray, kernel_size: int) -> np.ndarray:
    """Küçük ölçekte arka planı kestirip tam çözünürlükte uygular."""
    height, width = gray.shape
    longest_side = max(height, width)
    if longest_side > BACKGROUND_MAX_DIMENSION:
        scale = BACKGROUND_MAX_DIMENSION / longest_side
        reduced_width = max(1, round(width * scale))
        reduced_height = max(1, round(height * scale))
        reduced = cv2.resize(
            gray, (reduced_width, reduced_height), interpolation=cv2.INTER_AREA
        )
        reduced_kernel = max(3, int(round((kernel_size - 1) * scale)) + 1)
        if reduced_kernel % 2 == 0:
            reduced_kernel += 1
        kernel_limit = min(reduced_height, reduced_width)
        if kernel_limit % 2 == 0:
            kernel_limit -= 1
        reduced_kernel = min(reduced_kernel, max(1, kernel_limit))
        kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (reduced_kernel, reduced_kernel)
        )
        reduced_background = cv2.morphologyEx(reduced, cv2.MORPH_CLOSE, kernel)
        background = cv2.resize(
            reduced_background, (width, height), interpolation=cv2.INTER_LINEAR
        )
    else:
        kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (kernel_size, kernel_size)
        )
        background = cv2.morphologyEx(gray, cv2.MORPH_CLOSE, kernel)
    return cv2.divide(gray, background, scale=255)


def flatten_background(img: np.ndarray, ksize: int | None = None) -> np.ndarray:
    """Gri veya BGR görüntüde yavaş değişen aydınlatmayı düzleştirir.

    Problem
    -------
    Yavaş değişen gölge ve aydınlatma gradyanı, global eşikleme öncesinde
    kâğıdın farklı bölgelerindeki parlaklıkları birbirinden uzaklaştırabilir.

    Algoritma
    ---------
    Uzun kenar 600 pikseli aşıyorsa görüntü küçültülür, kernel aynı oranda
    ölçeklenerek arka plan kestirilir ve kestirim tam çözünürlüğe büyütülür.
    Daha küçük görüntülerde tam çözünürlükte eliptik kernel ile closing
    uygulanır. Ardından
    ``cv2.divide(gray, bg, scale=255)`` ile normalize edilir. BGR girdide HSV'nin
    V (parlaklık) kanalı düzeltilir; H/S kanalları korunarak ayrı B/G/R kazançlarının
    oluşturabileceği renk kayması azaltılır.

    Parametreler
    ------------
    img : np.ndarray
        uint8, iki boyutlu gri veya üç kanallı BGR görüntü.
    ksize : int | None
        2200x1650 referansındaki tek sayılı kernel çapı; varsayılan 301.
        None ise 301 seçilir. Her iki durumda da gerçek görüntü boyutuna ölçeklenir.

    Başarısız Olabilir
    ------------------
    Kernel metin/nesne boyutundan küçükse closing onları arka plan sanabilir;
    aydınlatma hızlı değişiyorsa veya görüntü kenarında kâğıt kesilmişse kestirim
    bozulabilir. Koyu zemin/kâğıt ve büyük koyu alanlar da bu varsayımı bozar.
    """
    if not isinstance(img, np.ndarray):
        raise TypeError("flatten_background: NumPy görüntüsü bekleniyor")
    validate_uint8(img, "flatten_background")
    if img.ndim == 2:
        validate_gray(img, "flatten_background")
        spatial_shape = img.shape
    elif img.ndim == 3 and img.shape[2] == 3:
        spatial_shape = img.shape[:2]
    else:
        raise ValueError("flatten_background: gri veya üç kanallı BGR görüntü bekleniyor")
    if min(spatial_shape) < 1:
        raise ValueError("flatten_background: boş görüntü işlenemez")

    kernel_size = _scaled_odd_ksize(ksize, spatial_shape)
    if img.ndim == 2:
        return _flatten_gray(img, kernel_size)

    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    hsv[:, :, 2] = _flatten_gray(hsv[:, :, 2], kernel_size)
    return cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

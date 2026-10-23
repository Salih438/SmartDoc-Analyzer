"""
src/evaluation/noise.py
-----------------------
Algoritma dayanıklılığını (robustness) ölçmek için sentetik bozulma modülü.

Gaussian gürültü, tuz-biber (salt-and-pepper) gürültüsü ve dengesiz aydınlatma
(uneven illumination / gölge) simüle eden fonksiyonları içerir. Tüm fonksiyonlar
`seed` parametresi sayesinde %100 tekrarlanabilir deneyler sağlar.
"""

from __future__ import annotations

import cv2
import numpy as np

from src.utils.validation import validate_gray

_validate_gray = validate_gray


# ---------------------------------------------------------------------------
# 1. Gaussian Gürültü
# ---------------------------------------------------------------------------

def add_gaussian_noise(
    gray: np.ndarray,
    sigma: float = 20.0,
    seed: int | None = None,
) -> np.ndarray:
    """Görüntüye sıfır ortalamalı Gaussian (normal) gürültü ekler.

    Problem
    -------
    Düşük ışık veya sensör sıcaklığından kaynaklanan elektronik sensör gürültüsünü
    simüle ederek filtrelerin gürültü bastırma performansını ölçmek.

    Parametreler
    ------------
    gray : np.ndarray
        2 boyutlu gri tonlamalı görüntü dizisi (uint8).
    sigma : float, varsayılan 20.0
        Gaussian gürültüsünün standart sapması.
    seed : int | None, varsayılan None
        Rastgele sayı üreteci tohumu (tekrarlanabilirlik için).

    Döndürür
    --------
    np.ndarray
        uint8 formatında gürültülü görüntü [0, 255].

    Başarısız Olabilir
    ------------------
    - ValueError : gray 2 boyutlu değilse veya sigma < 0 ise.
    """
    validate_gray(gray, "add_gaussian_noise")
    if sigma < 0:
        raise ValueError(f"add_gaussian_noise: sigma negatif olamaz: {sigma}")

    rng = np.random.default_rng(seed)
    gauss = rng.normal(loc=0.0, scale=sigma, size=gray.shape)
    noisy = gray.astype(np.float64) + gauss
    return np.clip(noisy, 0, 255).astype(np.uint8)


# ---------------------------------------------------------------------------
# 2. Tuz ve Biber (Salt-and-Pepper) Gürültüsü
# ---------------------------------------------------------------------------

def add_salt_pepper_noise(
    gray: np.ndarray,
    amount: float = 0.05,
    seed: int | None = None,
) -> np.ndarray:
    """Görüntüye rastgele serpilmiş siyah (0) ve beyaz (255) pikseller ekler.

    Problem
    -------
    Tarayıcı tozları, iletim hatası veya arızalı piksel sensörleri nedeniyle
    oluşan ani impuls gürültüsünü simüle etmek.

    Parametreler
    ------------
    gray : np.ndarray
        2 boyutlu gri tonlamalı görüntü dizisi (uint8).
    amount : float, varsayılan 0.05
        Bozulacak piksel oranı [0.0, 1.0]. Örnek: 0.05 = piksellerin %5'i.
    seed : int | None, varsayılan None
        Rastgele sayı üreteci tohumu.

    Döndürür
    --------
    np.ndarray
        uint8 formatında tuz-biber gürültülü görüntü.

    Başarısız Olabilir
    ------------------
    - ValueError : gray 2 boyutlu değilse veya amount [0, 1] aralığında değilse.
    """
    validate_gray(gray, "add_salt_pepper_noise")
    if not (0.0 <= amount <= 1.0):
        raise ValueError(
            f"add_salt_pepper_noise: amount [0, 1] aralığında olmalı, alınan: {amount}"
        )

    out = gray.copy()
    if amount == 0.0:
        return out

    rng = np.random.default_rng(seed)
    h, w = gray.shape
    total_pixels = h * w
    num_noisy = int(round(amount * total_pixels))
    num_salt = num_noisy // 2
    num_pepper = num_noisy - num_salt

    # Tuz (beyaz 255)
    if num_salt > 0:
        salt_r = rng.integers(0, h, size=num_salt)
        salt_c = rng.integers(0, w, size=num_salt)
        out[salt_r, salt_c] = 255

    # Biber (siyah 0)
    if num_pepper > 0:
        pep_r = rng.integers(0, h, size=num_pepper)
        pep_c = rng.integers(0, w, size=num_pepper)
        out[pep_r, pep_c] = 0

    return out


# ---------------------------------------------------------------------------
# 3. Dengesiz Aydınlatma Simülasyonu (Uneven Illumination)
# ---------------------------------------------------------------------------

def simulate_uneven_illumination(
    gray: np.ndarray,
    strength: float = 0.5,
    seed: int | None = None,
) -> np.ndarray:
    """Belge üzerinde doğrusal gradyan ile dengesiz aydınlatma bozulması simüle eder.

    Problem
    -------
    Mobil cihazla çekilmiş belgelerde masa lambası, pencere ışığı ya da el gölgesi
    nedeniyle belgenin bir kısmının aşırı açık, diğer kısmının karanlık çıkması
    durumunu modellemek. Global eşikleme ile uyarlanabilir eşikleme arasındaki
    farkı test etmek için temel bir sentetik deney aracıdır.

    Algoritma ve Yön Seçimi
    -----------------------
    - `seed` parametresi verildiğinde gradyan yönü rastgele ('diag' çapraz,
      'horizontal' yatay veya 'vertical' dikey) seçilir. Aynı seed verildiğinde
      aynı yön ve gradyan katsayıları üretilerek deneyin tekrarlanabilirliği
      sağlanır.
    - `seed` verilmediğinde (None) varsayılan olarak sol üstten sağ alta doğru
      çapraz ('diag') gradyan kullanılır.
    Piksel parlaklığı şu formülle dengesizleştirilir:
    factor = 1.0 - (strength * gradyan)

    Parametreler
    ------------
    gray : np.ndarray
        2 boyutlu gri tonlamalı görüntü dizisi (uint8).
    strength : float, varsayılan 0.5
        Bozulma gücü [0.0, 1.0]. 0.0 etki yaratmaz; 1.0 bir ucu tamamen karartır.
    seed : int | None, varsayılan None
        Rastgele yön varyasyonu ve tekrarlanabilirlik için tohum değeri.

    Döndürür
    --------
    np.ndarray
        uint8 formatında dengesiz aydınlatılmış görüntü.

    Başarısız Olabilir
    ------------------
    - ValueError : gray 2 boyutlu değilse veya strength [0, 1] aralığında değilse.
    """
    validate_gray(gray, "simulate_uneven_illumination")
    if not (0.0 <= strength <= 1.0):
        raise ValueError(
            f"simulate_uneven_illumination: strength [0, 1] aralığında olmalı, alınan: {strength}"
        )

    if strength == 0.0:
        return gray.copy()

    h, w = gray.shape
    rng = np.random.default_rng(seed)

    # seed verildiğinde yön rastgele seçilir; aynı seed aynı yönü üretir.
    direction = rng.choice(["diag", "horizontal", "vertical"]) if seed is not None else "diag"

    if direction == "diag":
        x = np.linspace(0.0, 1.0, w, dtype=np.float64)
        y = np.linspace(0.0, 1.0, h, dtype=np.float64)
        xx, yy = np.meshgrid(x, y)
        grad = (xx + yy) / 2.0
    elif direction == "horizontal":
        x = np.linspace(0.0, 1.0, w, dtype=np.float64)
        grad = np.tile(x, (h, 1))
    else:
        y = np.linspace(0.0, 1.0, h, dtype=np.float64)
        grad = np.tile(y[:, np.newaxis], (1, w))

    factor = 1.0 - (strength * grad)
    corrupted = gray.astype(np.float64) * factor
    return np.clip(corrupted, 0, 255).astype(np.uint8)

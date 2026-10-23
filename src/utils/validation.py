"""
src/utils/validation.py
-----------------------
Görüntü işleme fonksiyonları için merkezi girdi doğrulama yardımcıları.

Bu modül, modüller arasında tekrarlanan girdi türü, boyut, çekirdek (ksize)
ve eşleşme denetimlerini tek bir standart çatı altında toplar.
"""

from __future__ import annotations

from typing import Any
import numpy as np


def validate_uint8(img: np.ndarray, func_name: str) -> None:
    """Görüntünün OpenCV işlem hattı için uint8 olmasını doğrular.

    Problem: OpenCV'nin filtre, eşik, morfoloji, keskinleştirme ve Canny
    çağrıları bu modülde açıkça uint8 ile sınırlandırılır. Parametreler:
    ``img`` görüntüsü ve hata mesajındaki ``func_name``. Başarısız olur:
    float32, int32, uint16 gibi diğer dtype'larda açık ``TypeError`` üretir.
    """
    if not isinstance(img, np.ndarray):
        raise TypeError(f"{func_name}: np.ndarray bekleniyor, alınan: {type(img)}")
    if img.dtype != np.uint8:
        raise TypeError(
            f"{func_name}: uint8 görüntü bekleniyor, alınan dtype: {img.dtype}"
        )


def validate_gray(img: np.ndarray, func_name: str) -> None:
    """Girdinin 2 boyutlu (2-D) gri tonlamalı bir NumPy dizisi olduğunu doğrular.

    Parametreler
    ------------
    img : np.ndarray
        Doğrulanacak görüntü dizisi.
    func_name : str
        Hata mesajlarında raporlanacak çağıran fonksiyon adı.

    Hatalar
    -------
    TypeError
        Girdi np.ndarray türünde değilse.
    ValueError
        Girdi 2 boyutlu değilse (ör. 3 boyutlu renkli görüntü).
    """
    if not isinstance(img, np.ndarray):
        raise TypeError(f"{func_name}: np.ndarray bekleniyor, alınan: {type(img)}")
    if img.ndim != 2:
        raise ValueError(
            f"{func_name}: 2 boyutlu gri tonlamalı görüntü bekleniyor, alınan şekil: {img.shape}. "
            "Renkli bir görüntü kullanıyorsanız önce to_grayscale() uygulayınız."
        )


def validate_ksize(ksize: int, func_name: str) -> None:
    """Çekirdek boyutunun pozitif ve tek sayı olduğunu doğrular.

    Parametreler
    ------------
    ksize : int
        Filtre veya morfoloji çekirdek boyutu.
    func_name : str
        Çağıran fonksiyon adı.

    Hatalar
    -------
    ValueError
        ksize < 1 ise veya çift sayı ise.
    """
    if ksize < 1 or ksize % 2 == 0:
        raise ValueError(
            f"{func_name}: ksize tek sayı ve >= 1 olmalı, alınan: {ksize}"
        )


def validate_binary(img: np.ndarray, func_name: str) -> None:
    """Girdinin 2 boyutlu ikili görüntü olduğunu doğrular.

    Parametreler
    ------------
    img : np.ndarray
        Doğrulanacak ikili (binary) görüntü.
    func_name : str
        Çağıran fonksiyon adı.
    """
    validate_gray(img, func_name)


def validate_match(ref: np.ndarray, img: np.ndarray, func_name: str) -> None:
    """İki görüntünün şekil ve veri tiplerinin birebir eşleştiğini doğrular.

    Parametreler
    ------------
    ref : np.ndarray
        Referans görüntü.
    img : np.ndarray
        Karşılaştırılacak hedef görüntü.
    func_name : str
        Çağıran fonksiyon adı.

    Hatalar
    -------
    TypeError
        Girdilerden biri np.ndarray değilse.
    ValueError
        Şekil (shape) veya veri tipi (dtype) uyuşmuyorsa.
    """
    if not isinstance(ref, np.ndarray) or not isinstance(img, np.ndarray):
        raise TypeError(f"{func_name}: Her iki girdi de np.ndarray olmalıdır.")
    if ref.shape != img.shape:
        raise ValueError(
            f"{func_name}: Şekil uyuşmazlığı! ref={ref.shape}, img={img.shape}"
        )
    if ref.dtype != img.dtype:
        raise ValueError(
            f"{func_name}: Veri tipi (dtype) uyuşmazlığı! ref={ref.dtype}, img={img.dtype}"
        )

"""
src/evaluation/metrics.py
-------------------------
Görüntü işleme ve iyileştirme algoritmalarının nicel değerlendirme metrikleri.

PSNR, SSIM, Laplacian varyansı ile keskinlik ölçümü, ikili segmentasyon metrikleri
(Precision, Recall, F1-Score, IoU), ön plan piksel oranı (foreground_ratio)
ve tekrarlı medyan süre ölçümünü (timed_median) sunar.
"""

from __future__ import annotations

import time
from typing import Any, Callable, TypeVar

import cv2
import numpy as np
from skimage.metrics import structural_similarity as ssim_fn

from src.utils.validation import validate_gray, validate_match


T = TypeVar("T")


# ---------------------------------------------------------------------------
# 1. PSNR (Peak Signal-to-Noise Ratio)
# ---------------------------------------------------------------------------

def psnr(
    ref: np.ndarray,
    img: np.ndarray,
    max_val: float = 255.0,
) -> float:
    """Referans ve hedef görüntü arasındaki Tepeden Tepeye Sinyal-Gürültü Oranını hesaplar.

    Problem
    -------
    İyileştirme veya filtreleme sonrası görüntünün referansa göre piksel bazında
    ne kadar bozulduğunu veya yaklaştığını dB cinsinden ölçmek.

    Özel Davranış (MSE == 0)
    ------------------------
    İki görüntü tamamen özdeş olduğunda (MSE == 0), matematiksel formüldeki
    log(1/0) tanımsız/sonsuz (inf) sonucunu verir. Sayısal hesaplamalarda ve veri
    kayıtlarında (CSV, JSON) NaN/inf tutarsızlıklarını önlemek amacıyla, özdeş
    görüntüler için açık bir üst sınır olarak **100.0 dB** değeri döndürülür.

    Parametreler
    ------------
    ref : np.ndarray
        Referans görüntü dizisi.
    img : np.ndarray
        Karşılaştırılacak görüntü dizisi (ref ile aynı şekil ve dtype olmalı).
    max_val : float, varsayılan 255.0
        Piksel değerinin alabileceği maksimum dinamik aralık genliği.

    Döndürür
    --------
    float
        dB cinsinden PSNR değeri.

    Başarısız Olabilir
    ------------------
    - ValueError : ref ve img boyutları veya veri tipleri (dtype) uyuşmuyorsa.
    - TypeError  : girdiler np.ndarray değilse.
    """
    validate_match(ref, img, "psnr")

    diff = ref.astype(np.float64) - img.astype(np.float64)
    mse = float(np.mean(diff ** 2))

    if mse == 0.0:
        return 100.0

    return float(10.0 * np.log10((max_val ** 2) / mse))


# ---------------------------------------------------------------------------
# 2. SSIM (Structural Similarity Index)
# ---------------------------------------------------------------------------

def ssim(
    ref: np.ndarray,
    img: np.ndarray,
    data_range: float | None = None,
) -> float:
    """Yapısal Benzerlik İndeksini (SSIM) hesaplar (scikit-image tabanlı).

    Problem
    -------
    İnsan görme sisteminin parlaklık, kontrast ve yapısal örüntüleri algılama
    biçimini modelleyerek, iki görüntü arasındaki algısal benzerliği ölçmek.

    Parametreler
    ------------
    ref : np.ndarray
        Referans görüntü dizisi (2-D veya 3-D).
    img : np.ndarray
        Karşılaştırılacak hedef görüntü dizisi.
    data_range : float | None, varsayılan None
        Dinamik piksel aralığı. None ise uint8 için 255.0 seçilir.

    Döndürür
    --------
    float
        [-1, 1] aralığında yapısal benzerlik skoru. 1.0 kusursuz benzerliktir.
    """
    validate_match(ref, img, "ssim")

    if data_range is None:
        if ref.dtype == np.uint8:
            dr = 255.0
        else:
            span = float(ref.max() - ref.min())
            dr = max(span, 1.0)
    else:
        dr = float(data_range)

    channel_axis = 2 if ref.ndim == 3 else None
    score = ssim_fn(ref, img, data_range=dr, channel_axis=channel_axis)
    return float(score)


# ---------------------------------------------------------------------------
# 3. Laplacian Varyansı ile Keskinlik (Blurriness Metric)
# ---------------------------------------------------------------------------

def sharpness_laplacian_var(gray: np.ndarray) -> float:
    """Laplacian filtresinin varyansını hesaplayarak görüntünün keskinliğini ölçer.

    Problem
    -------
    Referans görüntü olmadan (no-reference), tek bir belgenin ne kadar keskin veya
    odak dışı (bulanık) olduğunu belirlemek.

    Parametreler
    ------------
    gray : np.ndarray
        2 boyutlu gri tonlamalı görüntü dizisi.

    Döndürür
    --------
    float
        Laplacian varyansı. Yüksek değer daha keskin kenarları ve yüksek uzamsal frekansı gösterir.
    """
    validate_gray(gray, "sharpness_laplacian_var")
    lap = cv2.Laplacian(gray, cv2.CV_64F)
    return float(lap.var())


# ---------------------------------------------------------------------------
# 4. İkili Segmentasyon Metrikleri (Precision, Recall, F1, IoU)
# ---------------------------------------------------------------------------

def precision(gt_mask: np.ndarray, pred_mask: np.ndarray) -> float:
    """İkili segmentasyon doğruluğunu (kesinlik / pozitif tahmin değeri) hesaplar.

    Formül: TP / (TP + FP)
    Ön plan pikselleri 255 (veya > 0) olarak kabul edilir.
    """
    validate_match(gt_mask, pred_mask, "precision")
    gt = (gt_mask > 0)
    pred = (pred_mask > 0)
    tp = np.count_nonzero(gt & pred)
    fp = np.count_nonzero(~gt & pred)
    if tp + fp == 0:
        return 1.0 if np.count_nonzero(gt) == 0 else 0.0
    return float(tp / (tp + fp))


def recall(gt_mask: np.ndarray, pred_mask: np.ndarray) -> float:
    """İkili segmentasyon duyarlılığını (yakalama oranı) hesaplar.

    Formül: TP / (TP + FN)
    """
    validate_match(gt_mask, pred_mask, "recall")
    gt = (gt_mask > 0)
    pred = (pred_mask > 0)
    tp = np.count_nonzero(gt & pred)
    fn = np.count_nonzero(gt & ~pred)
    if tp + fn == 0:
        return 1.0 if np.count_nonzero(pred) == 0 else 0.0
    return float(tp / (tp + fn))


def f1_score(gt_mask: np.ndarray, pred_mask: np.ndarray) -> float:
    """Precision ve Recall'un harmonik ortalaması olan F1-Score'u hesaplar.

    Formül: 2 * (Precision * Recall) / (Precision + Recall)
    """
    p = precision(gt_mask, pred_mask)
    r = recall(gt_mask, pred_mask)
    if p + r == 0.0:
        return 0.0
    return float(2.0 * p * r / (p + r))


def iou_score(gt_mask: np.ndarray, pred_mask: np.ndarray) -> float:
    """Kesişim / Birleşim Oranını (Jaccard İndeksi) hesaplar.

    Formül: Area(GT ∩ PRED) / Area(GT ∪ PRED)
    """
    validate_match(gt_mask, pred_mask, "iou_score")
    gt = (gt_mask > 0)
    pred = (pred_mask > 0)
    intersection = np.count_nonzero(gt & pred)
    union = np.count_nonzero(gt | pred)
    if union == 0:
        return 1.0
    return float(intersection / union)


def foreground_ratio(mask: np.ndarray) -> float:
    """İkili görüntüdeki ön plan (255) piksellerinin tüm görüntüye oranını hesaplar.

    Parametreler
    ------------
    mask : np.ndarray
        2 boyutlu ikili görüntü dizisi.

    Döndürür
    --------
    float
        Ön plan piksellerinin toplam piksel sayısına oranı [0.0, 1.0].
    """
    validate_gray(mask, "foreground_ratio")
    if mask.size == 0:
        raise ValueError("foreground_ratio: Boş görüntü verilemez.")
    return float(np.count_nonzero(mask > 0) / mask.size)


def corner_error(
    gt: np.ndarray, pred: np.ndarray | None
) -> tuple[float, float]:
    """Dairesel köşe eşleşmesiyle ortalama piksel ve köşegen hatası döndürür."""
    if pred is None:
        return float("inf"), float("inf")
    gt_arr = np.asarray(gt, dtype=np.float32).reshape(4, 2)
    pred_arr = np.asarray(pred, dtype=np.float32).reshape(4, 2)
    diagonal = float(np.linalg.norm(gt_arr[0] - gt_arr[2]))
    errors = [
        float(np.mean(np.linalg.norm(gt_arr - np.roll(pred_arr, shift, axis=0), axis=1)))
        for shift in range(4)
    ]
    mean_error = min(errors)
    return mean_error, mean_error / diagonal if diagonal > 0 else float("inf")


def quad_iou(gt: np.ndarray, pred: np.ndarray | None, shape: tuple[int, int]) -> float:
    """İki dörtgeni dolu maskelere çevirerek IoU hesaplar."""
    if pred is None:
        return 0.0
    gt_mask = np.zeros(shape[:2], dtype=np.uint8)
    pred_mask = np.zeros(shape[:2], dtype=np.uint8)
    cv2.fillPoly(gt_mask, [np.rint(gt).astype(np.int32)], 1)
    cv2.fillPoly(pred_mask, [np.rint(pred).astype(np.int32)], 1)
    union = np.count_nonzero(gt_mask | pred_mask)
    return float(np.count_nonzero(gt_mask & pred_mask) / union) if union else 1.0


def rectification_ssim(original_doc: np.ndarray, warped: np.ndarray) -> float:
    """Orijinali warp boyutuna getirip dört dönme yönünün en iyi SSIM'ini döndürür."""
    if original_doc.ndim == 3:
        original_gray = cv2.cvtColor(original_doc, cv2.COLOR_BGR2GRAY)
    else:
        original_gray = original_doc
    if warped.ndim == 3:
        warped_gray = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY)
    else:
        warped_gray = warped
    target_size = (warped_gray.shape[1], warped_gray.shape[0])
    references = [
        cv2.resize(np.ascontiguousarray(np.rot90(original_gray, k)), target_size)
        for k in range(4)
    ]
    return max(ssim(reference, warped_gray, data_range=255.0) for reference in references)


# ---------------------------------------------------------------------------
# 5. Süre Ölçümü (Timed & Timed Median)
# ---------------------------------------------------------------------------

def timed(fn: Callable[..., T], *args: Any, **kwargs: Any) -> tuple[T, float]:
    """Bir fonksiyonu bir kez çalıştırır ve (sonuç, süre_ms) döndürür."""
    start = time.perf_counter()
    result = fn(*args, **kwargs)
    elapsed_ms = (time.perf_counter() - start) * 1000.0
    return result, float(elapsed_ms)


def timed_median(
    fn: Callable[..., T],
    *args: Any,
    repeats: int = 20,
    warmup: int = 3,
    **kwargs: Any,
) -> tuple[T, float]:
    """Bir fonksiyonu tekrarlı çalıştırarak medyan çalışma süresini (ms) ölçer.

    Problem
    -------
    İşletim sistemi görev zamanlaması, önbellek soğukluğu ve anlık arka plan
    kesintilerinin yol açtığı süre dalgalanmalarını filtrelemek; deterministik
    ve gürültüsüz bir çalışma süresi ölçümü sunmak.

    Parametreler
    ------------
    fn : Callable
        Çalıştırılacak fonksiyon.
    *args, **kwargs
        Fonksiyona aktarılacak parametreler.
    repeats : int, varsayılan 20
        Ölçüm için yapılacak tekrar sayısı.
    warmup : int, varsayılan 3
        Ölçüm öncesi CPU önbelleğini ısıtmak için yapılan, hesaba katılmayan ısınma tekrar sayısı.

    Döndürür
    --------
    tuple[T, float]
        (fonksiyon_sonucu, medyan_sure_ms)
    """
    # Isınma turları
    for _ in range(max(0, warmup)):
        fn(*args, **kwargs)

    durations: list[float] = []
    last_res: Any = None
    for _ in range(max(1, repeats)):
        start = time.perf_counter()
        last_res = fn(*args, **kwargs)
        elapsed_ms = (time.perf_counter() - start) * 1000.0
        durations.append(elapsed_ms)

    median_ms = float(np.median(durations))
    return last_res, median_ms

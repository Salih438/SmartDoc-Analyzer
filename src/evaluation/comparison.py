"""
src/evaluation/comparison.py
----------------------------
Filtreleme ve eşikleme algoritmalarının tarafsız karşılaştırılması modülü.

Filtrelerin (Gaussian, Median, Bilateral), eşikleme yöntemlerinin (Global, Otsu,
Adaptive) ve aydınlatma iyileştirmelerinin çıktılarını/çalışma sürelerini derler.
PSNR/SSIM yalnızca temiz referans görüntüsü sağlanan karşılaştırmalarda hesaplanır.
"""

from __future__ import annotations

from typing import Any
import numpy as np

from src.utils.validation import validate_gray
from src.preprocessing.filtering import gaussian_blur, median_blur, bilateral_filter
from src.preprocessing.enhancement import (
    apply_clahe,
    contrast_stretch,
    gamma_correction,
    histogram_equalization,
)
from src.preprocessing.auto_parameters import (
    AUTO_GAMMA_BRIGHT_MEAN,
    AUTO_GAMMA_DARKEN,
    AUTO_GAMMA_DARK_MEAN,
    AUTO_GAMMA_LIGHTEN,
)
from src.segmentation.threshold import global_threshold, otsu_threshold, adaptive_threshold
from src.evaluation.metrics import psnr, ssim, timed, foreground_ratio

_validate_gray = validate_gray
def _gamma_auto(gray: np.ndarray) -> tuple[np.ndarray, str, dict[str, float]]:
    """Ortalama parlaklığa göre gamma seçer ve seçimi raporlar."""
    mean = float(gray.astype(np.float64).mean())
    if mean < AUTO_GAMMA_DARK_MEAN:
        gamma = AUTO_GAMMA_LIGHTEN
    elif mean > AUTO_GAMMA_BRIGHT_MEAN:
        gamma = AUTO_GAMMA_DARKEN
    else:
        gamma = 1.0
    return gamma_correction(gray, gamma), "gamma", {"gamma": gamma, "mean": mean}


def compare_filters(
    gray_clean: np.ndarray,
    gray_noisy: np.ndarray,
    ksize: int = 5,
) -> dict[str, dict[str, Any]]:
    """Gaussian, Median ve Bilateral filtrelerini PSNR, SSIM ve süre açısından karşılaştırır."""
    validate_gray(gray_clean, "compare_filters")
    validate_gray(gray_noisy, "compare_filters")
    if gray_clean.shape != gray_noisy.shape:
        raise ValueError(
            f"compare_filters: Boyut uyuşmazlığı! clean={gray_clean.shape}, noisy={gray_noisy.shape}"
        )

    results: dict[str, dict[str, Any]] = {}

    img_gauss, t_gauss = timed(gaussian_blur, gray_noisy, ksize=ksize)
    results["gaussian"] = {
        "image": img_gauss,
        "psnr": psnr(gray_clean, img_gauss),
        "ssim": ssim(gray_clean, img_gauss),
        "duration_ms": t_gauss,
    }

    img_med, t_med = timed(median_blur, gray_noisy, ksize=ksize)
    results["median"] = {
        "image": img_med,
        "psnr": psnr(gray_clean, img_med),
        "ssim": ssim(gray_clean, img_med),
        "duration_ms": t_med,
    }

    img_bilat, t_bilat = timed(bilateral_filter, gray_noisy, d=ksize, sigma_color=75, sigma_space=75)
    results["bilateral"] = {
        "image": img_bilat,
        "psnr": psnr(gray_clean, img_bilat),
        "ssim": ssim(gray_clean, img_bilat),
        "duration_ms": t_bilat,
    }

    return results


def compare_thresholds(
    gray: np.ndarray,
    invert: bool = False,
) -> dict[str, dict[str, Any]]:
    """Global, Otsu ve Adaptive eşikleme yöntemlerini karşılaştırır."""
    validate_gray(gray, "compare_thresholds")

    results: dict[str, dict[str, Any]] = {}

    img_glob, t_glob = timed(global_threshold, gray, thresh=127, invert=invert)
    results["global"] = {
        "image": img_glob,
        "duration_ms": t_glob,
        "foreground_ratio": foreground_ratio(img_glob),
        "threshold_val": 127.0,
    }

    (img_otsu, val_otsu), t_otsu = timed(otsu_threshold, gray, invert=invert)
    results["otsu"] = {
        "image": img_otsu,
        "duration_ms": t_otsu,
        "foreground_ratio": foreground_ratio(img_otsu),
        "threshold_val": float(val_otsu),
    }

    img_ad_mean, t_ad_mean = timed(adaptive_threshold, gray, method="mean", block_size=11, c=2, invert=invert)
    results["adaptive_mean"] = {
        "image": img_ad_mean,
        "duration_ms": t_ad_mean,
        "foreground_ratio": foreground_ratio(img_ad_mean),
        "threshold_val": None,
    }

    img_ad_gauss, t_ad_gauss = timed(adaptive_threshold, gray, method="gaussian", block_size=11, c=2, invert=invert)
    results["adaptive_gaussian"] = {
        "image": img_ad_gauss,
        "duration_ms": t_ad_gauss,
        "foreground_ratio": foreground_ratio(img_ad_gauss),
        "threshold_val": None,
    }

    return results


def compare_enhancements(gray: np.ndarray) -> dict[str, dict[str, Any]]:
    """İyileştirme yöntemlerini aynı gri görüntüde çıktı ve süreyle karşılaştırır.

    Temiz referans verilmediğinden PSNR/SSIM hesaplanmaz; yalnızca yöntem
    çıktısı ve süre döndürülür. Otomatik gamma, ortalama parlaklık eşiklerini
    kullanır ve seçilen gamma değerini parametrelerde bildirir.
    """
    validate_gray(gray, "compare_enhancements")
    methods = {
        "none": (lambda: gray.copy(), {}),
        "contrast_stretch": (lambda: contrast_stretch(gray), {"low_pct": 2, "high_pct": 98}),
        "gamma_auto": (lambda: _gamma_auto(gray), {"mode": "brightness_based"}),
        "gamma_0.6": (lambda: gamma_correction(gray, 0.6), {"gamma": 0.6}),
        "gamma_0.8": (lambda: gamma_correction(gray, 0.8), {"gamma": 0.8}),
        "gamma_1.2": (lambda: gamma_correction(gray, 1.2), {"gamma": 1.2}),
        "gamma_1.6": (lambda: gamma_correction(gray, 1.6), {"gamma": 1.6}),
        "gamma_2.0": (lambda: gamma_correction(gray, 2.0), {"gamma": 2.0}),
        "clahe_1.0": (lambda: apply_clahe(gray, clip_limit=1.0), {"clip_limit": 1.0}),
        "clahe_2.0": (lambda: apply_clahe(gray, clip_limit=2.0), {"clip_limit": 2.0}),
        "clahe_4.0": (lambda: apply_clahe(gray, clip_limit=4.0), {"clip_limit": 4.0}),
        "histogram_equalization": (lambda: histogram_equalization(gray), {}),
    }
    results: dict[str, dict[str, Any]] = {}
    for name, (operation, parameters) in methods.items():
        output, duration = timed(operation)
        selected_parameters = dict(parameters)
        if name == "gamma_auto":
            output, selected, auto_parameters = output
            selected_parameters = {"selected_method": selected, **auto_parameters}
        results[name] = {
            "image": output,
            "duration_ms": duration,
            "parameters": selected_parameters,
        }
    return results

"""Klasik OpenCV/NumPy belge işleme hattı ve dosya tabanlı çalışma yardımcıları.

Varsayılanlar, proje CSV'lerindeki ölçümlerden hareketle seçilmiştir: belge
tespiti için Canny/fixed kombinasyonu 12 senaryonun 11'inde 5/5 başarı koşulunu
sağladı; ``partial_crop`` beklenen ret olarak ayrıca raporlanır. Threshold,
low-contrast ve clutter'da 0/5 başarı verdi; ``low_contrast_hard`` için de
ölçüm CSV'si üretilmiştir.
Segmentasyonda normal ışıkta Otsu ölçümü F1=0.9977, 3B'de 0.9951 vermiştir.
Bu nedenle varsayılan binarize Otsu'dur; diğer deneylerde koşula göre farklı
ölçümler alınmıştır. Filtreleme, keskinleştirme ve morfoloji varsayılan olarak
kapalıdır; 3F'deki Median(5)+Bilateral etkisi adaptive eşiklerle sınırlı
değiştiğinden her görüntüye zorunlu uygulanmaz. Bunlar yapılandırma
varsayılanlarıdır, yöntem üstünlüğü iddiası değildir.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
import json
from pathlib import Path
import time
from typing import Literal

import cv2
import numpy as np

from src.analysis.image_info import get_image_info, load_image, save_image
from src.document.detection import MAX_CONFIDENT_AREA_RATIO, DetectionResult, detect_document
from src.document.perspective import four_point_transform
from src.evaluation.metrics import sharpness_laplacian_var
from src.preprocessing.enhancement import apply_clahe, contrast_stretch, gamma_correction
from src.preprocessing.auto_parameters import (
    AUTO_GAMMA_BRIGHT_MEAN,
    AUTO_GAMMA_DARKEN,
    AUTO_GAMMA_DARK_MEAN,
    AUTO_GAMMA_LIGHTEN,
    AUTO_LOW_CONTRAST_STD,
    AUTO_NOISE_SIGMA_THRESHOLD,
)
from src.preprocessing.filtering import bilateral_filter, median_blur
from src.preprocessing.grayscale import to_grayscale
from src.preprocessing.sharpening import unsharp_mask
from src.segmentation.morphology import closing, opening
from src.segmentation.threshold import adaptive_threshold, otsu_threshold


REFERENCE_HEIGHT = 2200
REFERENCE_WIDTH = 1650
MIN_FILTER_KERNEL = 3
MAX_FILTER_KERNEL = 31
MIN_ADAPTIVE_BLOCK = 3
MAX_ADAPTIVE_BLOCK = 99
CONFIGURED_GAMMA = 1.2

EnhanceMode = Literal["none", "auto", "stretch", "gamma", "clahe"]
DenoiseMode = Literal["none", "auto", "median", "bilateral"]
BinarizeMode = Literal["none", "otsu", "adaptive_mean", "adaptive_gaussian"]
MorphologyMode = Literal["none", "opening", "closing", "open_close"]
OcrSource = Literal["binary", "gray", "denoised"]


@dataclass
class PipelineConfig:
    """Pipeline seçimleri; varsayılanların CSV dayanağı modül docstring'indedir."""

    detect: bool = True
    detect_denoise: bool = False
    detection_fallback: bool = False
    detection_strategy: Literal["canny", "threshold"] = "canny"
    canny_mode: Literal["fixed", "auto", "otsu"] = "fixed"
    enhance: EnhanceMode = "none"
    denoise: DenoiseMode = "none"
    sharpen: Literal["none", "unsharp"] = "none"
    binarize: BinarizeMode = "otsu"
    morphology: MorphologyMode = "none"
    allow_unrectified: bool = False
    ocr_source: OcrSource = "binary"

    def __post_init__(self) -> None:
        """Geçersiz pipeline seçimini çalışmaya başlamadan reddeder."""
        choices = {
            "detection_strategy": ("canny", "threshold"),
            "canny_mode": ("fixed", "auto", "otsu"),
            "enhance": ("none", "auto", "stretch", "gamma", "clahe"),
            "denoise": ("none", "auto", "median", "bilateral"),
            "sharpen": ("none", "unsharp"),
            "binarize": ("none", "otsu", "adaptive_mean", "adaptive_gaussian"),
            "morphology": ("none", "opening", "closing", "open_close"),
            "ocr_source": ("binary", "gray", "denoised"),
        }
        for name, allowed in choices.items():
            if getattr(self, name) not in allowed:
                raise ValueError(f"{name} geçersiz; desteklenenler: {', '.join(allowed)}")
        if not isinstance(self.detect_denoise, bool):
            raise ValueError("detect_denoise bool olmalıdır")
        if not isinstance(self.detection_fallback, bool):
            raise ValueError("detection_fallback bool olmalıdır")


@dataclass
class PipelineResult:
    """İşlenmiş ara çıktılar, tespit sonucu, ölçümler ve uyarılar."""

    original: np.ndarray
    rectified: bool
    gray: np.ndarray | None
    enhanced: np.ndarray | None
    denoised: np.ndarray | None
    binary: np.ndarray | None
    detection: DetectionResult
    steps: list[dict[str, object]] = field(default_factory=list)
    info_before: dict[str, object] = field(default_factory=dict)
    info_after: dict[str, object] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)
    rectified_image: np.ndarray | None = None


def select_ocr_input(
    result: PipelineResult, config: PipelineConfig | None = None
) -> tuple[np.ndarray | None, str]:
    """Yapılandırılan ara görüntüyü OCR'a verir; bulunmazsa griye düşer.

    ``binary`` seçiliyken binarize kapalıysa ve ``denoised`` görüntü yoksa
    ``gray`` kullanılır. Dönüşte OCR'a verilecek görüntü ve fiili kaynak adı
    döner; görüntü kopyalandığından sonuç dizisi değiştirilemez.
    """
    source = (config or PipelineConfig()).ocr_source
    selected = {
        "binary": result.binary,
        "gray": result.gray,
        "denoised": result.denoised,
    }[source]
    actual_source = source
    if selected is None and source != "gray":
        selected = result.gray
        actual_source = "gray"
    return (selected.copy() if selected is not None else None), actual_source


def scale_odd_parameter(value: int, shape: tuple[int, ...], minimum: int, maximum: int) -> int:
    """1650×2200 referansına göre tek parametreyi tek sayıya ölçekler.

    Boyut oranı iki eksenin küçüğünden alınır; böylece en küçük eksende
    pencerelerin orantısız büyümesi önlenir. Min/max sınırları uygulanır.
    Çok küçük görüntülerde alt sınır görüntü boyutundan büyükse OpenCV
    işleminin geçerli olacağı en büyük tek değer seçilir.
    """
    if value < 1 or minimum < 1 or maximum < minimum:
        raise ValueError("ölçekleme parametreleri pozitif ve sınırlar sıralı olmalı")
    height, width = int(shape[0]), int(shape[1])
    scale = min(height / REFERENCE_HEIGHT, width / REFERENCE_WIDTH)
    scaled = int(round((value - 1) * scale)) + 1
    scaled = max(minimum, min(maximum, scaled))
    if scaled % 2 == 0:
        scaled = scaled + 1 if scaled < maximum else scaled - 1
    dimension_limit = min(height, width)
    if dimension_limit % 2 == 0:
        dimension_limit -= 1
    return max(1, min(scaled, dimension_limit))


def _record_step(
    steps: list[dict[str, object]], name: str, parameters: dict[str, object], started: float
) -> None:
    steps.append({"ad": name, "parametreler": parameters, "sure_ms": (time.perf_counter() - started) * 1000.0})


def _noise_sigma(gray: np.ndarray) -> float:
    """Laplacian MAD tabanlı yaklaşık gürültü sigma değerini hesaplar."""
    lap = cv2.Laplacian(gray, cv2.CV_64F, ksize=3)
    center = float(np.median(lap))
    mad = float(np.median(np.abs(lap - center)))
    return mad / (0.67448975 * np.sqrt(20.0))


def _enhance_auto(gray: np.ndarray) -> tuple[np.ndarray, str, dict[str, object]]:
    info = get_image_info(gray)
    mean = float(info["mean_brightness"])
    contrast = float(info["std_contrast"])
    if mean < AUTO_GAMMA_DARK_MEAN:
        return gamma_correction(gray, AUTO_GAMMA_LIGHTEN), "gamma", {"gamma": AUTO_GAMMA_LIGHTEN, "mean": mean}
    if mean > AUTO_GAMMA_BRIGHT_MEAN:
        return gamma_correction(gray, AUTO_GAMMA_DARKEN), "gamma", {"gamma": AUTO_GAMMA_DARKEN, "mean": mean}
    if contrast < AUTO_LOW_CONTRAST_STD:
        return contrast_stretch(gray), "stretch", {"std_esik": AUTO_LOW_CONTRAST_STD, "std": contrast}
    return gray.copy(), "none", {"mean": mean, "std": contrast}


def _analysis_info(img: np.ndarray) -> dict[str, object]:
    info: dict[str, object] = get_image_info(img)
    gray = to_grayscale(img)
    info["sharpness"] = sharpness_laplacian_var(gray)
    return info


def process_image(img: np.ndarray, config: PipelineConfig | None = None) -> PipelineResult:
    """İşler görüntüyü, başarısız tespitte sessizce perspektif düzeltmeden geçmez.

    Algoritma
    ---------
    Ön analiz → orijinal çözünürlükte tespit → başarılıysa warp → gri →
    enhance → denoise → sharpen → threshold → morphology → son analiz.
    Başarısız tespit varsayılan olarak sonraki adımları durdurur; yalnızca
    ``allow_unrectified`` açıkken ham görüntüyle devam edilir.

    Parametreler
    ------------
    img: uint8 gri/BGR/BGRA NumPy görüntüsü.
    config: PipelineConfig veya varsayılan seçimler.

    Başarısız olabilir
    ------------------
    Uygun dtype/kanal biçimi değilse TypeError/ValueError; dejenere tespit
    köşelerinde perspektif dönüşümü başarısız olabilir.
    """
    cfg = config or PipelineConfig()
    if not isinstance(img, np.ndarray) or img.dtype != np.uint8:
        raise TypeError("process_image: uint8 NumPy görüntüsü bekleniyor")
    if img.ndim not in (2, 3) or (img.ndim == 3 and img.shape[2] not in (3, 4)):
        raise ValueError("process_image: gri, BGR veya BGRA görüntü bekleniyor")
    original = img.copy()
    steps: list[dict[str, object]] = []
    warnings: list[str] = []
    started = time.perf_counter()
    info_before = _analysis_info(original)
    _record_step(steps, "analyze_before", {}, started)

    if cfg.detect:
        detection_input = original
        detection_parameters: dict[str, object] = {
            "enabled": cfg.detect_denoise,
            "method": "median" if cfg.detect_denoise else "none",
        }
        if cfg.detect_denoise:
            denoise_started = time.perf_counter()
            detection_input = median_blur(to_grayscale(original), ksize=5)
            detection_parameters["ksize"] = 5
            _record_step(steps, "detect_denoise", {"method": "median", "ksize": 5}, denoise_started)
        started = time.perf_counter()
        detection = detect_document(detection_input, strategy=cfg.detection_strategy, canny_mode=cfg.canny_mode)
        if (
            cfg.detection_fallback
            and cfg.detection_strategy == "canny"
            and detection.status != "ok"
        ):
            primary = detection
            fallback_started = time.perf_counter()
            fallback = detect_document(detection_input, strategy="threshold")
            h, w = original.shape[:2]
            fallback_area_ratio = 0.0
            touches_frame = False
            if fallback.corners is not None:
                fallback_area_ratio = abs(
                    float(cv2.contourArea(fallback.corners.astype(np.float32)))
                ) / float(h * w)
                margin = max(2.0, min(h, w) * 0.01)
                touches_frame = bool(
                    np.any(fallback.corners[:, 0] <= margin)
                    or np.any(fallback.corners[:, 1] <= margin)
                    or np.any(fallback.corners[:, 0] >= w - 1 - margin)
                    or np.any(fallback.corners[:, 1] >= h - 1 - margin)
                )
            fallback_accepted = (
                fallback.status == "ok"
                and fallback.corners is not None
                and not touches_frame
                and fallback_area_ratio <= MAX_CONFIDENT_AREA_RATIO
            )
            if fallback_accepted:
                detection = fallback
                warnings.append("Canny tespiti başarısız oldu; threshold yedeği kullanıldı.")
            else:
                reason = (
                    f"Canny başarısız ({primary.reason}); threshold yedeği güvenli aday vermedi"
                )
                if fallback.corners is not None:
                    reason += (
                        f" (kenar_yakını={touches_frame}, alan_oranı={fallback_area_ratio:.3f})"
                    )
                detection = DetectionResult(
                    fallback.corners,
                    "low_confidence" if fallback.corners is not None else "not_found",
                    fallback.score,
                    reason,
                    "threshold_fallback",
                    fallback.debug,
                )
            _record_step(
                steps,
                "detect_fallback",
                {
                    "primary": primary.strategy,
                    "primary_status": primary.status,
                    "fallback": "threshold",
                    "fallback_status": fallback.status,
                    "accepted": fallback_accepted,
                    "area_ratio": fallback_area_ratio,
                    "touches_frame": touches_frame,
                },
                fallback_started,
            )
    else:
        started = time.perf_counter()
        detection = DetectionResult(None, "not_found", 0.0, "tespit devre dışı", cfg.detection_strategy)
        warnings.append("Belge tespiti yapılandırmada devre dışı bırakıldı.")
    _record_step(
        steps,
        "detect",
        {
            "strategy": cfg.detection_strategy,
            "final_strategy": detection.strategy,
            "canny_mode": cfg.canny_mode,
            **(detection_parameters if cfg.detect else {}),
        },
        started,
    )

    rectified = False
    rectified_bgr: np.ndarray | None = None
    if cfg.detect and detection.status == "ok" and detection.corners is not None:
        started = time.perf_counter()
        working, _ = four_point_transform(original, detection.corners)
        rectified = True
        if working.ndim == 2:
            rectified_bgr = cv2.cvtColor(working, cv2.COLOR_GRAY2BGR)
        elif working.shape[2] == 4:
            rectified_bgr = cv2.cvtColor(working, cv2.COLOR_BGRA2BGR)
        else:
            rectified_bgr = working.copy()
        _record_step(steps, "rectify", {"output_shape": list(working.shape[:2])}, started)
    elif cfg.detect:
        warnings.append(f"Belge sınırı güvenilir biçimde tespit edilemedi ({detection.status}: {detection.reason}).")
        if not cfg.allow_unrectified:
            _record_step(steps, "rectify_skipped", {"allow_unrectified": False}, time.perf_counter())
            return PipelineResult(original, False, None, None, None, None, detection, steps, info_before, {}, warnings)
        working = original.copy()
    else:
        working = original.copy()

    started = time.perf_counter()
    gray = to_grayscale(working).copy()
    _record_step(steps, "grayscale", {"shape": list(gray.shape)}, started)

    started = time.perf_counter()
    enhance_choice = cfg.enhance
    if cfg.enhance == "auto":
        enhanced, enhance_choice, parameters = _enhance_auto(gray)
    elif cfg.enhance == "stretch":
        enhanced, parameters = contrast_stretch(gray), {"low_pct": 2, "high_pct": 98}
    elif cfg.enhance == "gamma":
        enhanced, parameters = gamma_correction(gray, CONFIGURED_GAMMA), {"gamma": CONFIGURED_GAMMA}
    elif cfg.enhance == "clahe":
        enhanced, parameters = apply_clahe(gray), {"clip_limit": 2.0, "tile_grid_size": [8, 8]}
    else:
        enhanced, parameters = gray.copy(), {}
    _record_step(steps, "enhance", {"method": enhance_choice, **parameters}, started)

    started = time.perf_counter()
    denoise_choice = cfg.denoise
    denoised = enhanced.copy()
    noise_sigma = _noise_sigma(enhanced) if cfg.denoise == "auto" else None
    kernel = scale_odd_parameter(5, enhanced.shape, MIN_FILTER_KERNEL, MAX_FILTER_KERNEL)
    denoise_parameters: dict[str, object] = {}
    if cfg.denoise == "median" or (cfg.denoise == "auto" and noise_sigma is not None and noise_sigma >= AUTO_NOISE_SIGMA_THRESHOLD):
        denoise_choice = "median"
        denoised = median_blur(denoised, ksize=kernel)
        denoise_parameters["ksize"] = kernel
        if cfg.denoise == "auto":
            denoise_choice = "median+bilateral"
            diameter = scale_odd_parameter(9, enhanced.shape, MIN_FILTER_KERNEL, MAX_FILTER_KERNEL)
            denoised = bilateral_filter(denoised, d=diameter, sigma_color=75, sigma_space=75)
            denoise_parameters.update({"d": diameter, "sigma_color": 75, "sigma_space": 75})
    elif cfg.denoise == "bilateral":
        diameter = scale_odd_parameter(9, enhanced.shape, MIN_FILTER_KERNEL, MAX_FILTER_KERNEL)
        denoised = bilateral_filter(denoised, d=diameter, sigma_color=75, sigma_space=75)
        denoise_parameters.update({"d": diameter, "sigma_color": 75, "sigma_space": 75})
    elif cfg.denoise == "auto" and noise_sigma is not None and noise_sigma >= AUTO_NOISE_SIGMA_THRESHOLD / 2:
        denoise_choice = "median+bilateral"
        denoised = median_blur(denoised, ksize=kernel)
        diameter = scale_odd_parameter(9, enhanced.shape, MIN_FILTER_KERNEL, MAX_FILTER_KERNEL)
        denoised = bilateral_filter(denoised, d=diameter, sigma_color=75, sigma_space=75)
        denoise_parameters.update({"ksize": kernel, "d": diameter, "sigma_color": 75, "sigma_space": 75})
    else:
        denoise_choice = "none" if cfg.denoise == "auto" else cfg.denoise
    if noise_sigma is not None:
        denoise_parameters["noise_sigma_estimate"] = noise_sigma
    _record_step(steps, "denoise", {"method": denoise_choice, **denoise_parameters}, started)

    started = time.perf_counter()
    sharpened = denoised
    if cfg.sharpen == "unsharp":
        sharp_kernel = scale_odd_parameter(5, denoised.shape, MIN_FILTER_KERNEL, MAX_FILTER_KERNEL)
        sharpened = unsharp_mask(denoised, ksize=sharp_kernel, sigma=1.0, amount=1.0)
    _record_step(steps, "sharpen", {"method": cfg.sharpen, "ksize": sharp_kernel if cfg.sharpen == "unsharp" else None}, started)

    started = time.perf_counter()
    binary: np.ndarray | None = None
    block_size: int | None = None
    if cfg.binarize == "otsu":
        binary, threshold_value = otsu_threshold(sharpened, invert=True)
        bin_params: dict[str, object] = {"method": "otsu", "threshold": threshold_value, "invert": True}
    elif cfg.binarize in {"adaptive_mean", "adaptive_gaussian"}:
        block_size = scale_odd_parameter(31, sharpened.shape, MIN_ADAPTIVE_BLOCK, MAX_ADAPTIVE_BLOCK)
        method = "mean" if cfg.binarize == "adaptive_mean" else "gaussian"
        binary = adaptive_threshold(sharpened, method=method, block_size=block_size, c=10, invert=True)
        bin_params = {"method": cfg.binarize, "block_size": block_size, "c": 10, "invert": True}
    else:
        bin_params = {"method": "none"}
    _record_step(steps, "binarize", bin_params, started)

    started = time.perf_counter()
    morphology_kernel = scale_odd_parameter(3, sharpened.shape, MIN_FILTER_KERNEL, MAX_FILTER_KERNEL)
    if binary is not None:
        if cfg.morphology in {"opening", "open_close"}:
            binary = opening(binary, ksize=morphology_kernel)
        if cfg.morphology in {"closing", "open_close"}:
            binary = closing(binary, ksize=morphology_kernel)
    _record_step(steps, "morphology", {"method": cfg.morphology, "ksize": morphology_kernel if cfg.morphology != "none" else None}, started)

    final_for_info = binary if binary is not None else sharpened
    started = time.perf_counter()
    info_after = _analysis_info(final_for_info)
    _record_step(steps, "analyze_after", {}, started)
    return PipelineResult(
        original, rectified, gray, enhanced, denoised, binary, detection,
        steps, info_before, info_after, warnings,
        rectified_image=rectified_bgr,
    )


def process_file(
    path: str | Path, out_dir: str | Path, config: PipelineConfig | None = None
) -> PipelineResult:
    """Unicode uyumlu dosyayı işler, mevcut görüntü aşamalarını ve JSON metaverisini kaydeder."""
    source_path = Path(path)
    result = process_image(load_image(source_path), config)
    destination = Path(out_dir)
    destination.mkdir(parents=True, exist_ok=True)
    stem = source_path.stem
    save_image(destination / f"{stem}_original.png", result.original)
    for name in ("gray", "enhanced", "denoised", "binary"):
        image = getattr(result, name)
        if image is not None:
            save_image(destination / f"{stem}_{name}.png", image)
    metadata = {
        "source": str(source_path),
        "config": asdict(config or PipelineConfig()),
        "rectified": result.rectified,
        "detection": {
            "status": result.detection.status,
            "score": result.detection.score,
            "reason": result.detection.reason,
            "strategy": result.detection.strategy,
            "corners": result.detection.corners.tolist() if result.detection.corners is not None else None,
        },
        "steps": result.steps,
        "warnings": result.warnings,
        "info_before": result.info_before,
        "info_after": result.info_after,
    }
    with (destination / f"{stem}_pipeline.json").open("w", encoding="utf-8") as handle:
        json.dump(metadata, handle, ensure_ascii=False, indent=2)
    return result
